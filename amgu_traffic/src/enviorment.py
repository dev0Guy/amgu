import math
import types
from collections import namedtuple
from amgu_abstract import RewardWrapper, Preprocessor
import os
import json
import cityflow
import numpy as np
import gym


__all__ = ["CFWrapper", "MultiDiscreteCF", "DiscreteCF"]

Information = namedtuple(
    "Preprocess",
    [
        "eng",
        "road_mapper",
        "state_shape",
        "intersections",
        "actionSpaceArray",
        "action_impact",
        "intersectionNames",
        "summary",
    ],
)


class CFWrapper:
    """Class Integrate With Cityflow & Create The Env From JSON File."""

    def __init__(
        self, config: dict, reward_class: RewardWrapper, preprocess_dict: dict
    ):
        # Asserts
        assert "steps_per_episode" in config
        assert "config_path" in config
        assert "res_path" in config
        # assert 'channel_num' in config
        # assert 'cooldown' in config
        assert issubclass(reward_class, RewardWrapper)
        assert (
            isinstance(preprocess_dict, dict)
            and "func" in preprocess_dict
            and "argument_list" in preprocess_dict
        )
        assert isinstance(preprocess_dict["func"], types.FunctionType)
        # get information from config
        self.eps_num_steps = config["steps_per_episode"]
        self.res_path = config["res_path"]
        file_name = config["config_path"].split("/")[-1]
        idx_last = config["config_path"].rindex(file_name)
        assert idx_last != -1
        self.sub_folder = config["config_path"][:idx_last]
        self.cfg_path = config["config_path"]
        print("=" * 50)
        print(self.cfg_path)
        print("=" * 50)

        self.chl_num = config.get("channel_num", 4)
        self.cooldown = config.get("cooldown", 6)
        self.information = self._extract_information()
        self.reward_func = reward_class(self).get
        self.preprocess = None
        # distruct arguments
        (
            self.eng,
            self.road_mapper,
            self.org_state_shape,
            self.intersections,
            self.actionSpaceArray,
            self.action_impact,
            self.intersectionNames,
            self.summary,
        ) = self.information
        # define information
        self.is_done: bool = False
        self.last_is_empty = False
        self.prev_action = False
        self.current_step = 0
        self.count_zero_frames = 0
        self.observation = self._reset()
        self.preprocess = Preprocessor(self.observation, preprocess_dict)
        self.state_shape = self.preprocess.transform(self.observation).shape

    def _extract_information(self):
        """Return all information from env as Inforamtion object"""
        # _____ init data _____
        action_impact: list = (
            []
        )  # for each intersection each light phase effect what lane
        intersections: dict = {}
        road_mapper: dict = {}
        summary: dict = {
            "maxSpeed": 0,
            "length": 10,
            "minGap": 5,
            "size": 300,
        }
        config_path = self.cfg_path
        # _____ load files from global path _____
        config = json.load(open(config_path))
        dir = os.path.join(self.sub_folder, config["dir"])
        roadnet_path = os.path.join(dir, config["roadnetFile"])
        flow_path = os.path.join(dir, config["flowFile"])
        # update config file
        config["roadnetFile"] = roadnet_path
        config["flowFile"] = flow_path
        config["roadnetLogFile"] = os.path.join(self.res_path, "roadnet.json")
        config["replayLogFile"] = os.path.join(self.res_path, "replay.json")
        config_path = os.path.join(self.sub_folder, "config_copy.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        roadnet = json.load(open(roadnet_path))
        flow = json.load(open(flow_path))
        # _____ difine ENV _____
        eng = cityflow.Engine(config_path, thread_num=2)
        # _____ Flow Data into dict _____
        for flow_info in flow:
            summary["maxSpeed"] = max(
                summary["maxSpeed"], flow_info["vehicle"]["maxSpeed"]
            )
            summary["length"] = min(summary["length"], flow_info["vehicle"]["length"])
            summary["minGap"] = min(summary["minGap"], flow_info["vehicle"]["minGap"])
        # _____ Roadnet Data _____
        for _, intersection in enumerate(roadnet["intersections"]):
            # Controlled by script
            if not intersection["virtual"]:
                incomingLanes: list = []
                outgoingLanes: list = []
                directions: list = []
                # run on roads
                for road_link in intersection["roadLinks"]:
                    incomingRoads: list = []
                    outgoingRoads: list = []
                    directions.append(road_link["direction"])
                    # run on lanes (add start and end)
                    for lane_link in road_link["laneLinks"]:
                        incomingRoads.append(
                            road_link["startRoad"]
                            + "_"
                            + str(lane_link["startLaneIndex"])
                        )
                        outgoingRoads.append(
                            road_link["endRoad"] + "_" + str(lane_link["endLaneIndex"])
                        )
                    incomingLanes.append(incomingRoads)
                    outgoingLanes.append(outgoingRoads)
                lane_to_phase = dict()
                for phase, traffic_light_phase in enumerate(
                    intersection["trafficLight"]["lightphases"]
                ):
                    for _, lane_link in enumerate(
                        traffic_light_phase["availableRoadLinks"]
                    ):
                        lane_to_phase[lane_link] = phase
                incomingLanes = np.array(incomingLanes)
                outgoingLanes = np.array(outgoingLanes)
                action_impact.append(lane_to_phase)
                # summary of all input in intesection id
                intersections[intersection["id"]] = [
                    len(intersection["trafficLight"]["lightphases"]),
                    (incomingLanes, outgoingLanes),
                    directions,
                ]
        # setup intersectionNames list for agent actions
        intersectionNames: list = []
        actionSpaceArray: list = []
        for id, info in intersections.items():
            intersectionNames.append(id)
            actionSpaceArray.append(info[0])
        for inter_id, inter_info in intersections.items():
            incomingLanes, outgoingLanes = inter_info[1]
            road_mapper[inter_id] = np.concatenate(
                (incomingLanes, outgoingLanes), axis=0
            ).flatten()
        counter = np.array(
            [
                np.array([info[1][0].size, info[1][1].size])
                for info in intersections.values()
            ]
        )
        in_lane, out_lane = np.max(counter, axis=0)
        summary["inLanes"] = in_lane
        summary["outLanes"] = out_lane
        summary["division"] = math.ceil(
            summary["size"] / (summary["length"] + summary["minGap"])
        )
        # define state size
        state_shape = (
            self.chl_num,
            len(intersections),
            (in_lane + out_lane),
            summary["division"],
        )
        return Information(
            eng,
            road_mapper,
            state_shape,
            intersections,
            actionSpaceArray,
            action_impact,
            intersectionNames,
            summary,
        )

    def _activate_action(self, action):
        """Activate action on env.
        Args:
            action (np.array): the action for each intersection.
        """
        for i in range(len(self.intersectionNames)):
            discrete_action = action
            if type(action) != int and type(action) != np.int32:
                discrete_action = action[i]
            self.eng.set_tl_phase(self.intersectionNames[i], discrete_action)
        self.eng.next_step()

    def _activate_cooldown(self):
        for i in range(len(self.intersectionNames)):
            self.eng.set_tl_phase(self.intersectionNames[i], 0)
        for i in range(self.cooldown):
            self.eng.next_step()

    def step_information(self, action: np.array):
        """Activate Step In Env,
            Extract all
            Have to be activate before each step

        Args:
            action (np.array): the action to activate.

        Raises:
            Warning: if number of action is not equal to number of intersection

        """
        # Check that input action size is equal to number of intersections
        if (type(action) != int and type(action) != np.int32) and len(action) != len(
            self.intersectionNames
        ):
            raise Warning("Action length not equal to number of intersections")
        # Set each trafficlight phase to specified action
        self._activate_action(action)
        if action != self.prev_action:
            self._activate_cooldown()

        self.current_step += 1
        self.prev_action = action
        self.observation = self.get_observation()

        current_empty = len(self.eng.get_vehicles(include_waiting=True)) == 0
        self.count_zero_frames = self.count_zero_frames + 1 if current_empty else 0
        # self.count_repeat_frames = self.count_repeat_frames + 1 if np.array_equal(self.observation[3],self.prev_waiting_state[3]) else 0
        self.prev_waiting_state = self.observation
        self.last_is_empty = current_empty

    def _step(self, action: np.array):
        self.step_information(action)
        self.reward = self.reward_func(self.observation) / len(self.intersections)
        more_then_max = self.current_step >= self.eps_num_steps
        finish_run = self.count_zero_frames >= 15
        self.is_done = more_then_max or finish_run
        if self.is_done:
            self.reward += 100 if more_then_max else -100
        return self.observation, self.reward, self.is_done, {}

    def _reset(self):
        """Reset ENV For next Episode.

        Returns:
            np.ndarray: obs after reset.
        """
        self.eng.reset(seed=420)
        self.is_done = False
        self.current_step = 0
        self.count_repeat_frames = 0
        self.count_zero_frames = 0
        self.last_is_empty = False
        self.observation = self.get_observation()
        self.prev_waiting_state = self.observation
        return self.observation

    def get_att(self):
        """Return the Avrage Travel Time.
        Returns:
            float: avg travel time.
        """
        return self.eng.get_average_travel_time()

    def get_ql(self):
        """Return's Qeueue Length for each lane.
        Returns:
            list: the queue length for each lane.
        """
        return list(self.eng.get_lane_vehicle_count().values())

    def get_observation(self):
        info = {}
        info[
            "lane_vehicle_count"
        ] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        # # info['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        # info['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        info[
            "lane_vehicles"
        ] = (
            self.eng.get_lane_vehicles()
        )  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        info[
            "vehicle_speed"
        ] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        info[
            "vehicle_distance"
        ] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        # info['current_time'] = self.eng.get_current_time()
        state = np.zeros(self.org_state_shape)
        division_size = self.summary["length"] + self.summary["minGap"]
        for row, intersection in enumerate(self.intersections.values()):
            roads = np.concatenate((intersection[1][0].T, intersection[1][1].T), axis=0)
            lanes = roads.ravel()
            for col, lane in enumerate(lanes):
                for vehicle_id in info["lane_vehicles"][lane]:
                    leader_id = self.eng.get_leader(vehicle_id)
                    distance = info["vehicle_distance"][vehicle_id]
                    speed = info["vehicle_speed"][vehicle_id]
                    division_idx = int(distance // division_size)
                    norm_speed = speed / self.summary["maxSpeed"]
                    state[0, row, col, division_idx] = norm_speed
                    state[1, row, col, division_idx] = (
                        int(distance % division_size) / division_size
                    )
                    state[3, row, col, division_idx] = norm_speed == 0
                    if leader_id:
                        leader_distance = info["vehicle_distance"][vehicle_id]
                        state[2, row, col, division_idx] = (
                            leader_distance - distance
                        ) / self.summary["size"]
        state = state * 255
        print("#" * 50)
        print(np.max(state))
        print("#" * 50)
        return self.preprocess.transform(state) if self.preprocess else state


class MultiDiscreteCF(CFWrapper, gym.Env):
    def __init__(
        self, config: dict, reward_class: RewardWrapper, preprocess_dict: dict
    ):
        gym.Env.__init__(config)
        CFWrapper.__init__(self, config, reward_class, preprocess_dict)
        self.observation_space: gym.spaces = gym.spaces.Box(
            np.zeros(self.state_shape),
            np.zeros(self.state_shape) + 255,
            dtype=np.float64,
        )
        self.action_space: gym.spaces = gym.spaces.MultiDiscrete(self.actionSpaceArray)

    def reset(self):
        """Reset ENV For next Episode.
        Returns:
            np.ndarray: obs after reset.
        """
        return self._reset()

    def step(self, action: np.array):
        return self._step(action)


class DiscreteCF(gym.Env, CFWrapper):
    def __init__(
        self, config: dict, reward_class: RewardWrapper, preprocess_dict: dict
    ):
        CFWrapper.__init__(self, config, reward_class, preprocess_dict)
        gym.Env.__init__(config)
        self.observation_space: gym.spaces = gym.spaces.Box(
            np.zeros(self.state_shape),
            np.zeros(self.state_shape) + 255,
            dtype=np.float64,
        )
        if len(self.actionSpaceArray) > 1:
            raise Exception("Have to have only one intersection")
        self.action_space: gym.spaces = gym.spaces.Discrete(self.actionSpaceArray[0])

    def reset(self):
        """Reset ENV For next Episode.
        Returns:
            np.ndarray: obs after reset.
        """
        return self._reset()

    def step(self, action: np.array):
        return self._step(action)
