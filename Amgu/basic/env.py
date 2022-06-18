import gym
import numpy as np
from functools import reduce
from .utils import extract_information


class CityFlow1D(gym.Env):
    """
    CityFlow Simulator,
    Can Run Traffic simulation online and ofline,
    the class implement gym.Env, allow for easy use in third party Libraries
    The Env.State is discibe the following(1-d array) <last_action,last_action_activation_time,..lanes queue length,..lane densaty>.
    """

    #########################
    #   HYPER PARAMATERS    #
    #########################
    META = {
        "maxSpeed": 16.67,
        "length": 10,
        "minGap": 5,
        "size": 300,
    }
    COOLDOWN_TIME = 6
    COOLDOWN_ACTION = 0
    MAX_VAL = 255
    MAX_REPEAT = 10

    #########################
    #      Funcinality      #
    #########################
    def __init__(self, config: dict, reward_func, district=False):
        """_summary_

        Args:
            config (_type_): _description_
            reward_func (_type_): _description_
            district (bool, optional): _description_. Defaults to False.
        """
        self.max_step_c = config["steps_per_episode"]
        self.results = config["save_path"]
        self.seed = config.get("seed", 123)
        self.district = district
        file_name = config["config_path"].split("/")[-1]
        idx_last = config["config_path"].rindex(file_name)
        assert idx_last != -1
        self.sub_folder = config["config_path"][:idx_last]
        self.cfg_path = config["config_path"]
        (
            self.eng,
            self.road_mapper,
            self.org_state_shape,
            self.intersections,
            self.actionSpaceArray,
            self.action_impact,
            self.intersectionNames,
        ) = extract_information(
            CityFlow1D.META, self.cfg_path, self.sub_folder, self.results
        )
        self.reward_func = reward_func
        if self.district:
            self.action_space = gym.spaces.Discrete(self.actionSpaceArray[0])
        else:
            self.action_space = gym.spaces.MultiDiscrete(self.actionSpaceArray)

        intersection_num, max_lane = self.org_state_shape
        self.num_lanes = intersection_num * max_lane
        self.shape = len(self.actionSpaceArray) + 1 + self.num_lanes * 2
        self.observation_space = gym.spaces.Box(
            np.zeros(self.shape) - CityFlow1D.MAX_VAL,
            np.zeros(self.shape) + CityFlow1D.MAX_VAL,
            dtype=np.float64,
        )
        lanes_info_size = reduce((lambda x,y: x * y),self.org_state_shape)
        max_count =  CityFlow1D.META["size"] / (
            CityFlow1D.META["length"] + CityFlow1D.META["minGap"]
        )
        self.state_division = {
            "Multiplyer": CityFlow1D.MAX_VAL,
            "prev_action":  (1 if self.district else len(self.prev_lights), 8),
            "prev_action_duration": (1,self.max_step_c),
            "lane_density": (lanes_info_size,max_count),
            "lane_queue": (lanes_info_size,max_count),
        }
        self.prev_lights = np.zeros(len(self.intersectionNames))
        self.same_action_c = 0
        self.observation = self.reset()
        self.prev_meta_data = self._get_meta_info()

    def _get_meta_info(self):
        """All Information needed to Reward function
        Returns:
            dict: {lane_vehicle_count,lane_vehicles,vehicle_speed,vehicle_distance,get_lane_waiting_vehicle_count,vehicle_waiting_time,action_time}
        """
        vehicle_speed = self.eng.get_vehicle_speed()
        rmv_list = []
        for v_id, speed in vehicle_speed.items():
            if v_id not in self.waiting_dict:
                self.waiting_dict[v_id] = 0
        for v_id, waiting_steps in self.waiting_dict.items():
            if v_id not in vehicle_speed:
                rmv_list.append(v_id)
            else:
                self.waiting_dict[v_id] += 0.5
        # remove all unnecessary v_id (finish run)
        for v_id in rmv_list:
            del self.waiting_dict[v_id]
        return {
            "lane_vehicle_count": self.eng.get_lane_vehicle_count(),  # {lane_id: lane_count, ...}
            "lane_vehicles": self.eng.get_lane_vehicles(),  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
            "vehicle_speed": self.eng.get_vehicle_speed(),  # {vehicle_id: vehicle_speed, ...}
            "vehicle_distance": self.eng.get_vehicle_distance(),  # {vehicle_id: distance, ...}
            "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
            "vehicle_waiting_time": self.waiting_dict,
            "action_time": self.same_action_c * 0.5,
        }

    def get_observation(self):
        """Create & Return Observation from ENV

        Returns:
            np.array: Env State
        """
        meta_info = self._get_meta_info()
        state = np.zeros(self.shape)
        after_prev_light = 1 if self.district else len(self.prev_lights)
        state[:after_prev_light] = self.prev_lights / 8
        state[after_prev_light : after_prev_light + 1] = (
            2 * self.same_action_c / self.max_step_c
        )
        lane_density = np.zeros(self.org_state_shape)
        lane_queue = np.zeros(self.org_state_shape, dtype=np.float64)
        for row, intersection in enumerate(self.intersections.values()):
            roads = np.concatenate((intersection[1][0].T, intersection[1][1].T), axis=0)
            lanes = roads.ravel()
            for col, lane in enumerate(lanes):
                total_awaiting = 0
                count = len(meta_info["lane_vehicles"][lane])
                max_count = CityFlow1D.META["size"] / (
                    CityFlow1D.META["length"] + CityFlow1D.META["minGap"]
                )
                for vehicle_id in meta_info["lane_vehicles"][lane]:
                    speed = meta_info["vehicle_speed"][vehicle_id]
                    norm_speed = speed / CityFlow1D.META["maxSpeed"]
                    total_awaiting += norm_speed == 0
                lane_density[row, col] = 1 - ((max_count - count) / max_count)
                lane_queue[row, col] = total_awaiting / max_count
        _from = after_prev_light + 1
        _to = _from + self.num_lanes
        state[_from:_to] = lane_density.flatten()
        _from = _to
        _to += self.num_lanes
        state[_from:] = lane_queue.flatten()
        return state * CityFlow1D.MAX_VAL

    def reset(self):
        """Reset simulation, all env parameters, return the new state in reseted env
        Returns:
            np.array: Observation
        """
        self.eng.reset(seed=self.seed)
        self.done = False
        self.waiting_dict = {}
        self.empty_roard_c = 0
        self.same_action_c = 0
        self.step_c = 0
        self.prev_lights = np.zeros(len(self.intersectionNames))
        self.observation = self.get_observation()
        self.prev_observation = self.observation
        self.observation = self.get_observation()
        self.prev_meta_data = self._get_meta_info()
        return self.observation

    def step(self, action: gym.spaces):
        """Activate [Action] inside the env[simulation]
            and show impact on return value
        Args:
            action (gym.spaces): _description_

        Returns:
            tuple[np.ndarray,float,bool,dict]: observation, reward, is finish run, info
        """
        same_as_last = action == self.prev_lights
        if np.all(same_as_last == True):
            self.same_action_c += 1
        else:
            self.same_action_c = 0
        is_district = self.district
        for i in range(len(self.intersectionNames)):
            intersection_act = action if is_district else action[i]
            self.eng.set_tl_phase(self.intersectionNames[i], intersection_act)
        self.eng.next_step()
        for sec in range(CityFlow1D.COOLDOWN_TIME):
            for i in range(len(self.intersectionNames)):
                if is_district and not same_as_last:
                    self.eng.set_tl_phase(
                        self.intersectionNames[0], CityFlow1D.COOLDOWN_ACTION
                    )
                elif not is_district and not same_as_last[i]:
                    self.eng.set_tl_phase(
                        self.intersectionNames[i], CityFlow1D.COOLDOWN_ACTION
                    )
            self.eng.next_step()
        self.prev_lights = action
        self.step_c += 1
        if np.all(self.prev_observation == self.observation):
            self.empty_roard_c += 1
        else:
            self.empty_roard_c = 0
        self.prev_observation = self.observation
        self.observation = self.get_observation()
        is_done = (
            self.max_step_c <= self.step_c
        )  # and CityFlow1D.MAX_REPEAT <= self.empty_roard_c
        meta_data = self._get_meta_info()
        reward = self.reward_func(self.prev_meta_data, meta_data)
        self.prev_meta_data = meta_data
        return self.observation, reward, is_done, {}


class CityFlow2D(gym.Env):
    """
    CityFlow Simulator,
    Can Run Traffic simulation online and ofline,
    the class implement gym.Env, allow for easy use in third party Libraries
    The Env.State is discibe the following(4-d array) <Number of information,number of intersections, number of lanes * number of divide>.
    """

    #########################
    #   HYPER PARAMATERS    #
    #########################
    META = {
        "maxSpeed": 16.67,
        "length": 10,
        "minGap": 5,
        "size": 300,
    }
    COOLDOWN_TIME = 6
    COOLDOWN_ACTION = 0
    MAX_VAL = 100
    MAX_REPEAT = 10

    #########################
    #      Funcinality      #
    #########################
    def __init__(self, config: dict, reward_func, district=False):
        """
        Args:
            config (_type_): _description_
            reward_func (_type_): _description_
            district (bool, optional): _description_. Defaults to False.
        """
        self.max_step_c = config["steps_per_episode"]
        self.results = config["save_path"]
        self.seed = config.get("seed", 123)
        self.district = district
        file_name = config["config_path"].split("/")[-1]
        idx_last = config["config_path"].rindex(file_name)
        assert idx_last != -1
        self.sub_folder = config["config_path"][:idx_last]
        self.cfg_path = config["config_path"]
        (
            self.eng,
            self.road_mapper,
            self.org_state_shape,
            self.intersections,
            self.actionSpaceArray,
            self.action_impact,
            self.intersectionNames,
        ) = extract_information(
            CityFlow2D.META, self.cfg_path, self.sub_folder, self.results
        )
        self.reward_func = reward_func
        if self.district:
            self.action_space = gym.spaces.Discrete(self.actionSpaceArray[0])
        else:
            self.action_space = gym.spaces.MultiDiscrete(self.actionSpaceArray)

        intersection_num, max_lane = self.org_state_shape
        self.num_lanes = intersection_num * max_lane
        max_vehichels_in_lane = int(
            self.META["size"] / (self.META["length"] + self.META["minGap"])
        )
        # The shape is [(3 values), amount of intersections, num_lanes, max_vehichels_in_lane
        self.before_shape = (3, intersection_num, self.num_lanes, max_vehichels_in_lane)
        self.shape = (3, intersection_num, self.num_lanes * max_vehichels_in_lane)
        self.observation_space = gym.spaces.Box(
            np.zeros(self.shape) - CityFlow1D.MAX_VAL,
            np.zeros(self.shape) + CityFlow1D.MAX_VAL,
            dtype=np.float64,
        )
        self.prev_lights = np.zeros(len(self.intersectionNames))
        self.same_action_c = 0
        self.eng.reset()

    def _get_meta_info(self):
        """All Information needed to Reward function
        Returns:
            dict: {lane_vehicle_count,lane_vehicles,vehicle_speed,vehicle_distance,get_lane_waiting_vehicle_count,vehicle_waiting_time,action_time}
        """
        vehicle_speed = self.eng.get_vehicle_speed()
        rmv_list = []
        for v_id, speed in vehicle_speed.items():
            if v_id not in self.waiting_dict:
                self.waiting_dict[v_id] = 0
        for v_id, waiting_steps in self.waiting_dict.items():
            if v_id not in vehicle_speed:
                rmv_list.append(v_id)
            else:
                self.waiting_dict[v_id] += 0.5
        # remove all unnecessary v_id (finish run)
        for v_id in rmv_list:
            del self.waiting_dict[v_id]
        return {
            "lane_vehicle_count": self.eng.get_lane_vehicle_count(),  # {lane_id: lane_count, ...}
            "lane_vehicles": self.eng.get_lane_vehicles(),  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
            "vehicle_speed": self.eng.get_vehicle_speed(),  # {vehicle_id: vehicle_speed, ...}
            "vehicle_distance": self.eng.get_vehicle_distance(),  # {vehicle_id: distance, ...}
            "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
            "vehicle_waiting_time": self.waiting_dict,
            "action_time": self.same_action_c * 0.5,
        }

    def get_observation(self):
        """Create & Return Observation from ENV

        Returns:
            np.array: Env State
        """
        meta_info = self._get_meta_info()
        state = np.zeros(self.before_shape)
        division_size = self.META["length"] + self.META["minGap"]
        for row, intersection in enumerate(self.intersections.values()):
            roads = np.concatenate((intersection[1][0].T, intersection[1][1].T), axis=0)
            lanes = roads.ravel()
            for col, lane in enumerate(lanes):
                num_vehicles = len(meta_info["lane_vehicles"][lane])
                for vehicle_id in meta_info["lane_vehicles"][lane]:
                    # distance from intersection
                    distance = meta_info["vehicle_distance"][vehicle_id]
                    speed = meta_info["vehicle_speed"][vehicle_id]
                    division_idx = int(distance // division_size)
                    norm_speed = speed / self.META["maxSpeed"]
                    state[0, row, col, division_idx] = norm_speed
                    state[1, row, col, division_idx] = (
                        int(distance % division_size) / division_size
                    )
                    state[2, row, col, division_idx] = norm_speed < 0.1
        return np.reshape(state, self.shape) * CityFlow2D.MAX_VAL

    def reset(self):
        """Reset simulation, all env parameters, return the new state in reseted env
        Returns:
            np.array: Observation
        """
        self.eng.reset(seed=self.seed)
        self.done = False
        self.waiting_dict = {}
        self.empty_roard_c = 0
        self.same_action_c = 0
        self.step_c = 0
        self.prev_lights = np.zeros(len(self.intersectionNames))
        self.observation = self.get_observation()
        self.prev_observation = self.observation
        self.prev_meta_data = self._get_meta_info()
        return self.observation

    def step(self, action: gym.spaces):
        """Activate [Action] inside the env[simulation]
            and show impact on return value
        Args:
            action (gym.Spaces): _description_

        Returns:
            tuple[np.ndarray,float,bool,dict]: observation, reward, is finish run, info
        """
        same_as_last = action == self.prev_lights
        if np.all(same_as_last == True):
            self.same_action_c += 1
        else:
            self.same_action_c = 0
        is_district = self.district
        for i in range(len(self.intersectionNames)):
            intersection_act = action if is_district else action[i]
            self.eng.set_tl_phase(self.intersectionNames[i], intersection_act)
        self.eng.next_step()
        for sec in range(CityFlow1D.COOLDOWN_TIME):
            for i in range(len(self.intersectionNames)):
                if is_district and not same_as_last:
                    self.eng.set_tl_phase(
                        self.intersectionNames[0], CityFlow1D.COOLDOWN_ACTION
                    )
                elif not is_district and not same_as_last[i]:
                    self.eng.set_tl_phase(
                        self.intersectionNames[i], CityFlow1D.COOLDOWN_ACTION
                    )
            self.eng.next_step()
        self.prev_lights = action
        self.step_c += 1
        if np.all(self.prev_observation == self.observation):
            self.empty_roard_c += 1
        else:
            self.empty_roard_c = 0

        self.prev_observation = self.observation
        self.observation = self.get_observation()
        is_done = (
            self.max_step_c <= self.step_c
        )  # and CityFlow1D.MAX_REPEAT <= self.empty_roard_c
        meta_data = self._get_meta_info()
        reward = self.reward_func(self.prev_meta_data, meta_data)
        self.prev_meta_data = meta_data
        return self.observation, reward, is_done, {}
