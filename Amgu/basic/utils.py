import collections
import numpy as np
import json
import math
import os
import cityflow


Information = collections.namedtuple(
    "Preprocess",
    [
        "eng",
        "road_mapper",
        "state_shape",
        "intersections",
        "actionSpaceArray",
        "action_impact",
        "intersectionNames",
    ],
)


def extract_information(meta, cfg_path, sub_folder, res_path):
    """Return all information from env as Inforamtion object"""
    # _____ init data _____
    action_impact: list = []  # for each intersection each light phase effect what lane
    intersections: dict = {}
    road_mapper: dict = {}
    config_path = cfg_path
    # _____ load files from global path _____
    config_file = open(config_path)
    config = json.load(config_file)
    dir = os.path.join(sub_folder, config["dir"])
    roadnet_path = os.path.join(dir, config["roadnetFile"])
    flow_path = os.path.join(dir, config["flowFile"])
    # update config file
    config["roadnetFile"] = roadnet_path
    config["flowFile"] = flow_path
    config["roadnetLogFile"] = os.path.join(res_path, "roadnet.json")
    config["replayLogFile"] = os.path.join(res_path, "replay.txt")
    config_path = os.path.join(sub_folder, "config_copy.json")
    with open(config_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    roadnet_file = open(roadnet_path)
    flow_file = open(flow_path)
    roadnet = json.load(roadnet_file)
    flow = json.load(flow_file)
    # _____ difine ENV _____
    eng = cityflow.Engine(config_path, thread_num=2)
    # _____ Flow Data into dict _____
    for flow_info in flow:
        meta["maxSpeed"] = max(meta["maxSpeed"], flow_info["vehicle"]["maxSpeed"])
        meta["length"] = min(meta["length"], flow_info["vehicle"]["length"])
        meta["minGap"] = min(meta["minGap"], flow_info["vehicle"]["minGap"])
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
                        road_link["startRoad"] + "_" + str(lane_link["startLaneIndex"])
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
    meta["inLanes"] = in_lane
    meta["outLanes"] = out_lane
    meta["division"] = math.ceil(meta["size"] / (meta["length"] + meta["minGap"]))
    # define state size
    state_shape = (
        len(intersections),
        (in_lane + out_lane),
    )
    config_file.close()
    flow_file.close()
    roadnet_file.close()
    return Information(
        eng,
        road_mapper,
        state_shape,
        intersections,
        actionSpaceArray,
        action_impact,
        intersectionNames,
    )
