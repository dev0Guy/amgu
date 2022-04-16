import json

import numpy as np
import gym
import cityflow

class GymCityFlow(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def _preprocess(self,config_path):
        # local vars init
        intersections = {}
        # load files from config and local
        config =  json.load(open(config_path))
        roadnet = json.load(open(config['dir'] + config['roadnetFile']))
        flow = json.load(open(config['dir'] + config['flowFile']))
        # 
        for intersection in roadnet['intersections']:
            # is controlled by runing script
            if not intersection['virtual']:
                # init local var
                incomingLanes = []
                outgoingLanes = []
                directions = []
                for road_link in intersection['roadLinks']:
                    incomingRoads = []
                    outgoingRoads = []
                    # 
                    directions.append(road_link['direction'])
                    for lane_link in road_link['laneLinks']:
                        incomingRoads.append(road_link['startRoad'] + '_' + str(lane_link['startLaneIndex']))
                        outgoingRoads.append(road_link['endRoad'] + '_' + str(lane_link['endLaneIndex']))
                    incomingLanes.append(incomingRoads)
                    outgoingLanes.append(outgoingRoads)
                incomingLanes = np.array(incomingLanes)
                outgoingLanes = np.array(outgoingLanes)
                
                intersections[intersection['id']] = [
                                                            len(intersection['trafficLight']['lightphases']),
                                                            (incomingLanes,outgoingLanes),
                                                            directions
                                                        ]
        
        #setup intersectionNames list for agent actions
        intersectionNames = []
        actionSpaceArray = []
        for id,info in intersections.items():
            intersectionNames.append(id)
            actionSpaceArray.append(info[0])
        counter = [ len(info[1][0]) + len(info[1][1]) for info in intersections.values()] 
        total_col = max(counter)
        state = np.zeros((len(intersections),total_col),dtype=np.float64) + 500
        return intersections, state, actionSpaceArray, intersectionNames

    def __init__(self, config):
        #steps per episode
        self.steps_per_episode = 1_000
        self.is_done = False
        self.current_step = 0
        self.intersections = [] # id => [number of actions, incomings, outgoings,directions]
        self.actionSpaceArray = []
        self.observationSpaceDict = []
        config_path = 'examples/2x3/config.json'
        # scrape all information from json files
        self.intersections, self.observationSpaceDict, self.actionSpaceArray,self.intersectionNames = self._preprocess(config_path)
        print(self.observationSpaceDict)
        # create spaces
        self.state_shape = self.observationSpaceDict.shape
        self.observation_space = gym.spaces.Box(np.zeros(self.observationSpaceDict.shape),self.observationSpaceDict,dtype=np.float64)       
        self.action_space = gym.spaces.MultiDiscrete(self.actionSpaceArray)
        # create cityflow engine
        self.eng = cityflow.Engine(config_path, thread_num=5)  
        #Waiting dict for reward function
        self.waiting_vehicles_reward = {}

    def step(self, action):
        #Check that input action size is equal to number of intersections
        if len(action) != len(self.intersectionNames):
            raise Warning('Action length not equal to number of intersections')

        #Set each trafficlight phase to specified action
        for i in range(len(self.intersectionNames)):
            self.eng.set_tl_phase(self.intersectionNames[i], action[i])

        #env step
        self.eng.next_step()
        #observation
        self.observation = self._get_observation()

        #reward
        self.reward = self._get_reward()
        #Detect if Simulation is finshed for done variable
        self.current_step += 1

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        #return observation, reward, done, info
        return self.observation, self.reward, self.is_done, {}

    def reset(self):
        self.eng.reset(seed=False)
        self.is_done = False
        self.current_step = 0
        return self._get_observation()

    def render(self, mode='human'):
        print("Current time: " + self.cityflow.get_current_time())

    def _get_observation(self):
        info = {}
        info['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        # # info['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        # info['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        # info['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        # info['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        # info['vehicle_distance'] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        # info['current_time'] = self.eng.get_current_time()
        state = np.zeros(self.state_shape)
        # print(self.state_shape)
        for row,intersection in enumerate(self.intersections.values()):
            roads = np.concatenate((intersection[1][0].T,intersection[1][1].T),axis=0)
            lanes = roads.ravel()
            for col,lane in enumerate(lanes):
                pass
                    # print(row,col)
                    # state[row,col] = info['lane_vehicle_count'][lane]        
        return state
        
        
        # for intersection in self.intersections.values():
            # print(len(intersection))
            # for incoming_lanes in intersection[1]:
            #     print(incoming_lanes)
            # for outgoing_lanes in intersection[2]:
            #     print(outgoing_lanes)

            
        # info['current_phase'] = self.current_phase
        # info['current_phase_time'] = self.current_phase_time
        # print(info['lane_vehicle_count'])
        return info
        # state = np.zeros(self.state_shape)
        # for idx,lane_id in enumerate(self.start_lane):
        #     # get all vehicle in lane 
        #     for v_id in info['lane_vehicles'][lane_id]:
        #         index = int(info['vehicle_distance'][v_id]//5)
        #         state[0][idx][index] = info['vehicle_distance'][v_id]%5
        #         state[1][idx][index] =  info['vehicle_speed'][v_id]
        #         # lets normlize 
        #         state[0][idx][index] /=  self.RGB_MAX/self.MAX_DISTANCE 
        #         state[1][idx][index] /=  self.RGB_MAX/self.MAX_SPEED
        #observation
        #get arrays of waiting cars on input lane vs waiting cars on output lane for each intersection
        # self.lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        # observation_dict = {}
        # for key in self.intersections:
        #     waitingIntersection=[]
        #     for i in range(len(self.intersections[key][1])):
        #         for j in range(len(self.intersections[key][1][i])):
        #             waitingIntersection.append(
        #                 np.array(
        #                 [self.lane_waiting_vehicles_dict[self.intersections[key][1][i][j]], 
        #                 self.lane_waiting_vehicles_dict[self.intersections[key][2][i][j]]]))
        #     observation_dict[key] = waitingIntersection
        # tmp = []
        # for _, values in observation_dict.items():
        #     tmp.append(np.concatenate( values, axis=0))
        # self.observation = np.array(tmp)
        # return self.observation
        # return np.array([1,2])

    def _get_reward(self):
        reward = []
        self.vehicle_speeds = self.eng.get_vehicle_speed()
        self.lane_vehicles = self.eng.get_lane_vehicles()
        return -20

    def _get_reward_nonExp(self):
        reward = []
        self.vehicle_speeds = self.eng.get_vehicle_speed()
        self.lane_vehicles = self.eng.get_lane_vehicles()

         #list of waiting vehicles
        waitingVehicles = []
        reward = []

        #for intersection in dict retrieve names of waiting vehicles
        for key in self.intersections:
            for i in range(len(self.intersections[key][1])):
                #reward val
                intersectionReward = 0
                for j in range(len(self.intersections[key][1][i])):
                    vehicle = self.lane_vehicles[self.intersections[key][1][i][j]]
                    #if lane is empty continue
                    if len(vehicle) == 0:
                            continue
                    for k in range(len(vehicle)):
                        #If vehicle is waiting check for it in dict
                        if self.vehicle_speeds[vehicle[k]] < 0.1:
                            waitingVehicles.append(vehicle[k])
                            if vehicle[k] not in self.waiting_vehicles_reward:
                                self.waiting_vehicles_reward[vehicle[k]] = 1
                            else:
                                self.waiting_vehicles_reward[vehicle[k]] += 1
                            #calculate reward for intersection, cap value to -2e+200
                            if intersectionReward > -1e+200:
                                intersectionReward += -(self.waiting_vehicles_reward[vehicle[k]])
                            else:
                                intersectionReward = -1e+200
            reward.append([key, intersectionReward])

        waitingVehiclesRemove = []
        for key in self.waiting_vehicles_reward:
            if key in waitingVehicles:
                continue
            else:
                waitingVehiclesRemove.append(key)

        for item in waitingVehiclesRemove:
            self.waiting_vehicles_reward.pop(item)
        
        return reward

    def seed(self, seed=None):
        self.eng.set_random_seed(seed)