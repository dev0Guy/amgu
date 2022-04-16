from __future__ import division
import json
import math

import numpy as np
import gym
import cityflow

class GymCityFlow(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def _preprocess(self,config_path):
        # local vars init
        intersections = {}
        summary = {
            'maxSpeed': 0,
            'length': 10,
            'minGap': 5,
            'size': 300,
        }

        # load files from config and local
        config =  json.load(open(config_path))
        roadnet = json.load(open(config['dir'] + config['roadnetFile']))
        flow = json.load(open(config['dir'] + config['flowFile']))
        
        # get all data from flow
        for flow_info in flow:
            summary['maxSpeed'] = max(summary['maxSpeed'] ,flow_info['vehicle']['maxSpeed'])
            summary['length'] = min(summary['length'] ,flow_info['vehicle']['length'])
            summary['minGap'] = min(summary['minGap'] ,flow_info['vehicle']['minGap'])
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
        counter = np.array([ np.array([info[1][0].size,info[1][1].size]) for info in intersections.values()])
        in_lane,out_lane = np.max(counter,axis=0)
        summary['inLanes'] = in_lane
        summary['outLanes'] = out_lane
        summary['division'] = math.ceil(summary['size'] /(summary['length'] + summary['minGap']))
        # set state size
        state = np.zeros((self.channel_num,len(intersections),(in_lane+out_lane),summary['division']),dtype=np.float64) + 255
        return intersections, state, actionSpaceArray, intersectionNames, summary

    def __init__(self, config):
        #steps per episode
        self.steps_per_episode = 1_000
        self.is_done = False
        self.current_step = 0
        self.intersections = [] # id => [number of actions, incomings, outgoings,directions]
        self.actionSpaceArray = []
        self.observationSpaceDict = []
        self.summary = {}
        self.channel_num = 3
        config_path = 'examples/1x1/config.json'
        # scrape all information from json files
        self.intersections, self.observationSpaceDict, self.actionSpaceArray,self.intersectionNames, self.summary = self._preprocess(config_path)
        # create spaces
        self.state_shape = self.observationSpaceDict.shape
        self.observation_space = gym.spaces.Box(np.zeros(self.observationSpaceDict.shape),self.observationSpaceDict,dtype=np.float64)       
        self.action_space = gym.spaces.MultiDiscrete(self.actionSpaceArray)
        # create cityflow engine
        self.eng = cityflow.Engine(config_path, thread_num=2)  
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
        info['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        info['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        info['vehicle_distance'] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        # info['current_time'] = self.eng.get_current_time()
        state = np.zeros(self.state_shape)
        division_size = self.summary['length'] + self.summary['minGap']
        for row,intersection in enumerate(self.intersections.values()):
            roads = np.concatenate((intersection[1][0].T,intersection[1][1].T),axis=0)
            lanes = roads.ravel()
            for col,lane in enumerate(lanes):
                for vehicle_id in info['lane_vehicles'][lane]:
                    leader_id = self.eng.get_leader(vehicle_id)
                    distance = info['vehicle_distance'][vehicle_id]
                    speed = info['vehicle_speed'][vehicle_id]
                    division_idx = int(distance//division_size)
                    state[0,row,col,division_idx] = speed / self.summary['maxSpeed']
                    state[1,row,col,division_idx] = int(distance%division_size) / division_size
                    if leader_id:
                        leader_distance = info['vehicle_distance'][vehicle_id]
                        # leader_speed = info['vehicle_speed'][vehicle_id]
                        state[2,row,col,division_idx] = (leader_distance - distance) / self.summary['size']              
        return state

    def _get_reward(self):
        vehicle_num = sum(self.eng.get_lane_vehicle_count().values())
        vehicle_spped = list(self.eng.get_vehicle_speed().values())
        vehicle_spped = 1 - (np.array(vehicle_spped)/ self.summary['maxSpeed'])
        d = np.array(list(map(lambda x: max(0,x), vehicle_spped)))
        return - np.sum((vehicle_num - d)/vehicle_num)
    
    def seed(self, seed=None):
        self.eng.set_random_seed(seed)