import json
import math
import os
from typing import Tuple
import numpy as np
import gym
import cityflow
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from utils import Rewards


class AgentCityFlowWitingLanes(gym.Env):
    """_summary_

    Args:
        gym (_type_): _description_

    Raises:
        Warning: _description_

    Returns:
        _type_: _description_
    """
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
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir,config_path)
        config =  json.load(open(config_path))
        roadnet_path =  os.path.join(config['dir'],config['roadnetFile'])
        flow_path =  os.path.join(config['dir'],config['flowFile'])
        roadnet = json.load(open(roadnet_path))
        flow = json.load(open(flow_path))
        self.eng = cityflow.Engine(config_path, thread_num=2)  
        # get all data from flow
        for flow_info in flow:
            summary['maxSpeed'] = max(summary['maxSpeed'] ,flow_info['vehicle']['maxSpeed'])
            summary['length'] = min(summary['length'] ,flow_info['vehicle']['length'])
            summary['minGap'] = min(summary['minGap'] ,flow_info['vehicle']['minGap'])
        # 
        for idx,intersection in enumerate(roadnet['intersections']):
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
        for inter_id,inter_info  in intersections.items():
            incomingLanes,outgoingLanes = inter_info[1]
            self.road_mapper[inter_id] = np.concatenate((incomingLanes,outgoingLanes),axis=0).flatten()
        counter = np.array([ np.array([info[1][0].size,info[1][1].size]) for info in intersections.values()])
        in_lane,out_lane = np.max(counter,axis=0)
        summary['inLanes'] = in_lane
        summary['outLanes'] = out_lane
        summary['division'] = math.ceil(summary['size']/(summary['length'] + summary['minGap']))
        # set state size
        self.state_shape = (len(intersections),(in_lane+out_lane))
        return intersections, actionSpaceArray, intersectionNames, summary

    def __init__(self, config):
        config = config or {}
        #steps per episode
        self.steps_per_episode = config.get('steps_per_episode',500)
        config_path = config.get('config_path','examples/1x1/config.json')
        self.reward_func = config.get('reward_func','waiting_count')
        self.reward_func = Rewards.get(self.reward_func)
        self.is_done = False
        self.current_step = 0
        self.intersections = [] # id => [number of actions, incomings, outgoings,directions]
        self.actionSpaceArray = []
        self.summary = {}
        self.road_mapper = {}
        self.channel_num = 1
        # scrape all information from json files
        self.intersections, self.actionSpaceArray,self.intersectionNames, self.summary = self._preprocess(config_path)
        # create spaces
        self.state_shape = self.state_shape
        self.observation_space = gym.spaces.Box(np.zeros(self.state_shape),np.zeros(self.state_shape)+255,dtype=np.float64)  
        self.action_space = gym.spaces.MultiDiscrete(self.actionSpaceArray)
        
        # create cityflow engine
        self.observation = self.reset()

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
        print(f'Current time: {self.eng.get_current_time()}')

    def _get_observation(self):
        info = {}
        info['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}     
        state = np.zeros(self.state_shape)
        for row,intersection in enumerate(self.intersections.values()):
            roads = np.concatenate((intersection[1][0].T,intersection[1][1].T),axis=0)
            lanes = roads.ravel()
            for col,lane in enumerate(lanes):
                state[row,col] = info['lane_waiting_vehicle_count'][lane] / self.summary['division']      
        return state

    def _get_reward(self):
        reward = 0
        for idx,name in enumerate(self.intersectionNames):
            reward += self.reward_func(self.eng,self.observation[idx],self.road_mapper[name],self.summary)
        return reward
    
    def seed(self, seed=None):
        self.eng.set_random_seed(seed)

class SingleAgentCityFlow(gym.Env):
    """_summary_

    Args:
        gym (_type_): _description_

    Raises:
        Warning: _description_

    Returns:
        _type_: _description_
    """
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
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir,config_path)
        config =  json.load(open(config_path))
        roadnet_path =  os.path.join(config['dir'],config['roadnetFile'])
        flow_path =  os.path.join(config['dir'],config['flowFile'])
        roadnet = json.load(open(roadnet_path))
        flow = json.load(open(flow_path))
        self.eng = cityflow.Engine(config_path, thread_num=2)  

        # get all data from flow
        for flow_info in flow:
            summary['maxSpeed'] = max(summary['maxSpeed'] ,flow_info['vehicle']['maxSpeed'])
            summary['length'] = min(summary['length'] ,flow_info['vehicle']['length'])
            summary['minGap'] = min(summary['minGap'] ,flow_info['vehicle']['minGap'])
        # 
        for idx,intersection in enumerate(roadnet['intersections']):
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
        for inter_id,inter_info  in intersections.items():
            incomingLanes,outgoingLanes = inter_info[1]
            self.road_mapper[inter_id] = np.concatenate((incomingLanes,outgoingLanes),axis=0).flatten()
        counter = np.array([ np.array([info[1][0].size,info[1][1].size]) for info in intersections.values()])
        in_lane,out_lane = np.max(counter,axis=0)
        summary['inLanes'] = in_lane
        summary['outLanes'] = out_lane
        summary['division'] = math.ceil(summary['size']/(summary['length'] + summary['minGap']))
        
        # set state size
        self.state_shape = (self.channel_num,len(intersections)*(in_lane+out_lane),summary['division'])
        return intersections, actionSpaceArray, intersectionNames, summary

    def __init__(self, config):
        config = config or {}
        #steps per episode
        self.steps_per_episode = config.get('steps_per_episode',500)
        config_path = config.get('config_path','examples/1x1/config.json')
        self.reward_func = config.get('reward_func','waiting_count')
        self.reward_func = Rewards.get(self.reward_func)
        self.is_done = False
        self.current_step = 0
        self.intersections = [] # id => [number of actions, incomings, outgoings,directions]
        self.actionSpaceArray = []
        self.summary = {}
        self.road_mapper = {}
        self.channel_num = 3
        # scrape all information from json files
        self.intersections, self.actionSpaceArray,self.intersectionNames, self.summary = self._preprocess(config_path)
        # create spaces
        self.state_shape = self.state_shape
        self.observation_space = gym.spaces.Box(np.zeros(self.state_shape),np.zeros(self.state_shape)+255,dtype=np.float64)  
        self.action_space = gym.spaces.MultiDiscrete(self.actionSpaceArray)
        
        # create cityflow engine
        self.observation = self.reset()

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
        print(f'Current time: {self.eng.get_current_time()}')

    def _get_observation(self):
        info = {}
        info['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        # # info['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        # info['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        info['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        info['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        info['vehicle_distance'] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        # info['current_time'] = self.eng.get_current_time()
        (self.channel_num,len(intersections)*(in_lane+out_lane),summary['division'])

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
                    state[0,row*col,division_idx] = speed / self.summary['maxSpeed']
                    state[1,row*col,division_idx] = int(distance%division_size) / division_size
                    if leader_id:
                        leader_distance = info['vehicle_distance'][vehicle_id]
                        state[2,row*col,division_idx] = (leader_distance - distance) / self.summary['size']              
        return state

    def _get_reward(self):
        reward = 0
        for idx,name in enumerate(self.intersectionNames):
            reward += self.reward_func(self.eng,self.observation[idx],self.road_mapper[name],self.summary)
        return reward
    
    def seed(self, seed=None):
        self.eng.set_random_seed(seed)

class MultiAgentCityFlow(MultiAgentEnv):
    """_summary_

    Args:
        gym (_type_): _description_

    Raises:
        Warning: _description_

    Returns:
        _type_: _description_
    """
    
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
        for inter_id,inter_info  in intersections.items():
            incomingLanes,outgoingLanes = inter_info[1]
            self.road_mapper[inter_id] = np.concatenate((incomingLanes,outgoingLanes),axis=0).flatten()
        counter = np.array([ np.array([info[1][0].size,info[1][1].size]) for info in intersections.values()])
        in_lane,out_lane = np.max(counter,axis=0)
        summary['inLanes'] = in_lane
        summary['outLanes'] = out_lane
        summary['division'] = math.ceil(summary['size'] /(summary['length'] + summary['minGap']))
        # set state size
        self.state_shape = (self.channel_num,(in_lane+out_lane),summary['division'])
        return intersections, actionSpaceArray, intersectionNames, summary

    def __init__(self, config):
        config = config or {}
        #steps per episode
        self.steps_per_episode = config.get('steps_per_episode',500)
        config_path = config.get('config_path','examples/1x1/config.json')
        self.reward_func = config.get('reward_func','waiting_count')
        self.reward_func = Rewards.get(self.reward_func)
        self.is_done = False
        self.current_step = 0
        self.intersections = [] # id => [number of actions, incomings, outgoings,directions]
        self.actionSpaceArray = []
        self.road_mapper = {}
        self.summary = {}
        self.channel_num = 3
        # scrape all information from json files
        self.intersections, self.actionSpaceArray,self.intersectionNames, self.summary = self._preprocess(config_path)
        # create agents ids
        self._agent_ids = [f'agent_{index}' for index in range(len(self.intersections))]
        # create spaces
        self.observation_space = {agent_id:gym.spaces.Box(np.zeros(self.state_shape),np.zeros(self.state_shape) + 255,dtype=np.float64) \
            for agent_id in self._agent_ids}
        self.observation_space = gym.spaces.Dict(self.observation_space)   
        self.action_space = {agent_id:gym.spaces.Discrete(self.actionSpaceArray[idx]) for idx,agent_id in enumerate(self._agent_ids)}
        self.action_space = gym.spaces.Dict(self.action_space)   
        
        # create cityflow engine
        self.eng = cityflow.Engine(config_path, thread_num=2)  
        #Waiting dict for reward function
        self.rewards = {agent_id: 0 for agent_id in self._agent_ids}
        self.observations = self.reset()

    def step(self, action):
        #Check that input action size is equal to number of intersections
        if len(action) != len(self.intersectionNames):
            raise Warning('Action length not equal to number of intersections')
        action = list(action.values())
        #Set each trafficlight phase to specified action
        for i in range(len(self.intersectionNames)):
            self.eng.set_tl_phase(self.intersectionNames[i], action[i])

        #env step
        self.eng.next_step()
        #observation
        self.observations = self._get_observations()

        #reward
        self.rewards = self._get_rewards()
        #Detect if Simulation is finshed for done variable
        self.current_step += 1

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        #return observation, reward, done, info
        return self.observations, self.rewards, self.is_done, {}

    def reset(self):
        self.eng.reset(seed=False)
        self.is_done = False
        self.current_step = 0
        return self._get_observations()

    def render(self, mode='human'):
        print(f'Current time: {self.eng.get_current_time()}')

    def _get_observations(self):
        info = {}
        info['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        # # info['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        # info['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        info['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        info['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        info['vehicle_distance'] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        # info['current_time'] = self.eng.get_current_time()
        state_dict = {agent_id: np.zeros(self.state_shape) for agent_id in self._agent_ids}
        division_size = self.summary['length'] + self.summary['minGap']
        for row,intersection in enumerate(self.intersections.values()):
            roads = np.concatenate((intersection[1][0].T,intersection[1][1].T),axis=0)
            lanes = roads.ravel()
            state = state_dict[self._agent_ids[row]]
            for col,lane in enumerate(lanes):
                for vehicle_id in info['lane_vehicles'][lane]:
                    leader_id = self.eng.get_leader(vehicle_id)
                    distance = info['vehicle_distance'][vehicle_id]
                    speed = info['vehicle_speed'][vehicle_id]
                    division_idx = int(distance//division_size)
                    state[0,col,division_idx] = speed / self.summary['maxSpeed']
                    state[1,col,division_idx] = int(distance%division_size) / division_size
                    if leader_id:
                        leader_distance = info['vehicle_distance'][vehicle_id]
                        # leader_speed = info['vehicle_speed'][vehicle_id]
                        state[2,col,division_idx] = (leader_distance - distance) / self.summary['size']              
        return state_dict

    def _get_rewards(self):
        return {agent_id:self.reward_func(self.eng,self.observations[agent_id],self.road_mapper[self.intersectionNames[idx]],self.summary)  for idx, agent_id in enumerate(self._agent_ids)}
    
    def seed(self, seed=None):
        self.eng.set_random_seed(seed)