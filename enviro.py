import cityflow
import gym
import numpy as np
import json


class GymCityFlow(gym.Env):
    
    def __init__(self,t):
        super(GymCityFlow, self).__init__()
        self.config_path = 'examples/1x1/config.json'
        self.intersection_num = 1
        self.number_of_threads = 4
        # define ENV metadata
        # self.action_space = gym.spaces.MultiDiscrete([9 for i in range(self.intersection_num)])
        # self.observation_space = gym.spaces.Box(low=0, high=255,
        #                             shape=(3, 4, 1), dtype=np.float64
        #                         )
        # self.engine = cityflow.Engine(self.config_path,thread_num=self.number_of_threads)
        config = None
        # steps per episode
        self.steps_per_episode = 1000
        self.is_done = False
        self.current_step = 0
        #open cityflow config file into object
        self.config_file = json.load(open(self.config_path))
        #open cityflow roadnet file into object
        self.roadnet = json.load(open(self.config_file['dir'] + self.config_file['roadnetFile']))
        self.flow = json.load(open(self.config_file['dir'] + self.config_file['flowFile']))
        # create cityflow engine
        self.eng = cityflow.Engine(self.config_path, thread_num=1)  

        # create dict of controllable intersections and number of light phases
        self.intersections = {}
        for i in range(len(self.roadnet['intersections'])):
            # check if intersection is controllable
            if self.roadnet['intersections'][i]['virtual'] == False:
                # for each roadLink in intersection store incoming lanes, outgoing lanes and direction in lists
                incomingLanes = []
                outgoingLanes = []
                directions = []
                for j in range(len(self.roadnet['intersections'][i]['roadLinks'])):
                    incomingRoads = []
                    outgoingRoads = []
                    directions.append(self.roadnet['intersections'][i]['roadLinks'][j]['direction'])
                    for k in range(len(self.roadnet['intersections'][i]['roadLinks'][j]['laneLinks'])):
                        incomingRoads.append(self.roadnet['intersections'][i]['roadLinks'][j]['startRoad'] + 
                                            '_' + 
                                            str(self.roadnet['intersections'][i]['roadLinks'][j]['laneLinks'][k]['startLaneIndex']))
                        outgoingRoads.append(self.roadnet['intersections'][i]['roadLinks'][j]['endRoad'] + 
                                            '_' + 
                                            str(self.roadnet['intersections'][i]['roadLinks'][j]['laneLinks'][k]['endLaneIndex']))
                    incomingLanes.append(incomingRoads)
                    outgoingLanes.append(outgoingRoads)

                # add intersection to dict where key = intersection_id
                # value = no of lightPhases, incoming lane names, outgoing lane names, directions for each lane group
                self.intersections[self.roadnet['intersections'][i]['id']] = [
                                                                                [len(self.roadnet['intersections'][i]['trafficLight']['lightphases'])],
                                                                                incomingLanes,
                                                                                outgoingLanes,
                                                                                directions
                                                                            ]

        #setup intersectionNames list for agent actions
        self.intersectionNames = []
        for key in self.intersections:
            self.intersectionNames.append(key)

        #define action space MultiDiscrete()
        actionSpaceArray = []
        for key in self.intersections:
            actionSpaceArray.append(self.intersections[key][0][0])
        # define observation space
        observationSpaceDict = {}
        for key in self.intersections:
            totalCount = 0
            for i in range(len(self.intersections[key][1])):
                totalCount += len(self.intersections[key][1][i])

            intersectionObservation = []
            maxVehicles = len(self.flow)
            for i in range(totalCount):
                intersectionObservation.append([maxVehicles, maxVehicles])

            observationSpaceDict[key] = gym.spaces.MultiDiscrete(intersectionObservation)
        
        self.observation_space = gym.spaces.Dict(observationSpaceDict)
        self.action_space = gym.spaces.MultiDiscrete(actionSpaceArray)

        #Waiting dict for reward function
        self.waiting_vehicles_reward = {}

    def step(self,action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """
        #Check that input action size is equal to number of intersections
        if len(action) != len(self.intersectionNames):
            raise Warning('Action length not equal to number of intersections')
        #Set each trafficlight phase to specified action
        for idx in range(len(self.intersectionNames)):
            self.eng.set_tl_phase(self.intersectionNames[idx], action[idx])
        
        self.eng.next_step()
        
        self.observation = self._get_observation()
        self.reward = self._get_reward()
        
        self.current_step += 1
        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True
        return self.observation, self.reward , self.is_done, {}
    
    def reset(self):
        """Reset Simulation to start position(from json files)"""
        self.eng.reset(False)
        self.is_done = False
        self.current_step = 0
        return np.array(self._get_observation())
        
    def _get_observation(self):
        #get arrays of waiting cars on input lane vs waiting cars on output lane for each intersection
        self.lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        self.observation = {}
        for intersection_id in self.intersections:
            waitingIntersection=[]
            for i in range(len(self.intersections[intersection_id][1])):
                for j in range(len(self.intersections[intersection_id][1][i])):
                    waitingIntersection.append(np.array([self.lane_waiting_vehicles_dict[self.intersections[intersection_id][1][i][j]], 
                        self.lane_waiting_vehicles_dict[self.intersections[intersection_id][2][i][j]]]))
        return np.array(self.observation)
    
    
    def render(self):
        pass
    
    def close(self):
        pass

    def seed(self,seed=0):
        pass

# class GymCityFlow(gym.Env):
    
#     def __init__(self, config):
#         # used var
#         configPath = 'examples/1x1/config.json'
#         self.action_space = gym.space.Box(0,50,shape=(1,), dtype=np.float32)
#         self.observation_space = gym.space.Box(0,50,shape=(1,), dtype=np.float32)
#         #steps per episode
#         self.steps_per_episode = 1_000
#         self.is_done = False
#         self.current_step = 0
#         #open cityflow config file into dict
#         self.configDict = json.load(open(configPath))
#         #open cityflow roadnet file into dict
#         self.roadnetDict = json.load(open(self.configDict['dir'] + self.configDict['roadnetFile']))
#         self.flowDict = json.load(open(self.configDict['dir'] + self.configDict['flowFile']))
#         # create dict of controllable intersections and number of light phases
#         self.intersections = {}
#         for i in range(len(self.roadnetDict['intersections'])):
#             # check if intersection is controllable
#             if self.roadnetDict['intersections'][i]['virtual'] == False:
#                 # for each roadLink in intersection store incoming lanes, outgoing lanes and direction in lists
#                 incomingLanes = []
#                 outgoingLanes = []
#                 directions = []
#                 for j in range(len(self.roadnetDict['intersections'][i]['roadLinks'])):
#                     incomingRoads = []
#                     outgoingRoads = []
#                     directions.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['direction'])
#                     for k in range(len(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'])):
#                         incomingRoads.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['startRoad'] + 
#                                             '_' + 
#                                             str(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'][k]['startLaneIndex']))
#                         outgoingRoads.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['endRoad'] + 
#                                             '_' + 
#                                             str(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'][k]['endLaneIndex']))
#                     incomingLanes.append(incomingRoads)
#                     outgoingLanes.append(outgoingRoads)

#                 # add intersection to dict where key = intersection_id
#                 # value = no of lightPhases, incoming lane names, outgoing lane names, directions for each lane group
#                 self.intersections[self.roadnetDict['intersections'][i]['id']] = [
#                                                                                   [len(self.roadnetDict['intersections'][i]['trafficLight']['lightphases'])],
#                                                                                   incomingLanes,
#                                                                                   outgoingLanes,
#                                                                                   directions
#                                                                                  ]

#         #setup intersectionNames list for agent actions
#         self.intersectionNames = []
#         for key in self.intersections:
#             self.intersectionNames.append(key)

#         #define action space MultiDiscrete()
#         actionSpaceArray = []
#         for key in self.intersections:
#             actionSpaceArray.append(self.intersections[key][0][0])
#         self.action_space = gym.spaces.MultiDiscrete(actionSpaceArray)

#         # define observation space
#         observationSpaceDict = {}
#         for key in self.intersections:
#             totalCount = 0
#             for i in range(len(self.intersections[key][1])):
#                 totalCount += len(self.intersections[key][1][i])

#             intersectionObservation = []
#             maxVehicles = len(self.flowDict)
#             for i in range(totalCount):
#                 intersectionObservation.append([maxVehicles, maxVehicles])

#             observationSpaceDict[key] = gym.spaces.MultiDiscrete(intersectionObservation)
#         self.observation_space = gym.spaces.Dict(observationSpaceDict)

#         # create cityflow engine
#         self.eng = cityflow.Engine(configPath, thread_num=1)  

#         #Waiting dict for reward function
#         self.waiting_vehicles_reward = {}
    
#     def step(self, action):
#         """

#         Parameters
#         ----------
#         action :

#         Returns
#         -------
#         ob, reward, episode_over, info : tuple
#             ob (object) :
#                 an environment-specific object representing your observation of
#                 the environment.
#             reward (float) :
#                 amount of reward achieved by the previous action. The scale
#                 varies between environments, but the goal is always to increase
#                 your total reward.
#             episode_over (bool) :
#                 whether it's time to reset the environment again. Most (but not
#                 all) tasks are divided up into well-defined episodes, and done
#                 being True indicates the episode has terminated. (For example,
#                 perhaps the pole tipped too far, or you lost your last life.)
#             info (dict) :
#                 diagnostic information useful for debugging. It can sometimes
#                 be useful for learning (for example, it might contain the raw
#                 probabilities behind the environment's last state change).
#                 However, official evaluations of your agent are not allowed to
#                 use this for learning.
#         """
#         #Check that input action size is equal to number of intersections
#         if len(action) != len(self.intersectionNames):
#             raise Warning('Action length not equal to number of intersections')
#         #Set each trafficlight phase to specified action
#         for idx in range(len(self.intersectionNames)):
#             self.eng.set_tl_phase(self.intersectionNames[idx], action[idx])
        
#         self.eng.next_step()
        
#         self.observation = self._get_observation()
#         self.reward = self._get_reward()
        
#         self.current_step += 1
#         if self.current_step + 1 == self.steps_per_episode:
#             self.is_done = True
#         return self.observation, self.reward , self.is_done, {}
    
#     def reset(self):
#         """Reset Simulation to start position(from json files)"""
#         self.eng.reset(False)
#         self.is_done = False
#         self.current_step = 0
#         return np.array(self._get_observation())
        
#     def _get_observation(self):
#         #get arrays of waiting cars on input lane vs waiting cars on output lane for each intersection
#         self.lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
#         self.observation = {}
#         for intersection_id in self.intersections:
#             waitingIntersection=[]
#             for i in range(len(self.intersections[intersection_id][1])):
#                 for j in range(len(self.intersections[intersection_id][1][i])):
#                     waitingIntersection.append(np.array([self.lane_waiting_vehicles_dict[self.intersections[intersection_id][1][i][j]], 
#                         self.lane_waiting_vehicles_dict[self.intersections[intersection_id][2][i][j]]]))
#         # return np.stack(self.observation['intersection_1_1'],axis=1)
#         return np.array([[1.0,2.0]])
    
#     def _get_reward(self):
#         self.vehicle_speeds = self.eng.get_vehicle_speed()
#         self.lane_vehicles = self.eng.get_lane_vehicles()

#         #list of waiting vehicles
#         waitingVehicles = []
#         reward = []

#         #for intersection in dict retrieve names of waiting vehicles
#         for key in self.intersections:
#             for i in range(len(self.intersections[key][1])):
#                 #reward val
#                 intersectionReward = 0
#                 for j in range(len(self.intersections[key][1][i])):
#                     vehicle = self.lane_vehicles[self.intersections[key][1][i][j]]
#                     #if lane is empty continue
#                     if len(vehicle) == 0:
#                             continue
#                     for k in range(len(vehicle)):
#                         #If vehicle is waiting check for it in dict
#                         if self.vehicle_speeds[vehicle[k]] < 0.1:
#                             waitingVehicles.append(vehicle[k])
#                             if vehicle[k] not in self.waiting_vehicles_reward:
#                                 self.waiting_vehicles_reward[vehicle[k]] = 1
#                             else:
#                                 self.waiting_vehicles_reward[vehicle[k]] += 1
#                             #calculate reward for intersection, cap value to -2e+200
#                             if intersectionReward > -1e+200:
#                                 if self.waiting_vehicles_reward[vehicle[k]] < 460:
#                                     intersectionReward += -np.exp(self.waiting_vehicles_reward[vehicle[k]])
#                                 else:
#                                     intersectionReward += -1e-200
#                             else:
#                                 intersectionReward = -1e+200
#             reward.append([key, intersectionReward])

#         waitingVehiclesRemove = []
#         for key in self.waiting_vehicles_reward:
#             if key in waitingVehicles:
#                 continue
#             else:
#                 waitingVehiclesRemove.append(key)

#         for item in waitingVehiclesRemove:
#             self.waiting_vehicles_reward.pop(item)
        
#         return sum(map(lambda info: info[1], reward))

#     def render(self, mode='human'):
#             print("Current time: " + self.eng.get_current_time())

#     def seed(self, seed=None):
#         """ Set the seed of cityflow simulation

#         Args:
#             seed (int): with this seed the simulation will generate random numbers
#         """
#         self.eng.set_random_seed(seed)