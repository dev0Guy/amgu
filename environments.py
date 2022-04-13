import cityflow
import gym
import numpy as np
import json

class GymCityFlow(gym.Env):
    
    def __init__(self, config):
        # used var
        configPath = config['configPath']
        #steps per episode
        self.steps_per_episode = config['episodeSteps']
        self.is_done = False
        self.current_step = 0
        #open cityflow config file into dict
        self.configDict = json.load(open(configPath))
        #open cityflow roadnet file into dict
        self.roadnetDict = json.load(open(self.configDict['dir'] + self.configDict['roadnetFile']))
        self.flowDict = json.load(open(self.configDict['dir'] + self.configDict['flowFile']))
        # create dict of controllable intersections and number of light phases
        self.intersections = {}
        for (_,intersection) in enumerate(self.roadnetDict['intersections']):
            print(json.dumps(intersection, indent=4, sort_keys=True))
            # check if intersection is controllable
            if intersection['virtual'] == False:
                # for each roadLink in intersection store incoming lanes, outgoing lanes and direction in lists
                incomingLanes = []
                outgoingLanes = []
                directions = []
                for (_,road_link)in enumerate(intersection['roadLinks']):
                    incomingRoads = []
                    outgoingRoads = []
                    directions.append(road_link['direction'])
                    for (_,road_link) in enumerate(road_link['laneLinks']):
                        print(json.dumps(road_link, indent=4, sort_keys=True))
                        incomingRoads.append(road_link['startRoad'] + 
                                            '_' + 
                                            str(road_link['startLaneIndex']))
                        outgoingRoads.append(road_link['endRoad'] + 
                                            '_' + 
                                            str(road_link['endLaneIndex']))
                    incomingLanes.append(incomingRoads)
                    outgoingLanes.append(outgoingRoads)
                # add intersection to dict where key = intersection_id
                # value = no of lightPhases, incoming lane names, outgoing lane names, directions for each lane group
                self.intersections[intersection['id']] = [[len(intersection['trafficLight']['lightphases'])], 
                                                                                incomingLanes,
                                                                                outgoingLanes,
                                                                                directions]

        #setup intersectionNames list for agent actions
        self.intersectionNames = []
        for instersection_id in self.intersections:
            self.intersectionNames.append(instersection_id)

        #define action space MultiDiscrete()
        actionSpaceArray = []
        for instersection_id in self.intersections:
            actionSpaceArray.append(self.intersections[instersection_id][0][0])
        self.action_space = gym.spaces.MultiDiscrete(actionSpaceArray)

        # define observation space
        observationSpaceDict = {}
        for key in self.intersections:
            totalCount = 0
            for i in range(len(self.intersections[key][1])):
                totalCount += len(self.intersections[key][1][i])

            intersectionObservation = []
            maxVehicles = len(self.flowDict)
            for i in range(totalCount):
                intersectionObservation.append([maxVehicles, maxVehicles])

            observationSpaceDict[key] = gym.spaces.MultiDiscrete(intersectionObservation)
        self.observation_space = gym.spaces.Dict(observationSpaceDict)

        # create cityflow engine
        self.eng = cityflow.Engine(configPath, thread_num=1)  

        #Waiting dict for reward function
        self.waiting_vehicles_reward = {}
    
    def step(self, action):
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
        self.eng.set_tl_phase(self.intersection_id, action)
        
        self.observation = self._get_observation()
        self.reward = self._get_reward()
        
        self.current_step += 1
        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        return self.observation, self.reward , self.has_done, {}
    
    def reset(self):
        """Reset Simulation to start position(from json files)"""
        self.eng.reset(None)
        self.is_done = False
        self.current_step = 0
        return self._get_observation()
        
    def _get_observation(self):
        #get arrays of waiting cars on input lane vs waiting cars on output lane for each intersection
        self.lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        self.observation = {}
        for intersection_id in self.intersections:
            waitingIntersection=[]
            for i in range(len(self.intersections[intersection_id][1])):
                for j in range(len(self.intersections[intersection_id][1][i])):
                    waitingIntersection.append([self.lane_waiting_vehicles_dict[self.intersections[intersection_id][1][i][j]], 
                        self.lane_waiting_vehicles_dict[self.intersections[intersection_id][2][i][j]]])
            self.observation[intersection_id] = waitingIntersection

        return self.observation
    
    def _get_reward(self):
        # get information from env
        self.vehicle_speeds = self.eng.get_vehicle_speed()
        self.lane_vehicles = self.eng.get_lane_vehicles()
        #list of waiting vehicles
        waitingVehicles = []
        reward = []
        #for intersection in dict retrieve names of waiting vehicles
        for instersection in self.intersections:
            for i in range(len(self.intersections[instersection][1])):
                #reward val
                intersectionReward = 0
                for j in range(len(self.intersections[instersection][1][i])):
                    vehicle = self.lane_vehicles[self.intersections[instersection][1][i][j]]
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
                                if self.waiting_vehicles_reward[vehicle[k]] < 460:
                                    intersectionReward += -np.exp(self.waiting_vehicles_reward[vehicle[k]])
                                else:
                                    intersectionReward += -1e-200
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
    
    def render(self, mode='human'):
            print("Current time: " + self.eng.get_current_time())

    def seed(self, seed=None):
        """ Set the seed of cityflow simulation

        Args:
            seed (int): with this seed the simulation will generate random numbers
        """
        self.eng.set_random_seed(seed)