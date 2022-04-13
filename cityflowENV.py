import gym
import numpy as np

class CityFlowEnvironment(gym.Env):
    
    MAX_SPEED = 16.67
    MAX_DISTANCE = 300
    MAX_CAR_LENGTH = 5
    CHANNEL_NUM = 2 
    RGB_MAX = 2

    def _state_mul(self):
        val = 1 
        for elm in self.state_shape:
            val *= elm
        return val

    def __init__(self, config):
        """
        """
        #get info from config file
        self.eng = cityflow.Engine(config['cityflow_config_file'], thread_num=config['thread_num'])
        self.num_step = config['step_n']
        self.lane_phase_info = config['lane_phase_info']
        self.intersection_id = config["intersection_id"]
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.lane_num = len(self.start_lane)
        self.state_shape =(self.CHANNEL_NUM,len(self.start_lane),self.MAX_DISTANCE//self.MAX_CAR_LENGTH)
        self.state_size = self._state_mul()
        config["state_size"] = self.state_size
        config["state_shape"] = self.state_shape
        self.config = config
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_start_lane"]
        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        # how much time should be between red light (cooldown)
        self.yellow_time = 5
        self.state_store_i = 0
        self.phase_log = []

    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)

    def reset(self):
        """ Reset the simulation env """
        self.eng.reset()
        return self._get_state()

    def step(self, next_phase):
        """
        decide on phase to make in the simulation.
        activate phase and go on frame to future
        Args:
            next_phase (int): next phase action
        """
        has_phase_change = lambda: self.current_phase == next_phase

        if has_phase_change():
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1
        # change simulation action
        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        self.eng.next_step()
        self.phase_log.append(self.current_phase)
        return self.get_state(), self.get_reward() , False, {}

    def _waiting_count(self)-> np.array:
        state_pre = list(self.eng.get_lane_waiting_vehicle_count().values())
        state=np.zeros(8)
        state[0]=state_pre[1]+state_pre[15]
        state[1]=state_pre[3]+state_pre[13]
        state[2]=state_pre[0]+state_pre[14]
        state[3]=state_pre[2]+state_pre[12]
        state[4]=state_pre[1]+state_pre[0]
        state[5]=state_pre[14]+state_pre[15]
        state[6]=state_pre[3]+state_pre[2]
        state[7]=state_pre[12]+state_pre[13]
        return state

    def get_lanes(self):
        state = dict()
        for idx,(key,val) in enumerate(self.eng.get_lane_waiting_vehicle_count().items()):
            state[key] = idx
        return state

    def divide(self,info)->np.array:
        state = np.zeros(self.state_shape)
        for idx,lane_id in enumerate(self.start_lane):
            # get all vehicle in lane 
            for v_id in info['lane_vehicles'][lane_id]:
                index = int(info['vehicle_distance'][v_id]//5)
                state[0][idx][index] = info['vehicle_distance'][v_id]%5
                state[1][idx][index] =  info['vehicle_speed'][v_id]
                # lets normlize 
                state[0][idx][index] /=  self.RGB_MAX/self.MAX_DISTANCE 
                state[1][idx][index] /=  self.RGB_MAX/self.MAX_SPEED
        return state


    def get_state(self)->torch.Tensor:
        """ Return (Tensor): of witing veichle in simulation"""
        info = {}
        info['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        info['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        info['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        info['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        info['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        info['vehicle_distance'] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        info['current_time'] = self.eng.get_current_time()
        info['current_phase'] = self.current_phase
        info['current_phase_time'] = self.current_phase_time

        # state = np.array(list(info['lane_vehicle_count'].values()))
        # state = np.reshape(return_state, [1, self.state_size])
        state = self.divide(info)
        return torch.from_numpy(state).float()

    def get_reward(self):
        """
        """
        lane_vehicle_count = self._waiting_count()
        return -max(lane_vehicle_count)

    def get_score(self):
        """
        """
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -sum(list(lane_waiting_vehicle_count.values()))
        metric = 1 / ((1 + math.exp(-1 * reward)) * self.num_step )
        return metric