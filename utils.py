import numpy as np

class Rewards:
    
    @staticmethod
    def waiting_count(eng,intersection_state,roads,summary):
        waiting = eng.get_lane_waiting_vehicle_count()
        sum = 0
        for lane in roads:
            sum += waiting[lane]
        return -sum
    
    @staticmethod
    def avg_travel_time(eng,intersection_state,roads,summary):
        return -eng.get_average_travel_time()
    
    @staticmethod
    def delay_from_opt(eng,intersection_state,roads,summary):
        v_num = 0
        d = np.zeros(len(roads))
        count_dict = eng.get_lane_vehicle_count()
        speed_dict = eng.get_vehicle_speed()
        lane_v = eng.get_lane_vehicles()
        for idx,lane in enumerate(roads):
            if lane in count_dict and lane in lane_v:
                for vehicle_id in lane_v[lane]:
                    d[idx]+= max(0,1 -speed_dict[vehicle_id] / summary['maxSpeed'])
                v_num += count_dict[lane]
        return - np.sum(d) / v_num
    
    @staticmethod
    def exp_delay_from_opt(eng,intersection_state,roads,summary):
        C = 1.45
        v_num = 0
        val = np.zeros(len(roads))
        count_dict = eng.get_lane_vehicle_count()
        speed_dict = eng.get_vehicle_speed()
        lane_v = eng.get_lane_vehicles()
        dist_v = eng.get_vehicle_distance()
        for idx,lane in enumerate(roads):
            if lane in count_dict and lane in lane_v:
                for vehicle_id in lane_v[lane]:
                    leader = eng.get_leader(vehicle_id) 
                    w = dist_v[vehicle_id]
                    w -= dist_v[vehicle_id] if leader != "" else 0
                    d = max(0,1 -speed_dict[vehicle_id] / summary['maxSpeed'])
                    val[idx] += C ** (w*d)
                v_num += count_dict[lane]
        return - (np.sum(val)-1)/ v_num
    
    @staticmethod
    def get(name):
        if name ==  'waiting_count':
            return Rewards.waiting_count
        elif name == 'avg_travel_time':
            return Rewards.avg_travel_time
        elif name == 'delay_from_opt':
            return Rewards.delay_from_opt
        elif name == 'exp_delay_from_opt':
            return Rewards.exp_delay_from_opt
