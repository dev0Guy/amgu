from itertools import chain
from abstract import RewardWrapper

__all__ = ['QeueueLength','DeltaQeueueLength','AvgWaitingTime']

class QeueueLength(RewardWrapper):
    def __init__(self,env):
        super().__init__(env)

    def get(self,observation):
        waiting_in_lanes = list(self.env.eng.get_lane_waiting_vehicle_count().values())
        return sum(waiting_in_lanes)

class DeltaQeueueLength(RewardWrapper):
    def __init__(self,env):
        super().__init__(env)
    
    def get(self,observation):
        waiting_in_lanes = list(self.env.eng.get_lane_waiting_vehicle_count().values())
        running_in_lanes = list(self.env.eng.get_lane_waiting_vehicle_count().values())
        return sum(waiting_in_lanes) - sum(running_in_lanes)

class AvgWaitingTime(RewardWrapper):
    def __init__(self,env):
        super().__init__(env)
        self.v_current_intersection = dict()
        self.awaiting_current_intersection = dict()
        
    def get(self,observation):
        vehicles = list(self.env.eng.get_lane_vehicles().values())
        vehicles = list(chain(*vehicles))
        for vehicle in vehicles:
            current_intersection = self.env.eng.get_vehicle_info(vehicle)
            if vehicle in self.v_current_intersection:
                if self.v_current_intersection[vehicle] != current_intersection:
                    self.awaiting_current_intersection[vehicle] = -.5
                    self.v_current_intersection[vehicle] = current_intersection
            else:
                self.awaiting_current_intersection[vehicle] = -.5
                self.v_current_intersection[vehicle] = current_intersection
            self.awaiting_current_intersection[vehicle] += .5
        return sum(list(self.awaiting_current_intersection.values()))
