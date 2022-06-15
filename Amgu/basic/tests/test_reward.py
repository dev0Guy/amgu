import unittest
from Amgu.reward import (
    queue_length,
    queue_length_squared,
    queue_length_pln,
    delta_queue,
    delta_queue_pln,
    waiting_time,
    delta_waiting_time,
)


class RewardTestCase(unittest.TestCase):
    def test_queue_length(self):
        current_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 4,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
        }
        expected = -9
        res = queue_length(current_meta_data, current_meta_data)
        self.assertEqual(expected, res)

    def test_delta_queue(self):
        current_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 4,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
        }
        prev_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 1,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
        }
        expected = -3
        res = delta_queue(prev_meta_data, current_meta_data)
        self.assertEqual(expected, res)

    def test_queue_length_pln(self):
        current_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 4,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1.5,
        }
        prev_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 1,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1,
        }
        expected = -18
        res = queue_length_pln(prev_meta_data, current_meta_data)
        self.assertEqual(expected, res)

    def test_delta_queue_pln(self):
        current_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 4,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1.5,
        }
        prev_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 1,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1,
        }
        expected = -6
        res = delta_queue_pln(prev_meta_data, current_meta_data)
        self.assertEqual(expected, res)

    def test_queue_length_squared(self):
        current_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 4,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1.5,
        }
        prev_meta_data = {
            "get_lane_waiting_vehicle_count": {
                "r_0": 1,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1,
        }
        expected = -29
        res = queue_length_squared(prev_meta_data, current_meta_data)
        self.assertEqual(expected, res)
        pass

    def test_waiting_time(self):
        current_meta_data = {
            "vehicle_waiting_time": {
                "r_0": 4,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1.5,
        }
        prev_meta_data = {
            "vehicle_waiting_time": {
                "r_0": 1,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1,
        }
        expected = -9
        res = waiting_time(prev_meta_data, current_meta_data)
        self.assertEqual(expected, res)

    def test_delta_waiting_time(self):
        current_meta_data = {
            "vehicle_waiting_time": {
                "r_0": 4,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1.5,
        }
        prev_meta_data = {
            "vehicle_waiting_time": {
                "r_0": 1,
                "r_1": 0,
                "r_2": 2,
                "r_3": 3,
            },
            "action_time": 1,
        }
        expected = -3
        res = delta_waiting_time(prev_meta_data, current_meta_data)
        self.assertEqual(expected, res)
