import unittest
from amgu_traffic import environment
from amgu_abstract import RewardWrapper
import numpy as np


class DefualtReward(RewardWrapper):
    def get(self, observation):
        return 5


DefualtPreprocessDict = {"func": lambda x: x, "argument_list": []}

defualtConfig = {
    "env_config": {
        "config_path": "examples/hangzhou_1x1_bc-tyc_18041607_1h/config.json",
        "steps_per_episode": 200,
        "save_path": f"res/res/",
    }
}


env_multi = environment.MultiDiscreteCF(
    defualtConfig["env_config"], DefualtReward, DefualtPreprocessDict
)

env_dis = environment.DiscreteCF(
    defualtConfig["env_config"], DefualtReward, DefualtPreprocessDict
)


class EnvironmentTest(unittest.TestCase):
    def _test_extract_information(self, env):
        (
            eng,
            road_mapper,
            state_shape,
            intersections,
            actionSpaceArray,
            action_impact,
            intersectionNames,
            summary,
        ) = env.information
        t_road_mapper = {
            "intersection_1_1": np.array(
                [
                    "road_0_1_0_1",
                    "road_0_1_0_1",
                    "road_0_1_0_0",
                    "road_0_1_0_0",
                    "road_1_0_1_1",
                    "road_1_0_1_1",
                    "road_1_0_1_0",
                    "road_1_0_1_0",
                    "road_2_1_2_1",
                    "road_2_1_2_1",
                    "road_2_1_2_0",
                    "road_2_1_2_0",
                    "road_1_2_3_0",
                    "road_1_2_3_0",
                    "road_1_2_3_1",
                    "road_1_2_3_1",
                    "road_1_1_0_0",
                    "road_1_1_0_1",
                    "road_1_1_1_0",
                    "road_1_1_1_1",
                    "road_1_1_1_0",
                    "road_1_1_1_1",
                    "road_1_1_2_0",
                    "road_1_1_2_1",
                    "road_1_1_2_0",
                    "road_1_1_2_1",
                    "road_1_1_3_0",
                    "road_1_1_3_1",
                    "road_1_1_0_0",
                    "road_1_1_0_1",
                    "road_1_1_3_0",
                    "road_1_1_3_1",
                ],
            )
        }
        t_state_shape = (3, 1, 32, 40)
        t_intersections = {
            "intersection_1_1": [
                9,
                (
                    np.array(
                        [
                            ["road_0_1_0_1", "road_0_1_0_1"],
                            ["road_0_1_0_0", "road_0_1_0_0"],
                            ["road_1_0_1_1", "road_1_0_1_1"],
                            ["road_1_0_1_0", "road_1_0_1_0"],
                            ["road_2_1_2_1", "road_2_1_2_1"],
                            ["road_2_1_2_0", "road_2_1_2_0"],
                            ["road_1_2_3_0", "road_1_2_3_0"],
                            ["road_1_2_3_1", "road_1_2_3_1"],
                        ],
                        dtype="<U12",
                    ),
                    np.array(
                        [
                            ["road_1_1_0_0", "road_1_1_0_1"],
                            ["road_1_1_1_0", "road_1_1_1_1"],
                            ["road_1_1_1_0", "road_1_1_1_1"],
                            ["road_1_1_2_0", "road_1_1_2_1"],
                            ["road_1_1_2_0", "road_1_1_2_1"],
                            ["road_1_1_3_0", "road_1_1_3_1"],
                            ["road_1_1_0_0", "road_1_1_0_1"],
                            ["road_1_1_3_0", "road_1_1_3_1"],
                        ],
                        dtype="<U12",
                    ),
                ),
                [0, 0, 1, 1, 2, 2, 3, 3],
            ]
        }
        t_actionSpaceArray = [9]
        t_action_impact = [{0: 5, 4: 6, 2: 7, 7: 8, 1: 5, 5: 6, 3: 7, 6: 8}]
        t_intersectionNames = ["intersection_1_1"]
        t_summary = {
            "maxSpeed": 11.11,
            "length": 5.0,
            "minGap": 2.5,
            "size": 300,
            "inLanes": 16,
            "outLanes": 16,
            "division": 40,
        }
        road_eq = True
        inter_eq = True

        for inter_name in t_road_mapper:
            if inter_name not in road_mapper and len(t_road_mapper[inter_name]) == len(
                road_mapper[inter_name]
            ):
                road_eq = False
                break
            for val1, val2 in zip(t_road_mapper[inter_name], road_mapper[inter_name]):
                road_eq &= val1 == val2
            if not road_eq:
                break
        for inter_name in t_intersections:
            if inter_name not in road_mapper and len(
                t_intersections[inter_name]
            ) == len(intersections[inter_name]):
                inter_eq = False
                break
            x1, (x2, x3), x4 = t_intersections[inter_name]
            y1, (y2, y3), y4 = t_intersections[inter_name]
            inter_eq &= x1 == y1
            inter_eq &= np.all(x2 == y2)
            inter_eq &= np.all(x3 == y3)
            inter_eq &= x4 == y4
            if not inter_eq:
                break
        self.assertEqual(road_eq, True)
        self.assertEqual(inter_eq, True)
        self.assertTupleEqual(t_state_shape, state_shape)
        self.assertEqual(t_actionSpaceArray, actionSpaceArray)
        self.assertEqual(t_action_impact, action_impact)
        self.assertEqual(t_intersectionNames, intersectionNames)
        self.assertEqual(t_summary, summary)

    def _test_step(self, env):
        high = env.observation_space.high
        low = env.observation_space.low
        for i in range(8):
            obs = env.step([i])[0]
            self.assertEqual(np.all(low <= obs) and np.all(obs <= high), True)

    def _test_reset(self, env):
        env.reset()
        low = env.observation_space.low
        self.assertEqual(np.all(low == env.observation), True)

    def test_extract_information(self):
        self._test_extract_information(env_dis)
        self._test_extract_information(env_multi)

    def test_step(self):
        self._test_step(env_dis)
        self._test_step(env_multi)

    def test_reset(self):
        self._test_reset(env_dis)
        self._test_reset(env_multi)


if __name__ == "__main__":
    unittest.main()
