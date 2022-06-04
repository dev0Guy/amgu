import unittest
from amgu_traffic import model, preprocessor
import numpy as np
import gym
import torch

state = np.zeros((3, 1, 32, 40))
action = gym.spaces.Discrete(9)
seq_lens = 1

action_impact = [{0: 5, 4: 6, 2: 7, 7: 8, 1: 5, 5: 6, 3: 7, 6: 8}]
# s0 = np.zeros((3, 1, 32, 40),dtype=np.int)
# s0[2] += 1


class ModelTest(unittest.TestCase):
    def test_cnn(self):
        input_dict = {"obs": torch.from_numpy(state)[None, :]}
        space = gym.spaces.Box(state, state + 255, dtype=np.float64,)
        seq = model.CNN(space, action, 9, {"custom_model_config": {}}, "CNN")
        self.assertTupleEqual(seq(input_dict, True, seq_lens)[0].size(), (1, 9))

    def test_fcn(self):
        input_dict = {"obs": torch.from_numpy(preprocessor.LaneQeueueLength(state, 2))}
        space = gym.spaces.Box(state[1, :], state[1, :] + 255, dtype=np.float64,)
        seq = model.FCN(
            space,
            action,
            9,
            {"custom_model_config": {"intersection_num": 1, "hidden_size": 10}},
            "FCN",
        )
        self.assertTupleEqual(seq(input_dict, True, seq_lens)[0].size(), (1, 9))

    def test_qeueue(self):
        ql = model.Qeueue(action_impact)
        t = torch.from_numpy(state)
        t = t[None, :]
        best_action = ql(t)
        self.assertEqual(best_action.numpy(), np.array([[0]]))

    def test_random(self):
        ql = model.Qeueue(action_impact)
        t = torch.from_numpy(state)
        t = t[None, :]
        self.assertTupleEqual(ql(t).size(), (1, 1))


if __name__ == "__main__":
    unittest.main()
