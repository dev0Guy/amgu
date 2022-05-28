import unittest
from amgu_traffic import environment


class EnvironmentTest(unittest.TestCase):
    def test_extract_information(self):
        reward = lambda x: x
        config = {}
        preprocess_dict = {}
        # env = environment.CFWrapper()
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

    def test_step_information(self):
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

    def test_step(self):
        pass

    def test_reset(self):
        pass

    def test_get_att(self):
        pass

    def test_get_ql(self):
        pass

    def get_observation(self):
        pass

    def get_get_results(self):
        pass


if __name__ == "__main__":
    unittest.main()
