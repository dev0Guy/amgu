from stable_baselines3.common.env_checker import check_env
from Amgu.env import CityFlow1D, CityFlow2D
import unittest
import os.path

DEFUALT_DICT = {
    "env_config": {
        "config_path": "tests/data/config.json",
        "steps_per_episode": 400,
        "save_path": "tests/data/res/",
    },
    "env_param": {"reward_func": lambda x, y: 5},
}


class CityFlow1DTestCase(unittest.TestCase):
    def setUp(self):
        self.district = CityFlow1D(
            DEFUALT_DICT["env_config"], **DEFUALT_DICT["env_param"], district=True
        )
        self.not_district = CityFlow1D(
            DEFUALT_DICT["env_config"], **DEFUALT_DICT["env_param"], district=False
        )

    def check_env_files(self):
        replay_path = "tests/data/res/replay.txt"
        roadnet_path = "tests/data/res/roadnet.json"
        replay_exist = os.path.isfile(replay_path)
        roadnet_exist = os.path.isfile(roadnet_path)
        os.remove(replay_path)
        os.remove(roadnet_path)
        if not replay_exist or not roadnet_exist:
            self.fail("Outouts file of env dont exist")

    def test_district(self):
        check_env(self.district, warn=False)
        self.check_env_files()

    def test_not_district(self):
        check_env(self.not_district, warn=False)
        self.check_env_files()


class CityFlow2DTestCase(unittest.TestCase):
    def setUp(self):
        self.district = CityFlow2D(
            DEFUALT_DICT["env_config"], **DEFUALT_DICT["env_param"], district=True
        )
        self.not_district = CityFlow2D(
            DEFUALT_DICT["env_config"], **DEFUALT_DICT["env_param"], district=False
        )

    def check_env_files(self):
        replay_path = "tests/data/res/replay.txt"
        roadnet_path = "tests/data/res/roadnet.json"
        replay_exist = os.path.isfile(replay_path)
        roadnet_exist = os.path.isfile(roadnet_path)
        os.remove(replay_path)
        os.remove(roadnet_path)
        if not replay_exist or not roadnet_exist:
            self.fail("Outouts file of env dont exist")

    def test_district(self):
        check_env(self.district, warn=False)
        self.check_env_files()

    def test_not_district(self):
        check_env(self.not_district, warn=False)
        self.check_env_files()
