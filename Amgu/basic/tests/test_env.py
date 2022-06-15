from stable_baselines3.common.env_checker import check_env
from env import CityFlow1D, CityFlow2D
import unittest
import os.path

parent_path = 'example/1x1'
replay_path = f'{parent_path}/res/replay.txt'
roadnet_path = f"{parent_path}/res/roadnet.json"
DEFUALT_DICT = {
    "env_config": {
        "config_path": f"{parent_path}/config.json",
        "steps_per_episode": 400,
        "save_path": f"{parent_path}/res/",
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
        replay_exist = os.path.isfile(replay_path)
        roadnet_exist = os.path.isfile(roadnet_path)
        if not replay_exist or not roadnet_exist:
            self.fail("Outouts file of env dont exist")
        os.remove(replay_path)
        os.remove(roadnet_path)

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
        replay_exist = os.path.isfile(replay_path)
        roadnet_exist = os.path.isfile(roadnet_path)
        if not replay_exist or not roadnet_exist:
            self.fail("Outouts file of env dont exist")
        os.remove(replay_path)
        os.remove(roadnet_path)


    def test_district(self):
        check_env(self.district, warn=False)
        self.check_env_files()

    def test_not_district(self):
        check_env(self.not_district, warn=False)
        self.check_env_files()
