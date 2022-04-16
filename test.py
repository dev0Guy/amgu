import ray
from ray.rllib.agents import ppo
# from env import CityFlowGym_1x1
from enviorments import GymCityFlow
ray.init()

trainer = ppo.PPOTrainer(env=GymCityFlow, config={
    "env_config": {},  # config to pass to env class
    "framework": "torch",
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "render_env": True,
    },
})

for _ in range(1):
    trainer.train()
trainer.evaluate()