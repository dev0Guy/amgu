import ray
from ray.rllib.agents import ppo
# from env import CityFlowGym_1x1
from enviorments import SingleAgentCityFlow,MultiAgentCityFlow
ray.init()

trainer = ppo.PPOTrainer(env=MultiAgentCityFlow,config={
    "env_config": {
        "config_path":'examples/2x3/config.json',
        "steps_per_episode": 1_000,
    },  # config to pass to env class
    "framework": "torch",
    "evaluation_num_workers": 1,
    "model": {
        "fcnet_hiddens": [3, 72,40],
        "fcnet_activation": "relu",
    },
    "evaluation_config": {
        "render_env": True,
    },
})

for _ in range(10):
    trainer.train()
trainer.save("./res")
trainer.evaluate()