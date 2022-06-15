from .models import _Base
import collections

DATA = collections.namedtuple(
    "DATA",
    [
        "reward",
        "waiting_time",
    ],
)


def evaluation_generator(env_class, config, models: list[_Base], n_steps=400):
    n_models = len(models)
    envs = [
        env_class(config["env_config"], **config["env_param"]) for _ in range(n_models)
    ]
    for _ in range(n_steps):
        yield_lst = []
        for idx, env in enumerate(envs):
            last_obs = env.get_observation()
            action = models[idx].predict(last_obs, deterministic=True)
            _, reward, _, _ = env.step(action)
            yield_lst.append(DATA(reward, env.eng.get_average_travel_time()))
        yield tuple(yield_lst)
