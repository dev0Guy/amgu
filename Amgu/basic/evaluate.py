from .models import _Base, Attack
import collections

META_DATA = collections.namedtuple(
    "MetaData",
    [
        "reward",
        "waiting_time",
        "org_obs",
        "attack_obs",
        "state_division",
    ],
)


def evaluation_generator(env_class, config, models: list[_Base], n_steps=400):
    n_models = len(models)
    envs = [
        env_class(config["env_config"], **config["env_param"]) for _ in range(n_models)
    ]
    for p in range(n_steps):
        yield_lst = []
        for idx, env in enumerate(envs):
            last_obs = env.get_observation()
            save_obs = last_obs
            last_obs, action = models[idx].predict(last_obs, deterministic=True)
            obs, reward, _, _ = env.step(action)
            val = META_DATA(reward, env.eng.get_average_travel_time(),save_obs,last_obs,env.state_division)
            yield_lst.append(val)
        yield tuple(yield_lst)
        if p == 5:
            break
