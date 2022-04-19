from enviorments import MultiAgentCityFlow,SingleAgentCityFlow

gym_config = {
    'config_path': 'examples/2x3/config.json',
    'episodeSteps': 1_000,
    'reward_func': 'exp_delay_from_opt',
}
env = MultiAgentCityFlow(gym_config)
print("="*5 + "  Reset  " + "="*5)
# print(env.reset())
print("="*15)
action  = 0
counter = 0 
info = env.reset()
# print(json.dumps(env.flow, indent=4, sort_keys=True))
# print(info)
print(env.summary)
for i in range(150):
    # x = env.step([action])
    actions = {f'agent{idx}':action for idx in range(6)}
    x = env.step(actions)
    # print(env.observation_space.contains(x[0]))
    counter += 1
    # print(x[1])
    if counter == 50:
        print(x[1])
        counter = 0
        action = (action + 1) % 4