from environments import GymCityFlow

gym_config = {
    'configPath': 'examples/1x1/config.json',
    'episodeSteps': 1_000,
}
env = GymCityFlow(gym_config)
print("="*5 + "  Reset  " + "="*5)
print(env.reset())
print("="*15)
action  = 0
counter = 0 
for i in range(2_000):
    env.step(action)
    counter += 1
    if counter == 50:
        counter = 0
        action = (action + 1) % 4