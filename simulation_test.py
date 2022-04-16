import json
from enviorments import GymCityFlow

gym_config = {
    'configPath': 'examples/2x3/config.json',
    'episodeSteps': 1_000,
}
env = GymCityFlow(gym_config)
print("="*5 + "  Reset  " + "="*5)
# print(env.reset())
print("="*15)
action  = 0
counter = 0 
info = env.reset()
# print(json.dumps(env.flow, indent=4, sort_keys=True))
# print(info)
print(env.summary)
for i in range(51):
    # x = env.step([action,action,action,action,action,action])
    x = env.step([action])
    counter += 1
    if counter == 50:
        print(x[1])
        counter = 0
        action = (action + 1) % 4