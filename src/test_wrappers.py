
import gymnasium as gym
import minigrid
from wrappers import MiniGridTextWrapper1, MiniGridTextWrapper2


original_env = gym.make("MiniGrid-LavaGapS5-v0") 
#original_env = gym.make("MiniGrid-LavaCrossingS9N3-v0")

env = MiniGridTextWrapper1(original_env)

obs, _ = env.reset()
print(obs)


env = MiniGridTextWrapper2(original_env, show_numbers=False)

obs, _ = env.reset()
print(obs)


env = MiniGridTextWrapper2(original_env, show_numbers=True)

obs, _ = env.reset()
print(obs)