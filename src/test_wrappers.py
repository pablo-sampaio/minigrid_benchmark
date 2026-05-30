
import gymnasium as gym
import minigrid
from wrappers import MiniGridTextWrapper1, MiniGridTextWrapper2, MiniGridTextLocalObsWrapper


def _print_wrapper_obs(env_id, wrapper, label):
	print(f"\n=== {label} | {env_id} ===")
	original_env = gym.make(env_id)
	env = wrapper(original_env)
	obs, _ = env.reset()
	print(obs)
	# turn right
	obs, reward, terminated, truncated, info = env.step(1)
	print("After turning right:")
	print(obs)


def test_wrapper1():
	for env_id in ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaCrossingS9N3-v0"]:
		_print_wrapper_obs(env_id, lambda e: MiniGridTextWrapper1(e), "MiniGridTextWrapper1")


def test_wrapper2_without_numbers():
	for env_id in ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaCrossingS9N3-v0"]:
		_print_wrapper_obs(env_id, lambda e: MiniGridTextWrapper2(e, show_numbers=False), "MiniGridTextWrapper2(no numbers)")


def test_wrapper2_with_numbers():
	for env_id in ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaCrossingS9N3-v0"]:
		_print_wrapper_obs(env_id, lambda e: MiniGridTextWrapper2(e, show_numbers=True), "MiniGridTextWrapper2(with numbers)")


def test_local_wrapper_without_numbers():
	for env_id in ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaCrossingS9N3-v0"]:
		_print_wrapper_obs(env_id, lambda e: MiniGridTextLocalObsWrapper(e, show_numbers=False), "MiniGridTextLocalObsWrapper(no numbers)")


def test_local_wrapper_with_numbers():
	for env_id in ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaCrossingS9N3-v0"]:
		_print_wrapper_obs(env_id, lambda e: MiniGridTextLocalObsWrapper(e, show_numbers=True), "MiniGridTextLocalObsWrapper(with numbers)")


if __name__ == '__main__':
	#test_local_wrapper_with_numbers()
	test_local_wrapper_without_numbers()
	
	#test_wrapper1()
	#test_wrapper2_without_numbers()
	#test_wrapper2_with_numbers()
