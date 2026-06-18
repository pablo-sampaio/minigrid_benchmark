
import gymnasium as gym
import minigrid
from wrappers import MiniGridTextLocalObsWrapper, MiniGridTextGlobalObsWrapper


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


def test_local_wrapper(show_numbers=False, separate=False):
	for env_id in ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaCrossingS9N3-v0"]:
		_print_wrapper_obs(env_id, lambda e: MiniGridTextLocalObsWrapper(e, show_numbers=show_numbers, separate_cells=separate), f"MiniGridTextLocalObsWrapper(show_numbers={show_numbers}, separate_cells={separate})")


def test_global_wrapper(show_numbers=False, separate=False):
	for env_id in ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaCrossingS9N3-v0"]:
		_print_wrapper_obs(env_id, lambda e: MiniGridTextGlobalObsWrapper(e, show_numbers=show_numbers, separate_cells=separate), f"MiniGridTextGlobalObsWrapper(show_numbers={show_numbers}, separate_cells={separate})")


if __name__ == '__main__':
	#test_local_wrapper(show_numbers=True, separate=True)
	#test_local_wrapper(show_numbers=False, separate=False)
	test_global_wrapper(show_numbers=True, separate=False)
	test_global_wrapper(show_numbers=False, separate=True)
