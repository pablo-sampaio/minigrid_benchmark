
import os
from datetime import datetime
import json
import re
from os import path

from numpy.random import RandomState
import gymnasium as gym
from tqdm.auto import tqdm

try:
    from google.colab import output as colab_output
except ImportError:
    colab_output = None

from wrappers import MiniGridTextWrapper1, MiniGridTextWrapper2
from react_agent import ReActAgent


RESULTS_BASE_DIR = path.join(path.dirname(__file__), "results")
BASE_SEED = 3712

DEFAULT_ENVIRONMENT_IDS = {
    "MiniGrid-LavaGapS5-v0": 15,
    "MiniGrid-LavaCrossingS9N3-v0": 25,
}

RUNS_PER_ENV = 10 #1 #4

def wrapper1(env):
    return MiniGridTextWrapper1(env)

def wrapper2_with_numbers(env):
    return MiniGridTextWrapper2(env, show_numbers=True)

def wrapper2_without_numbers(env):
    return MiniGridTextWrapper2(env, show_numbers=False)


def _safe_path_component(name: str) -> str:
    # Replace invalid Windows path chars and trim trailing dots/spaces.
    safe_name = re.sub(r'[<>:"/\\|?*]+', "_", str(name)).strip().rstrip(". ")
    return safe_name or "unnamed"


def _write_json_atomic(filepath: str, payload) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    temp_path = f"{filepath}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
    os.replace(temp_path, filepath)


def _load_json_if_exists(filepath: str):
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError:
        return {}

    return loaded if isinstance(loaded, dict) else {}


def _normalize_history_path(results_dir: str, run_file_path: str) -> str:
    rel_path = os.path.relpath(run_file_path, start=results_dir)
    return f".{os.sep}{rel_path}"


def _sort_agent_results(agent_results):
    env_order = {env_id: idx for idx, env_id in enumerate(DEFAULT_ENVIRONMENT_IDS.keys())}
    return sorted(
        agent_results,
        key=lambda item: (
            env_order.get(item.get('env', ''), 10**6),
            str(item.get('env', '')),
            int(item.get('run', 0)),
        ),
    )



def run_and_save_experiments(experiment_configs, experiment_name=None, verbose=False):
    """
    Args:
    experiment_configs: List of dicts with {'name': str, 'agent': ReActAgent, 'wrapper_fn': function}

    experiment_name: Used to name the directory inside RESULTS_BASE_DIR that will hold the results of this experiment,
                 and also the main JSON file inside the same path. If not provided, a default name will be given with
                 the current date and time. When provided, and the directory already exists, any already-completed
                 runs (identified by their per-run JSON files) are skipped and the summary is saved back to the same file.
    
    verbose: If True, shows progress bars and prints more info about the experiment progress.
    """
    full_results_base_dir = os.path.abspath(os.fspath(RESULTS_BASE_DIR))

    if experiment_name is None:
        experiment_name = f"experimentos_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    experiment_name_safe = _safe_path_component(experiment_name)
    results_dir = os.path.abspath(os.path.join(full_results_base_dir, experiment_name_safe))
    filename = f"{experiment_name_safe}.json"
    filepath = os.path.abspath(os.path.join(results_dir, filename))

    if os.path.exists(filepath):
        all_experiment_data = _load_json_if_exists(filepath)
        if verbose:
            print(f"Resuming from: {filepath}")
    else:
        all_experiment_data = {}

    env_ids = list(DEFAULT_ENVIRONMENT_IDS.keys())
    max_steps_config = DEFAULT_ENVIRONMENT_IDS

    if verbose:
        print(f"Results will be saved to: {filepath}")

    os.makedirs(results_dir, exist_ok=True)
    rand_generator = RandomState(BASE_SEED)

    # Pre-generate seeds to ensure fairness across agents
    seeds_per_env = {env_id: [int(rand_generator.randint(3, 2**17 - 1)) for _ in range(RUNS_PER_ENV)] for env_id in env_ids}

    agents_iter = tqdm(experiment_configs, desc="Completed Experiment Configurations", disable=not verbose, position=0)
    for config in agents_iter:
        agent_name = config['name']
        agent = config['agent']
        wrapper_fn = config['wrapper_fn']
        experiment_dir = os.path.join(results_dir, _safe_path_component(agent_name))
        os.makedirs(experiment_dir, exist_ok=True)

        if verbose:
            agents_iter.set_postfix_str(agent_name)
        
        agent_results_index = {
            (item.get('env'), int(item.get('run', 0))): item
            for item in all_experiment_data.get(agent_name, [])
            if item.get('env') is not None and item.get('run') is not None
        }

        env_iter = tqdm(
            env_ids,
            desc=f"Completed Environments",
            disable=not verbose,
            leave=False,
            position=1,
        )
        for env_id in env_iter:
            seeds = seeds_per_env[env_id]
            max_steps = max_steps_config[env_id]
            env_dir = os.path.join(experiment_dir, _safe_path_component(env_id))
            os.makedirs(env_dir, exist_ok=True)

            if verbose:
                env_iter.set_postfix_str(env_id)

            test_env = wrapper_fn(gym.make(env_id))
            try:
                runs_iter = tqdm(
                    seeds,
                    desc=f"Finished Environment Runs:",
                    total=len(seeds),
                    disable=not verbose,
                    leave=False,
                    position=2,
                )
                for i, seed in enumerate(runs_iter):
                    run_number = i + 1
                    run_key = (env_id, run_number)
                    run_file_path = os.path.join(env_dir, f"{run_number:02d}.json")

                    if run_key in agent_results_index:
                        if verbose:
                            runs_iter.set_postfix_str(f"run {run_number} skipped (already in summary)")
                        continue
                    if verbose:
                        runs_iter.set_postfix_str(f"run {run_number}...")

                    if os.path.exists(run_file_path):
                        existing = _load_json_if_exists(run_file_path)
                        agent_results_index[run_key] = {
                            'env': env_id,
                            'run': run_number,
                            'seed': int(existing.get('seed', seed)),
                            'max_steps': int(existing.get('max_steps', max_steps)),
                            'steps': int(existing.get('steps', existing.get('step_count', -1))),
                            'success': int(existing.get('success', 0)),
                            'reward': existing.get('reward', 0.0),
                            'history_file': _normalize_history_path(results_dir, run_file_path),
                        }
                        all_experiment_data[agent_name] = _sort_agent_results(list(agent_results_index.values()))
                        _write_json_atomic(filepath, all_experiment_data)

                        if verbose:
                            runs_iter.set_postfix_str(f"run {run_number} skipped (already done)")
                        continue

                    obs, _ = test_env.reset(seed=seed)
                    reward = agent.solve_environment(test_env, obs, max_steps=max_steps)
                    success = 1 if reward > 0 else 0

                    run_payload = {
                        'experiment': agent_name,
                        'env': env_id,
                        'run': run_number,
                        'seed': seed,
                        'max_steps': max_steps,
                        'steps': int(getattr(agent, 'step_count', -1)),
                        'success': success,
                        'reward': reward,
                        'history': agent.get_full_history(),
                    }

                    _write_json_atomic(run_file_path, run_payload)

                    agent_results_index[run_key] = {
                        'env': env_id,
                        'run': run_number,
                        'seed': seed,
                        'max_steps': max_steps,
                        'steps': int(getattr(agent, 'step_count', -1)),
                        'success': success,
                        'reward': reward,
                        'history_file': _normalize_history_path(results_dir, run_file_path),
                    }

                    # Save summary after every completed run for crash resilience.
                    all_experiment_data[agent_name] = _sort_agent_results(list(agent_results_index.values()))
                    _write_json_atomic(filepath, all_experiment_data)

                    if colab_output is not None and verbose:
                        colab_output.clear()
            finally:
                test_env.close()

        all_experiment_data[agent_name] = _sort_agent_results(list(agent_results_index.values()))

    # 3. Save to Google Drive
    _write_json_atomic(filepath, all_experiment_data)

    return all_experiment_data, filepath

