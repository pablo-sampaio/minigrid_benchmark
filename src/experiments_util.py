
import os
import time
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

from wrappers import MiniGridTextLocalObsWrapper, MiniGridTextGlobalObsWrapper
from wrappers import SYSTEM_PROMPT_GLOBAL_1, SYSTEM_PROMPT_GLOBAL_2, SYSTEM_PROMPT_LOCAL_1, SYSTEM_PROMPT_LOCAL_2, OBS_TEMPLATE
from react_agent import ReActAgent


RESULTS_BASE_DIR = path.join(path.dirname(path.dirname(__file__)), "results")
BASE_SEED = 3712

DEFAULT_ENVIRONMENT_IDS = {
    "MiniGrid-LavaGapS5-v0": 15,
    "MiniGrid-LavaCrossingS9N3-v0": 25,
}

RUNS_PER_ENV = 5 #10
EPISODE_MAX_RETRIES_ON_ERROR = 1
EPISODE_RETRY_DELAY_SECONDS = 30

# TODO: remove all
def wrapper1(env):
    return MiniGridTextGlobalObsWrapper(env, show_numbers=False, separate_cells=False)

def wrapper2_with_numbers(env):
    return MiniGridTextGlobalObsWrapper(env, show_numbers=True, separate_cells=True)

def wrapper2_without_numbers(env):
    return MiniGridTextGlobalObsWrapper(env, show_numbers=False, separate_cells=True)

def wrapper_local_obs(env):
    return MiniGridTextLocalObsWrapper(env, show_numbers=False)

def wrapper_local_obs_noseparation(env):
    return MiniGridTextLocalObsWrapper(env, show_numbers=False, separate_cells=False)

def wrapper_local_obs_with_numbers(env):
    return MiniGridTextLocalObsWrapper(env, show_numbers=True)


def create_experiment_config(model_name, model, global_view, show_numbers, separate_cells, history_size=1):
    prompt = None
    view_str = ""
    if global_view:
        wrapper_fn = lambda env: MiniGridTextGlobalObsWrapper(env, show_numbers=show_numbers, separate_cells=separate_cells)
        if show_numbers and separate_cells:
            view_str = "global_special"
            prompt = SYSTEM_PROMPT_GLOBAL_2
        elif not show_numbers and not separate_cells:
            view_str = "global_simple"
            prompt = SYSTEM_PROMPT_GLOBAL_1
        else:
            raise ValueError("Invalid combination of show_numbers and separate_cells for global view. Only (False, False) and (True, True) are supported.")
    else:
        wrapper_fn = lambda env: MiniGridTextLocalObsWrapper(env, show_numbers=show_numbers, separate_cells=separate_cells)
        if separate_cells and separate_cells:
            view_str = "local_special"
            prompt = SYSTEM_PROMPT_LOCAL_2
        elif not show_numbers and not separate_cells:
            view_str = "local_simple"
            prompt = SYSTEM_PROMPT_LOCAL_1
        else:
            raise ValueError("Invalid combination of show_numbers and separate_cells for local view. Only separator-based views or fully simple views are supported.")

    agent = ReActAgent(model, prompt, OBS_TEMPLATE, history_window=history_size, verbose=False)
    config_name = f"{model_name}-{view_str}-history{history_size}"
    return {
        'name': config_name,
        'agent': agent,
        'wrapper_fn': wrapper_fn,
    }

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


def recompute_main_json_from_run_files(results_dir: str, write_file: bool = True):
    """
    Recompute the main summary JSON for an experiment directory by scanning
    per-run JSON files under: <results_dir>/<agent_name>/<env_id>/<run>.json

    Args:
        results_dir: Path to one experiment directory inside RESULTS_BASE_DIR.
        write_file: If True, writes the reconstructed summary to
                    <results_dir>/<basename(results_dir)>.json.

    Returns:
        Tuple[dict, str]: (reconstructed_summary, output_main_json_path)
    """
    results_dir = os.path.abspath(os.fspath(results_dir))
    experiment_name = os.path.basename(results_dir.rstrip("/\\"))
    main_json_path = os.path.join(results_dir, f"{experiment_name}.json")

    reconstructed = {}

    if not os.path.isdir(results_dir):
        if write_file:
            _write_json_atomic(main_json_path, reconstructed)
        return reconstructed, main_json_path

    for agent_name in sorted(os.listdir(results_dir)):
        agent_dir = os.path.join(results_dir, agent_name)
        if not os.path.isdir(agent_dir):
            continue

        entries = {}
        for env_name in sorted(os.listdir(agent_dir)):
            env_dir = os.path.join(agent_dir, env_name)
            if not os.path.isdir(env_dir):
                continue

            for filename in sorted(os.listdir(env_dir)):
                if not filename.lower().endswith('.json'):
                    continue

                run_file_path = os.path.join(env_dir, filename)
                payload = _load_json_if_exists(run_file_path)
                if not payload:
                    continue

                run_number = int(payload.get('run', os.path.splitext(filename)[0]))
                run_key = (env_name, run_number)
                entries[run_key] = {
                    'env': str(payload.get('env', env_name)),
                    'run': run_number,
                    'seed': int(payload.get('seed', -1)),
                    'max_steps': int(payload.get('max_steps', -1)),
                    'steps': int(payload.get('steps', payload.get('step_count', -1))),
                    'success': int(payload.get('success', 0)),
                    'reward': payload.get('reward', 0.0),
                    'history_file': _normalize_history_path(results_dir, run_file_path),
                }

        reconstructed[agent_name] = _sort_agent_results(list(entries.values()))

    if write_file:
        _write_json_atomic(main_json_path, reconstructed)

    return reconstructed, main_json_path



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
        # To make resume less dependent on the main summary file, this is the
        # place to call recompute_main_json_from_run_files(results_dir) and
        # initialize all_experiment_data from per-run JSON files.
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

                    reward = None
                    for episode_attempt in range(EPISODE_MAX_RETRIES_ON_ERROR + 1):
                        obs, _ = test_env.reset(seed=seed)
                        try:
                            reward = agent.solve_environment(test_env, obs, max_steps=max_steps)
                            break
                        except RuntimeError as exc:
                            is_last_attempt = episode_attempt == EPISODE_MAX_RETRIES_ON_ERROR
                            if is_last_attempt:
                                raise

                            if verbose:
                                print(
                                    f"Run {run_number} failed due to API error. "
                                    f"Retrying episode in {EPISODE_RETRY_DELAY_SECONDS}s "
                                    f"(attempt {episode_attempt + 1}/{EPISODE_MAX_RETRIES_ON_ERROR + 1}). "
                                    f"Last error: {exc}"
                                )
                            time.sleep(EPISODE_RETRY_DELAY_SECONDS)

                    if reward is None:
                        raise RuntimeError("Episode finished without reward due to unexpected retry flow.")

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

