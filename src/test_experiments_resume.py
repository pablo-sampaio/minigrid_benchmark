import os
import tempfile
import unittest
from datetime import datetime

import experiments_util as eu


class _DummyEnv:
    def reset(self, seed=None):
        return f"obs-{seed}", {}

    def close(self):
        return None


class _CountingAgent:
    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.calls = 0

    def solve_environment(self, env, obs, max_steps=25):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("forced interruption")
        return 1.0

    def get_full_history(self):
        return []


class ResumeExperimentsTests(unittest.TestCase):
    def test_resume_skips_completed_runs(self):
        original_env_ids = eu.DEFAULT_ENVIRONMENT_IDS
        original_runs_per_env = eu.RUNS_PER_ENV
        original_make = eu.gym.make
        original_get_git_code_version = eu._get_git_code_version
        original_results_base_dir = eu.RESULTS_BASE_DIR

        try:
            eu.DEFAULT_ENVIRONMENT_IDS = {"DummyEnv-v0": 3}
            eu.RUNS_PER_ENV = 3
            eu.gym.make = lambda _env_id: _DummyEnv()
            eu._get_git_code_version = lambda: {
                "git_commit": "abc123",
                "git_dirty": True,
            }

            with tempfile.TemporaryDirectory() as tmp_dir:
                eu.RESULTS_BASE_DIR = tmp_dir
                experiment_name = "experimentos_resume"
                summary_file = os.path.abspath(os.path.join(tmp_dir, experiment_name, eu.SUMMARY_FILENAME))

                first_agent = _CountingAgent(fail_after=2)
                config = [{
                    "name": "exp",
                    "agent": first_agent,
                    "wrapper_fn": lambda env: env,
                    "config_params": {
                        "model_name": "dummy-model",
                        "global_view": True,
                        "show_numbers": False,
                        "separate_cells": False,
                        "history_size": 1,
                    },
                }]

                with self.assertRaises(RuntimeError):
                    eu.run_and_save_experiments(
                        config,
                        experiment_name=experiment_name,
                        verbose=False,
                    )

                self.assertTrue(os.path.exists(os.path.join(tmp_dir, experiment_name, "exp", "DummyEnv-v0", "1.json")))
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, experiment_name, "exp", "DummyEnv-v0", "2.json")))

                resumed_agent = _CountingAgent()
                resumed_config = [{
                    "name": "exp",
                    "agent": resumed_agent,
                    "wrapper_fn": lambda env: env,
                    "config_params": {
                        "model_name": "dummy-model",
                        "global_view": True,
                        "show_numbers": False,
                        "separate_cells": False,
                        "history_size": 1,
                    },
                }]

                final_results, saved_path = eu.run_and_save_experiments(
                    resumed_config,
                    experiment_name=experiment_name,
                    verbose=False,
                )

                self.assertEqual(saved_path, os.path.abspath(summary_file))
                self.assertEqual(resumed_agent.calls, 1, "Only the missing run should execute during resume")

                runs = final_results["exp"]
                self.assertEqual(len(runs), 3)
                self.assertEqual(sorted((item["env"], item["run"]) for item in runs), [
                    ("DummyEnv-v0", 1),
                    ("DummyEnv-v0", 2),
                    ("DummyEnv-v0", 3),
                ])

                run_payload_path = os.path.join(tmp_dir, experiment_name, "exp", "DummyEnv-v0", "3.json")
                run_payload = eu._load_json_if_exists(run_payload_path)
                self.assertEqual(run_payload.get("config"), resumed_config[0]["config_params"])
                self.assertEqual(run_payload.get("code_version"), {
                    "git_commit": "abc123",
                    "git_dirty": True,
                })
                datetime.fromisoformat(run_payload["completed_at"])
                self.assertEqual(runs[-1]["config"], resumed_config[0]["config_params"])
                self.assertEqual(runs[-1]["code_version"], {
                    "git_commit": "abc123",
                    "git_dirty": True,
                })
                self.assertEqual(runs[-1]["completed_at"], run_payload["completed_at"])
        finally:
            eu.DEFAULT_ENVIRONMENT_IDS = original_env_ids
            eu.RUNS_PER_ENV = original_runs_per_env
            eu.gym.make = original_make
            eu._get_git_code_version = original_get_git_code_version
            eu.RESULTS_BASE_DIR = original_results_base_dir


if __name__ == "__main__":
    unittest.main()
