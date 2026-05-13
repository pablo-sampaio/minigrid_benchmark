import os
import tempfile
import unittest

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
        original_results_base_dir = eu.RESULTS_BASE_DIR

        try:
            eu.DEFAULT_ENVIRONMENT_IDS = {"DummyEnv-v0": 3}
            eu.RUNS_PER_ENV = 3
            eu.gym.make = lambda _env_id: _DummyEnv()

            with tempfile.TemporaryDirectory() as tmp_dir:
                eu.RESULTS_BASE_DIR = tmp_dir
                experiment_name = "experimentos_resume"
                summary_file = os.path.abspath(os.path.join(tmp_dir, experiment_name, f"{experiment_name}.json"))

                first_agent = _CountingAgent(fail_after=2)
                config = [{"name": "exp", "agent": first_agent, "wrapper_fn": lambda env: env}]

                with self.assertRaises(RuntimeError):
                    eu.run_and_save_experiments(
                        config,
                        experiment_name=experiment_name,
                        verbose=False,
                    )

                self.assertTrue(os.path.exists(os.path.join(tmp_dir, experiment_name, "exp", "DummyEnv-v0", "1.json")))
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, experiment_name, "exp", "DummyEnv-v0", "2.json")))

                resumed_agent = _CountingAgent()
                resumed_config = [{"name": "exp", "agent": resumed_agent, "wrapper_fn": lambda env: env}]

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
        finally:
            eu.DEFAULT_ENVIRONMENT_IDS = original_env_ids
            eu.RUNS_PER_ENV = original_runs_per_env
            eu.gym.make = original_make
            eu.RESULTS_BASE_DIR = original_results_base_dir


if __name__ == "__main__":
    unittest.main()
