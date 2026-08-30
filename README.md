# MiniGrid Benchmark

A benchmark for evaluating how well LLMs, used as ReAct-style agents, can navigate 2D grid-world environments described entirely in text.

## Overview

The agent receives a text rendering of a [MiniGrid](https://github.com/Farama-Foundation/Minigrid) environment (no images) and must reach a goal cell while avoiding lava, by repeatedly outputting a reasoning trace and one of three actions:

```
TURN_LEFT | TURN_RIGHT | MOVE_FORWARD
```

Each episode runs until the agent reaches the goal, dies, or exceeds a step budget. The benchmark reports success rate, reward, and steps-to-completion across environments and configurations.

## What is being measured

For each model, the benchmark runs **8 fixed configurations**, varying three factors:

| Factor | Values |
|---|---|
| View | `global` (whole map) vs. `local` (egocentric, partially observable) |
| Observation format | `simple` (bare characters) vs. `annotated` (row/column labels + cell separators) |
| History window | last 1 vs. last 5 turns kept in the prompt |

Environments (default set):

| Environment | Max steps |
|---|---|
| `MiniGrid-LavaGapS5-v0` | 15 |
| `MiniGrid-LavaCrossingS9N3-v0` | 25 |

Each configuration runs 5 episodes per environment, on pre-generated seeds shared across all agents for a fair comparison.

## Supported model providers

| Provider | Backend | Notes |
|---|---|---|
| `openai` | `langchain-openai` | Requires `OPENAI_API_KEY` |
| `deepseek` | `langchain-deepseek` | Requires `DEEPSEEK_API_KEY` |
| `hf` | `langchain-huggingface` (local inference) | Optional 4-bit/8-bit quantization via `bitsandbytes` |

## Quick start

The maintained entry point is a notebook, designed to run unmodified on local machines, Google Colab, and Kaggle:

```
src/run_full_benchmark_minigrid.ipynb
```

It detects the execution environment, resolves API keys/secrets, lets you pick a provider/model from a dropdown, and resumes an interrupted run automatically if one is found.

For local setup:

```bash
pip install -r requirements.txt
```

Then open and run the notebook from `src/`.

## Results layout

Each run is stored as:

```
results/<experiment_name>/<config_name>/<environment_id>/<run_number>.json
results/<experiment_name>/summary.json
```

- Each per-run JSON holds the full message history, the resolved configuration, the code version (git commit), and the outcome (success, reward, steps).
- `summary.json` is an index over all per-run files for one experiment. It is optional: if missing or deleted, it is transparently rebuilt from the per-run JSON files the next time the experiment is run (or via `experiments_util.recompute_main_json_from_run_files`).

## Repository structure

```
src/            Core library (agent loop, environment wrappers, prompts, experiment runner)
                and the benchmark-running notebook.
tools/          Standalone scripts/notebooks for analysis, dataset export, and a
                Streamlit results visualizer.
dataset/        Derived datasets built from experiment logs (e.g. for fine-tuning).
results/        Raw experiment outputs (per-run JSON + summary.json).
paper/          Manuscript describing the benchmark and findings (Quarto).
```

## Tools

- `tools/experiment_web_visualizer.py` — Streamlit app to browse experiments, inspect per-run metrics, and replay the full agent conversation. Run with `run_experiment_visualizer.cmd` or `streamlit run tools/experiment_web_visualizer.py`.
- `tools/analyze_run_failures.py` — flags runs where the model never produced a response (API-level failures).
- `tools/plot_experiment_results*.ipynb`, `tools/plot_helper.py` — aggregate results into pandas DataFrames and charts.
- `tools/truncate_experiment_runs.py`, `tools/rename_old_results.py` — maintenance utilities for the `results/` folder.
- `tools/build_grpo_dataset_from_deepseek.py` — extracts prompt/action pairs from experiment logs for RL/fine-tuning datasets.
- `tools/play_text_minigrid.py` — play an environment manually from the console, using the same text wrappers the agents see.

## Status

Actively evolving research project; interfaces and result formats may still change between experiment batches. See [paper/paper.qmd](paper/paper.qmd) for the accompanying write-up (in progress).

## License

Not yet specified. Contact the author before reuse.

## Contact

Pablo A. Sampaio — Universidade Federal Rural de Pernambuco (UFRPE)
