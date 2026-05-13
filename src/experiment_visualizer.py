from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"


@st.cache_data(show_spinner=False)
def load_json_file(file_path: str) -> Any:
    path_obj = Path(file_path)
    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def discover_experiments(results_dir_str: str) -> list[dict[str, str]]:
    results_dir = Path(results_dir_str)
    if not results_dir.exists():
        return []

    experiments: list[dict[str, str]] = []
    for child in sorted(results_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue

        named_summary = child / f"{child.name}.json"
        if named_summary.exists():
            experiments.append({"name": child.name, "dir": str(child), "summary": str(named_summary)})
            continue

        fallback_json_files = sorted(child.glob("*.json"), key=lambda p: p.name.lower())
        if fallback_json_files:
            experiments.append({"name": child.name, "dir": str(child), "summary": str(fallback_json_files[0])})

    return experiments


def resolve_history_path(experiment_dir: Path, prompt_name: str, row: dict[str, Any]) -> Path | None:
    history_file = row.get("history_file")
    if isinstance(history_file, str) and history_file.strip():
        normalized = history_file.replace("\\", "/").strip()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        candidate = (experiment_dir / normalized).resolve()
        if candidate.exists():
            return candidate

    env_name = str(row.get("env", ""))
    run_number = row.get("run")
    if isinstance(run_number, (int, float)):
        run_int = int(run_number)
        two_digit = experiment_dir / prompt_name / env_name / f"{run_int:02d}.json"
        one_digit = experiment_dir / prompt_name / env_name / f"{run_int}.json"
        if two_digit.exists():
            return two_digit.resolve()
        if one_digit.exists():
            return one_digit.resolve()

    return None


def summary_to_dataframe(summary_payload: Any, experiment_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not isinstance(summary_payload, dict):
        return pd.DataFrame(rows)

    for prompt_name, run_items in summary_payload.items():
        if not isinstance(run_items, list):
            continue

        for item in run_items:
            if not isinstance(item, dict):
                continue

            row = {
                "prompt": str(prompt_name),
                "env": str(item.get("env", "")),
                "run": item.get("run"),
                "seed": item.get("seed"),
                "max_steps": item.get("max_steps"),
                "steps": item.get("steps"),
                "success": item.get("success"),
                "reward": item.get("reward"),
            }
            history_path = resolve_history_path(experiment_dir, str(prompt_name), item)
            row["history_path"] = str(history_path) if history_path else ""
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["run"] = pd.to_numeric(frame["run"], errors="coerce").astype("Int64")
    frame["success"] = pd.to_numeric(frame["success"], errors="coerce").fillna(0).astype(int)
    frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce").fillna(0.0)
    frame["max_steps"] = pd.to_numeric(frame["max_steps"], errors="coerce")
    frame["steps"] = pd.to_numeric(frame["steps"], errors="coerce")
    frame = frame.sort_values(by=["prompt", "env", "run"], kind="stable")

    return frame


def format_metric_number(value: float | int, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_history_chat(history: Any) -> None:
    if not isinstance(history, list) or not history:
        st.info("No history entries found for this run.")
        return

    st.subheader("History")
    for idx, message in enumerate(history, start=1):
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "unknown")).lower()
        content = str(message.get("content", ""))

        if role == "human":
            with st.chat_message("user"):
                st.markdown(f"**Human #{idx}**")
                st.markdown(f"```text\n{content}\n```")
        elif role == "ai":
            with st.chat_message("assistant"):
                st.markdown(f"**AI #{idx}**")
                st.markdown(f"```text\n{content}\n```")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**{role.title()} #{idx}**")
                st.markdown(f"```text\n{content}\n```")


def main() -> None:
    st.set_page_config(
        page_title="MiniGrid Results Visualizer",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
:root {
  --bg-soft: #f6f3ea;
  --surface: #fffdf8;
  --line: #d8ccba;
  --text-strong: #2a2a20;
  --accent: #355f3f;
  --accent-soft: #e9f2e9;
  --ai-soft: #eef1f8;
}

.stApp {
  background: radial-gradient(circle at top right, #f4ead8 0%, #f6f3ea 45%, #f0ece2 100%);
}

h1, h2, h3 {
  font-family: "Trebuchet MS", "Segoe UI", sans-serif;
  letter-spacing: 0.2px;
  color: var(--text-strong);
}

div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 14px;
}

pre.grid-block {
  background: var(--accent-soft);
  border: 1px solid #b7ccb8;
  border-radius: 10px;
  color: #18311f;
  font-family: "Courier New", monospace;
  font-size: 0.90rem;
  line-height: 1.24;
  padding: 10px 12px;
  overflow-x: auto;
  white-space: pre;
}

pre.ai-block {
  background: var(--ai-soft);
  border: 1px solid #c6d1e3;
  border-radius: 10px;
  color: #1f2a3a;
  font-family: "Courier New", monospace;
  font-size: 0.90rem;
  line-height: 1.34;
  padding: 10px 12px;
  overflow-x: auto;
  white-space: pre-wrap;
}

div[data-testid="stSidebar"] {
  border-right: 1px solid var(--line);
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("MiniGrid Experiment Visualizer")
    st.caption("Browse experiment JSONs, inspect run metadata, and read the full conversation history as chat.")

    experiments = discover_experiments(str(RESULTS_DIR))
    if not experiments:
        st.error(f"No experiment folders with summary JSON found under: {RESULTS_DIR}")
        return

    experiment_labels = [item["name"] for item in experiments]
    selected_experiment_name = st.sidebar.selectbox("Experiment folder", experiment_labels)
    selected_experiment = next(item for item in experiments if item["name"] == selected_experiment_name)

    experiment_dir = Path(selected_experiment["dir"])
    summary_path = Path(selected_experiment["summary"])

    try:
        summary_payload = load_json_file(str(summary_path))
    except json.JSONDecodeError:
        st.error(f"Summary JSON is invalid: {summary_path}")
        return
    except OSError as exc:
        st.error(f"Could not read summary JSON: {exc}")
        return

    run_index_df = summary_to_dataframe(summary_payload, experiment_dir)
    if run_index_df.empty:
        st.warning("Summary JSON loaded, but no run items were found.")
        return

    prompt_options = sorted(run_index_df["prompt"].dropna().unique().tolist())
    selected_prompt = st.sidebar.selectbox("Prompt / Model config", prompt_options)

    prompt_df = run_index_df[run_index_df["prompt"] == selected_prompt].copy()
    env_options = sorted(prompt_df["env"].dropna().unique().tolist())
    selected_env = st.sidebar.selectbox("Environment", env_options)

    env_df = prompt_df[prompt_df["env"] == selected_env].copy()
    run_options = [int(v) for v in env_df["run"].dropna().tolist()]
    run_options = sorted(set(run_options))

    if not run_options:
        st.warning("No runs found for this prompt and environment.")
        return

    selected_run = st.sidebar.selectbox("Run", run_options)
    row = env_df[env_df["run"] == selected_run]
    if row.empty:
        st.warning("Selected run was not found in index.")
        return

    selected_row = row.iloc[0].to_dict()

    st.sidebar.markdown("---")
    st.sidebar.write("Summary file")
    st.sidebar.caption(str(summary_path))

    st.subheader("Aggregate Metrics")
    col1, col2, col3, col4 = st.columns(4)
    total_runs = int(prompt_df.shape[0])
    success_rate = float(prompt_df["success"].mean()) if total_runs else 0.0
    mean_reward = float(prompt_df["reward"].mean()) if total_runs else 0.0
    mean_steps = float(prompt_df["steps"].mean()) if prompt_df["steps"].notna().any() else float("nan")

    col1.metric("Runs in prompt", format_metric_number(total_runs))
    col2.metric("Success rate", f"{100.0 * success_rate:.1f}%")
    col3.metric("Mean reward", format_metric_number(mean_reward, digits=4))
    col4.metric("Mean steps", format_metric_number(mean_steps, digits=2))

    env_breakdown = (
        prompt_df.groupby("env", as_index=False)
        .agg(
            runs=("run", "count"),
            successes=("success", "sum"),
            success_rate=("success", "mean"),
            avg_reward=("reward", "mean"),
            avg_steps=("steps", "mean"),
        )
        .sort_values(by="env", kind="stable")
    )
    env_breakdown["success_rate"] = env_breakdown["success_rate"] * 100.0

    st.dataframe(
        env_breakdown,
        hide_index=True,
        use_container_width=True,
        column_config={
            "success_rate": st.column_config.NumberColumn("success_rate (%)", format="%.1f"),
            "avg_reward": st.column_config.NumberColumn(format="%.4f"),
            "avg_steps": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.subheader("Selected Run")
    left, right = st.columns([1.3, 1])
    with left:
        st.write(f"Experiment: {selected_experiment_name}")
        st.write(f"Prompt: {selected_prompt}")
        st.write(f"Environment: {selected_env}")
        st.write(f"Run: {selected_run}")
        st.write(f"Seed: {selected_row.get('seed')}")
    with right:
        st.write(f"Success: {selected_row.get('success')}")
        st.write(f"Reward: {selected_row.get('reward')}")
        st.write(f"Max steps: {selected_row.get('max_steps')}")
        st.write(f"Executed steps: {selected_row.get('steps')}")

    history_path = selected_row.get("history_path", "")
    history_payload: dict[str, Any] = {}

    if history_path:
        st.caption(f"Run file: {history_path}")
        try:
            loaded_run = load_json_file(history_path)
            if isinstance(loaded_run, dict):
                history_payload = loaded_run
        except json.JSONDecodeError:
            st.error(f"Run JSON is invalid: {history_path}")
        except OSError as exc:
            st.error(f"Could not read run JSON: {exc}")
    else:
        st.warning("No history file path was found for this run.")

    if history_payload:
        st.subheader("Run Attributes")
        top_keys = ["experiment", "env", "run", "seed", "max_steps", "steps", "success", "reward"]
        run_attributes = {k: history_payload.get(k) for k in top_keys if k in history_payload}
        st.json(run_attributes)
        render_history_chat(history_payload.get("history"))


if __name__ == "__main__":
    main()
