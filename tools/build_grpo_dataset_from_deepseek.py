"""
Build a GRPO-style prompt/action dataset from DeepSeek experiment logs.

It scans per-run JSON files for a target environment and converts each
human->ai turn pair into a dataset record:
- prompt: human message (observation text)
- output: action extracted from ai message

For successful runs, it includes every valid turn pair.
For unsuccessful runs, it includes only the last turn pair when the model
selected FRENTE, which is the step that typically moves the agent into lava.

Usage:
    python tools/build_grpo_dataset_from_deepseek.py
    python tools/build_grpo_dataset_from_deepseek.py --env MiniGrid-LavaGapS5-v0
    python tools/build_grpo_dataset_from_deepseek.py --output data/lavagap_grpo.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

VALID_ACTIONS = {"GIRA_ANTI_HORARIO", "GIRA_HORARIO", "FRENTE"}


def normalize_action_name(action_text: str) -> str:
    action = action_text.strip().upper()
    action = action.replace("Á", "A").replace("Ã", "A").replace("Ç", "C")
    return re.sub(r"[^A-Z_]", "", action)


def extract_action(response_text: str) -> str:
    action_xml = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL | re.IGNORECASE)
    if action_xml:
        action = normalize_action_name(action_xml.group(1))
        return action if action in VALID_ACTIONS else ""

    action_match = re.search(
        r"(?:\*\*)?(?:ACTION|AÇÃO)(?:\*\*)?\s*:\s*(GIRA_ANTI_HORARIO|GIRA_ANTI_HORÁRIO|GIRA_HORARIO|GIRA_HORÁRIO|FRENTE)\b",
        response_text,
        re.IGNORECASE,
    )
    if action_match:
        action = normalize_action_name(action_match.group(1))
        return action if action in VALID_ACTIONS else ""

    return ""


def load_json(filepath: Path):
    with filepath.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_run_files(results_dir: Path) -> Iterable[Path]:
    for run_file in sorted(results_dir.rglob("*.json")):
        yield run_file


def extract_entries_from_history(
    experiment_name: str,
    run_payload: dict,
    history_file: str,
    only_last_entry: bool = False,
) -> tuple[list[dict], int, int]:
    history = run_payload.get("history", [])
    if not isinstance(history, list) or not history:
        return [], 0, 0

    paired_messages: list[tuple[int, dict, dict]] = []
    for idx, message in enumerate(history):
        if message.get("role") != "human":
            continue
        if idx + 1 >= len(history):
            continue

        next_message = history[idx + 1]
        if next_message.get("role") != "ai":
            continue

        paired_messages.append((idx, message, next_message))

    if only_last_entry:
        paired_messages = paired_messages[-1:] if paired_messages else []

    entries: list[dict] = []
    skipped_unparsed = 0

    for step_index, (_, message, next_message) in enumerate(paired_messages, start=1):
        prompt = str(message.get("content", ""))
        action = extract_action(str(next_message.get("content", "")))
        if not action:
            skipped_unparsed += 1
            continue

        entries.append(
            {
                "prompt": prompt,
                "output": action,
                "meta": {
                    "experiment": experiment_name,
                    "env": run_payload.get("env"),
                    "run": run_payload.get("run"),
                    "seed": run_payload.get("seed"),
                    "step": step_index,
                    "history_file": history_file,
                },
            }
        )

    return entries, len(paired_messages), skipped_unparsed


def build_dataset_entries(results_dir: Path, env_name: str) -> tuple[list[dict], dict]:
    entries: list[dict] = []

    stats = {
        "runs_scanned": 0,
        "successful_runs": 0,
        "failed_runs_included": 0,
        "total_turn_pairs": 0,
        "pairs_with_unparsed_action": 0,
        "skipped_non_run_json": 0,
    }

    for run_file in iter_run_files(results_dir):
        stats["runs_scanned"] += 1

        try:
            run_payload = load_json(run_file)
        except (json.JSONDecodeError, OSError):
            stats["skipped_non_run_json"] += 1
            continue

        if not isinstance(run_payload, dict) or "history" not in run_payload or "experiment" not in run_payload:
            stats["skipped_non_run_json"] += 1
            continue

        if run_payload.get("env") != env_name:
            continue

        history_file = str(run_file.relative_to(results_dir))
        success = int(run_payload.get("success", 0))

        if success == 1:
            stats["successful_runs"] += 1
            run_entries, scanned_pairs, skipped_unparsed = extract_entries_from_history(
                str(run_payload.get("experiment", "")),
                run_payload,
                history_file,
                only_last_entry=False,
            )
            entries.extend(run_entries)
            stats["total_turn_pairs"] += scanned_pairs
            stats["pairs_with_unparsed_action"] += skipped_unparsed
            continue

        if success != 0:
            continue

        history = run_payload.get("history", [])
        if not isinstance(history, list) or len(history) < 2:
            continue

        last_action = ""
        for idx in range(len(history) - 2, -1, -1):
            if history[idx].get("role") == "human" and history[idx + 1].get("role") == "ai":
                last_action = extract_action(str(history[idx + 1].get("content", "")))
                break

        if last_action != "FRENTE":
            continue

        run_entries, scanned_pairs, skipped_unparsed = extract_entries_from_history(
            str(run_payload.get("experiment", "")),
            run_payload,
            history_file,
            only_last_entry=True,
        )
        if not run_entries:
            continue

        entries.extend(run_entries)
        stats["failed_runs_included"] += 1
        stats["total_turn_pairs"] += scanned_pairs
        stats["pairs_with_unparsed_action"] += skipped_unparsed

    return entries, stats


def write_jsonl(output_path: Path, entries: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_results_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "results" / "2026-05-11_deepseek"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GRPO prompt/action dataset from DeepSeek results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Path to DeepSeek results directory.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-LavaGapS5-v0",
        help="Environment name filter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. If omitted, uses <results-dir>/grpo_<env>.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()

    if args.output is None:
        safe_env = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.env)
        output_path = results_dir / f"grpo_{safe_env}.jsonl"
    else:
        output_path = args.output.resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    entries, stats = build_dataset_entries(results_dir, args.env)
    write_jsonl(output_path, entries)

    print(f"Saved {len(entries)} dataset entries to: {output_path}")
    print(f"Runs scanned: {stats['runs_scanned']}")
    print(f"Successful runs matched: {stats['successful_runs']}")
    print(f"Failed runs included: {stats['failed_runs_included']}")
    print(f"Turn pairs scanned: {stats['total_turn_pairs']}")
    if stats["pairs_with_unparsed_action"] > 0:
        print(f"Pairs with unparsed action: {stats['pairs_with_unparsed_action']}")
    if stats["skipped_non_run_json"] > 0:
        print(f"Skipped non-run JSON files: {stats['skipped_non_run_json']}")


if __name__ == "__main__":
    main()
