from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def detect_default_runs_per_env() -> int:
    """Read RUNS_PER_ENV from src/experiments_util.py without importing heavy deps."""
    default_value = 5
    experiments_util_path = Path(__file__).resolve().parent.parent / "src" / "experiments_util.py"

    try:
        source = experiments_util_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return default_value

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        has_target = any(isinstance(target, ast.Name) and target.id == "RUNS_PER_ENV" for target in node.targets)
        if not has_target:
            continue

        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            return int(value.value)

        if (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Constant)
            and isinstance(value.operand.value, int)
        ):
            return -int(value.operand.value)

    return default_value


def find_main_summary_file(experiment_dir: Path) -> Path | None:
    """Find the main summary JSON (usually <experiment_dir_name>.json)."""
    candidates = sorted(p for p in experiment_dir.glob("*.json") if p.is_file())
    if not candidates:
        return None

    preferred = experiment_dir / f"{experiment_dir.name}.json"
    if preferred in candidates:
        return preferred

    return candidates[0]


def is_within_dir(path_to_check: Path, parent_dir: Path) -> bool:
    try:
        path_to_check.resolve().relative_to(parent_dir.resolve())
        return True
    except ValueError:
        return False


def resolve_run_file_from_entry(
    experiment_dir: Path,
    config_name: str,
    entry: dict[str, Any],
    run_number: int,
) -> Path | None:
    history_file = entry.get("history_file")
    if isinstance(history_file, str) and history_file.strip():
        candidate = (experiment_dir / Path(history_file)).resolve()
        if candidate.suffix.lower() == ".json" and is_within_dir(candidate, experiment_dir):
            return candidate

    env_name = entry.get("env")
    if not isinstance(env_name, str) or not env_name:
        return None

    run_file_two_digits = experiment_dir / config_name / env_name / f"{run_number:02d}.json"
    if is_within_dir(run_file_two_digits, experiment_dir):
        return run_file_two_digits

    return None


def collect_extra_run_files(experiment_dir: Path, max_runs: int) -> set[Path]:
    """Collect per-run JSON files with run numbers above max_runs."""
    files_to_remove: set[Path] = set()

    for config_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        for env_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            for run_file in sorted(env_dir.glob("*.json")):
                if not run_file.is_file() or not run_file.stem.isdigit():
                    continue
                if int(run_file.stem) > max_runs:
                    files_to_remove.add(run_file.resolve())

    return files_to_remove


def truncate_experiment(experiment_dir: Path, max_runs: int, dry_run: bool) -> tuple[int, int, int]:
    """
    Truncate one experiment folder.

    Returns:
        (summary_entries_removed, run_files_deleted, summary_files_updated)
    """
    summary_path = find_main_summary_file(experiment_dir)
    if summary_path is None:
        print(f"SKIP (no summary JSON): {experiment_dir}")
        return 0, 0, 0

    try:
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"SKIP (invalid JSON): {summary_path} | {exc}")
        return 0, 0, 0

    if not isinstance(summary_data, dict):
        print(f"SKIP (summary must be dict): {summary_path}")
        return 0, 0, 0

    removed_entries = 0
    summary_changed = False
    files_to_remove: set[Path] = set()

    for config_name, entries in list(summary_data.items()):
        if not isinstance(config_name, str) or not isinstance(entries, list):
            continue

        kept_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                kept_entries.append(entry)
                continue

            run_value = entry.get("run")
            try:
                run_number = int(run_value)
            except (TypeError, ValueError):
                kept_entries.append(entry)
                continue

            if run_number <= max_runs:
                kept_entries.append(entry)
                continue

            removed_entries += 1
            run_file = resolve_run_file_from_entry(experiment_dir, config_name, entry, run_number)
            if run_file is not None:
                files_to_remove.add(run_file)

        if len(kept_entries) != len(entries):
            summary_data[config_name] = kept_entries
            summary_changed = True

    # Also remove stray per-run files above the limit, even when not listed in summary.
    files_to_remove.update(collect_extra_run_files(experiment_dir, max_runs))

    if summary_changed:
        print(f"UPDATE SUMMARY: {summary_path} (removed entries: {removed_entries})")
        if not dry_run:
            summary_path.write_text(json.dumps(summary_data, indent=4, ensure_ascii=False), encoding="utf-8")

    deleted_files = 0
    for run_file in sorted(files_to_remove):
        if not run_file.exists():
            continue
        if not is_within_dir(run_file, experiment_dir):
            print(f"SKIP (outside experiment dir): {run_file}")
            continue

        print(f"DELETE RUN FILE: {run_file}")
        deleted_files += 1
        if not dry_run:
            run_file.unlink()

    return removed_entries, deleted_files, int(summary_changed)


def main() -> None:
    default_runs = detect_default_runs_per_env()

    parser = argparse.ArgumentParser(
        description=(
            "Truncate experiment runs in results folders by limiting each configuration to a maximum run number."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="../results/",
        help='Directory containing level-1 experiment folders (default: "../results/").',
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=default_runs,
        help=(
            "Maximum run number to keep for each configuration (default: src/experiments_util.RUNS_PER_ENV)."
        ),
    )
    parser.set_defaults(dry_run=True)
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Show what would be changed/deleted without writing files (default).",
    )
    execution_mode.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="Actually write summary files and delete run files.",
    )

    args = parser.parse_args()

    if args.max_runs < 0:
        raise SystemExit("--max-runs must be >= 0")

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"Invalid results directory: {results_dir}")

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"Mode: {mode}")
    print(f"Results dir: {results_dir.resolve()}")
    print(f"Max runs: {args.max_runs}\n")

    total_removed_entries = 0
    total_deleted_files = 0
    total_updated_summaries = 0
    total_experiments = 0

    for experiment_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        total_experiments += 1
        print(f"=== Experiment: {experiment_dir.name} ===")
        removed_entries, deleted_files, updated_summaries = truncate_experiment(
            experiment_dir=experiment_dir,
            max_runs=args.max_runs,
            dry_run=args.dry_run,
        )
        total_removed_entries += removed_entries
        total_deleted_files += deleted_files
        total_updated_summaries += updated_summaries
        print()

    print("=== Summary ===")
    print(f"Experiments scanned: {total_experiments}")
    print(f"Summary files updated: {total_updated_summaries}")
    print(f"Summary entries removed: {total_removed_entries}")
    print(f"Run files deleted: {total_deleted_files}")


if __name__ == "__main__":
    main()
