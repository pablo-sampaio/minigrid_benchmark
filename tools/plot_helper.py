
import json
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    "env",
    "run",
    "seed",
    "steps",
    "success",
    "reward",
    "completed_at",
    "model_name",
    "global_view",
    "show_numbers",
    "separate_cells",
    "history_size",
    "config_name",
]


def _normalize_record(record, fallback_config_name=None):
    config = record.get("config", {}) if isinstance(record, dict) else {}

    return {
        "env": record.get("env"),
        "run": record.get("run"),
        "seed": record.get("seed"),
        "steps": record.get("steps"),
        "success": record.get("success"),
        "reward": record.get("reward"),
        "completed_at": record.get("completed_at"),
        "model_name": config.get("model_name"),
        "global_view": config.get("global_view"),
        "show_numbers": config.get("show_numbers"),
        "separate_cells": config.get("separate_cells"),
        "history_size": config.get("history_size"),
        "config_name": record.get("experiment", fallback_config_name),
    }


def _iter_json_files(parent_folder, recursive=True):
    parent_path = Path(parent_folder)
    pattern = "**/*.json" if recursive else "*.json"
    return [p for p in parent_path.glob(pattern) if p.is_file()]


# for this one, you may pass something like "results/benchmark_deepseek_deepseek-v4-pro_2026-06-20_09h-34min"
def create_dataframe_from_result_folder(parent_folder, recursive=True):
    records = []

    for json_path in _iter_json_files(parent_folder, recursive=recursive):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        # Per-run payload files are dictionaries with run-level keys.
        if isinstance(payload, dict) and "env" in payload and "run" in payload:
            records.append(_normalize_record(payload))
            continue

        # summary.json style: {config_name: [run_record, ...], ...}
        if isinstance(payload, dict):
            for config_name, run_list in payload.items():
                if not isinstance(run_list, list):
                    continue
                for run_record in run_list:
                    if isinstance(run_record, dict) and "env" in run_record and "run" in run_record:
                        records.append(_normalize_record(run_record, fallback_config_name=config_name))

    if not records:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    return pd.DataFrame.from_records(records, columns=DEFAULT_COLUMNS)


# for this one, you may pass something like ["results/benchmark_deepseek_deepseek-v4-pro_2026-06-20_09h-34min", "results/benchmark_deepseek_deepseek-v4-pro_2026-06-21_10h-00min"]
# it will call create_dataframe_from_result_folder for each of the folders in the list and concatenate the resulting dataframes together
def create_dataframe_from_list_of_result_folders(list_of_folders, recursive=True):
    dataframes = [
        create_dataframe_from_result_folder(folder, recursive=recursive)
        for folder in list_of_folders
    ]
    non_empty_dataframes = [df for df in dataframes if not df.empty]

    if not non_empty_dataframes:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    return pd.concat(non_empty_dataframes, ignore_index=True)
