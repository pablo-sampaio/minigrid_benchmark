"""
Play MiniGrid in text mode from the console.

Example:
    python tools/play_text_minigrid.py
    python tools/play_text_minigrid.py --env MiniGrid-LavaCrossingS9N3-v0 --wrapper local --numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 - needed to register MiniGrid envs

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wrappers import MiniGridTextLocalObsWrapper, MiniGridTextGlobalObsWrapper  # noqa: E402

ACTION_MAP = {
    "GIRA_ANTI_HORARIO": 0,
    "GIRA_HORARIO": 1,
    "FRENTE": 2,
    "ESQUERDA": 0,
    "DIREITA": 1,
    "FRENTE_PT": 2,
    "LEFT": 0,
    "RIGHT": 1,
    "FORWARD": 2,
    "L": 0,
    "R": 1,
    "F": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play MiniGrid in text mode.")
    parser.add_argument("--env", default="MiniGrid-LavaGapS5-v0", help="Environment id.")
    parser.add_argument(
        "--wrapper",
        choices=["w1", "w2", "local"],
        default="local",
        help="Text wrapper to use.",
    )
    parser.add_argument(
        "--numbers",
        action="store_true",
        default=True,
        help="Show row/column labels (only used by w2/local).",
    )
    parser.add_argument(
        "--no-numbers",
        dest="numbers",
        action="store_false",
        help="Hide row/column labels (only used by w2/local).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reset seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max episode steps override.",
    )
    return parser.parse_args()


def build_wrapped_env(env_id: str, wrapper_name: str, show_numbers: bool, max_steps: int | None):
    env_kwargs = {}
    if max_steps is not None:
        env_kwargs["max_episode_steps"] = max_steps

    env = gym.make(env_id, **env_kwargs)
    if wrapper_name == "w1":
        return MiniGridTextWrapper1(env)
    if wrapper_name == "w2":
        return MiniGridTextWrapper2(env, show_numbers=show_numbers)
    return MiniGridTextLocalObsWrapper(env, show_numbers=show_numbers)


def normalize_action(user_text: str) -> int | None:
    key = user_text.strip().upper()
    if key == "FRENTE":
        return ACTION_MAP["FRENTE"]
    return ACTION_MAP.get(key)


def print_help() -> None:
    print("\nCommands:")
    print("  GIRA_ANTI_HORARIO | LEFT | L")
    print("  GIRA_HORARIO      | RIGHT | R")
    print("  FRENTE            | FORWARD | F")
    print("  reset  -> reset current env")
    print("  help   -> show this help")
    print("  quit   -> exit")


def run_loop() -> None:
    args = parse_args()
    env = build_wrapped_env(args.env, args.wrapper, args.numbers, args.max_steps)

    try:
        obs, _ = env.reset(seed=args.seed)
        print(f"\nEnvironment: {args.env}")
        print(f"Wrapper: {args.wrapper} | show_numbers={args.numbers}")
        print_help()

        step_idx = 0
        print("\nInitial observation:")
        print(obs)

        while True:
            raw = input("\nAction> ").strip()
            if not raw:
                continue

            lowered = raw.lower()
            if lowered in {"quit", "exit", "q"}:
                print("Exiting.")
                break
            if lowered in {"help", "h", "?"}:
                print_help()
                continue
            if lowered in {"reset", "rs"}:
                obs, _ = env.reset()
                step_idx = 0
                print("\nEnvironment reset.")
                print(obs)
                continue

            action = normalize_action(raw)
            if action is None:
                print("Unknown action. Type 'help' to see valid commands.")
                continue

            obs, reward, terminated, truncated, _ = env.step(action)
            step_idx += 1

            print(f"\nStep {step_idx} | action={raw.upper()} ({action})")
            print(obs)
            print(f"reward={reward:.6f} | terminated={terminated} | truncated={truncated}")

            if terminated or truncated:
                print("\nEpisode finished. Type 'reset' to play again or 'quit' to exit.")

    finally:
        env.close()


if __name__ == "__main__":
    run_loop()
