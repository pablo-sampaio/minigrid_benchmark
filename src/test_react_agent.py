"""
Manual smoke-check for ReActAgent against a real OpenAI model.

Not an automated unit test: it makes real API calls and needs an API key,
so it only runs when executed directly (`python test_react_agent.py`).
"""

import os
import time

import gymnasium as gym
from langchain_openai import ChatOpenAI

import wrappers
from react_agent import ReActAgent

N = 3


def main() -> None:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    openai_model = ChatOpenAI(
        model="gpt-5.4-mini",
        api_key=openai_api_key,
        temperature=0,
        max_retries=2,
    )

    original_env = gym.make("MiniGrid-LavaGapS5-v0")
    #original_env = gym.make("MiniGrid-LavaCrossingS9N3-v0")

    agente = ReActAgent(openai_model, wrappers.prompts.SYSTEM_PROMPT_GLOBAL_2, wrappers.prompts.OBS_TEMPLATE, verbose=True)
    env = wrappers.MiniGridTextGlobalObsWrapper(original_env, show_numbers=True, separate_cells=True)

    sucessos = 0

    for i in range(N):
        time.sleep(0.5)

        print(f"Execução {i+1} de {N} (current success: {sucessos})")
        initial_obs, _ = env.reset()
        recompensa = agente.solve_environment(env, initial_obs)

        if recompensa > 0:
            sucessos += 1
        print(f"Quantidade de sucessos (parcial): {sucessos}")

    print(f"SUCESSOS: {sucessos} de {N}")


if __name__ == "__main__":
    main()
