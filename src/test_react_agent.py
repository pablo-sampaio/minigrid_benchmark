
import os
from asyncio import sleep

import gymnasium as gym
from langchain_openai import ChatOpenAI

from wrappers import MiniGridTextWrapper2, SYSTEM_PROMPT_WRAPPER_2d, OBS_TEMPLATE_ENG
from react_agent import ReActAgent

OPENAI_API_KEY = os.getenv("OPENAI_PABLO") #"OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

OPENAI_MODEL = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_retries=2,
)

N = 3

original_env = gym.make("MiniGrid-LavaGapS5-v0") 
#original_env = gym.make("MiniGrid-LavaCrossingS9N3-v0")

agente = ReActAgent(OPENAI_MODEL, SYSTEM_PROMPT_WRAPPER_2d, OBS_TEMPLATE_ENG, verbose=True)
env = MiniGridTextWrapper2(original_env, show_numbers=True)

SUCESSOS = 0

for i in range(N):
    sleep(0.5)

    print(f"Execução {i+1} de {N} (current success: {SUCESSOS})")
    initial_obs, _ = env.reset()
    recompensa = agente.solve_environment(env, initial_obs)

    if recompensa > 0:
        SUCESSOS += 1
    print(f"Quantidade de sucessos (parcial): {SUCESSOS}")


print(f"SUCESSOS: {SUCESSOS} de {N}") 
