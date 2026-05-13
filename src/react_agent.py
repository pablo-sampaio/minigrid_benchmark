import re
import time
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from chat_history import SlidingWindowChatHistory


ACTION_LOOKUP = {
    "GIRA_ANTI_HORARIO": 0,
    "GIRA_HORARIO": 1,
    "FRENTE": 2,
}


def _normalize_action_name(action_text: str) -> str:
    action = action_text.strip().upper()
    action = action.replace("Á", "A").replace("Ã", "A").replace("Ç", "C")
    # Keep only action token characters so trailing punctuation/formatting does not break parsing.
    return re.sub(r"[^A-Z_]", "", action)


def extract_thought_and_action(response_text):
    # Try XML format first
    thought_xml = re.search(r"<thought>(.*?)</thought>", response_text, re.DOTALL)
    action_xml = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL)

    if thought_xml and action_xml:
        thought_str = thought_xml.group(1).strip()
        action_str = _normalize_action_name(action_xml.group(1))
        return thought_str, action_str

    # Fallback to legacy text formats
    thought_match = re.search(r"(?:\*\*)?(?:THOUGHT|PENSAMENTO)(?:\*\*)?\s*:\s*(.*)", response_text, re.IGNORECASE)
    thought_str = thought_match.group(1).strip() if thought_match else "(not found)"

    action_match = re.search(
        r"(?:\*\*)?(?:ACTION|AÇÃO)(?:\*\*)?\s*:\s*(GIRA_ANTI_HORARIO|GIRA_ANTI_HORÁRIO|GIRA_HORARIO|GIRA_HORÁRIO|FRENTE)\b",
        response_text,
        re.IGNORECASE,
    )
    if action_match:
        action_str = _normalize_action_name(action_match.group(1))
        return thought_str, action_str

    return thought_str, ""


def langchain_response_to_text(response: AIMessage) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_chunks = []
        for chunk in content:
            if isinstance(chunk, str):
                text_chunks.append(chunk)
            elif isinstance(chunk, dict) and chunk.get("type") == "text":
                text_value = chunk.get("text", "")
                if text_value:
                    text_chunks.append(text_value)
        return "\n".join(text_chunks).strip()

    return str(content)


class ReActAgent:
    def __init__(self, model: BaseChatModel, system_prompt: str, obs_prompt_template: str, history_window: int = 1, verbose: bool = False):
        #if model is None:
        #    raise ValueError("model must be a LangChain BaseChatModel instance, got None")
        self.model = model
        self.system_prompt = system_prompt
        self.observation_template = obs_prompt_template
        self.verbose = verbose
        self.history = SlidingWindowChatHistory(system_prompt, window_size=history_window)

    def generate_model_response(self, obs_prompt, max_retries=3):
        """
        Generate a response from the model with error handling and retry logic.
        
        Args:
            obs_prompt: The observation prompt to send to the model
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            The model's response as text
            
        Raises:
            RuntimeError: If the API call fails after all retries
        """
        self.history.add_message("human", obs_prompt)
        
        for attempt in range(max_retries):
            try:
                response = self.model.invoke(self.history.get_messages())
                response = langchain_response_to_text(response)
                self.history.add_message("ai", response)
                return response
                
            except Exception as e:
                error_msg = f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                
                if self.verbose:
                    print(f"ERROR: {error_msg}")
                
                # If this was the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get response from model after {max_retries} attempts. Last error: {str(e)}") from e
                
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt
                if self.verbose:
                    print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    def get_full_history(self):
        return self.history.get_full_history()

    def solve_environment(self, env, initial_obs, max_steps=25):
        terminated = False
        truncated = False
        self.step_count = 0
        reward = 0

        # Loop Principal
        if self.verbose:
            print("Running ReAct Agent loop...\n")
            print("-" * 20)
            print("OBSERVAÇÃO INICIAL:")
            print(initial_obs)
            print("-" * 20)

        self.history.reset()
        obs = initial_obs
        episode_reward = 0.0

        while not (terminated or truncated) and self.step_count < max_steps:
            # 1. Prepare the prompt with current observation
            obs_prompt = self.observation_template.format(SALA_ATUAL=obs)

            # 2. Get Response from Language Model
            try:
                response_text = self.generate_model_response(obs_prompt)
            except RuntimeError as e:
                if self.verbose:
                    print(f"ERROR: Failed to get response from model: {e}")
                # Propagate to caller so it can decide based on an episode-level policy.
                raise
            
            self.step_count += 1

            # 3. Parse Thought and Action from the string response
            thought, action_str = extract_thought_and_action(response_text)

            if self.verbose:
                print("-" * 20)
                print(f"PASSO {self.step_count}:")

            if action_str == "":
                if self.verbose:
                    print(">> Erro: A resposta do modelo não está no formato correto.")
                    print(f">> Resposta: {response_text}")
                # volte para o início do loop
                continue

            if self.verbose:
                print(" - PENSAMENTO DO MODELO:", thought)
                print(" - ACAO ESCOLHIDA:", action_str)

            if action_str not in ACTION_LOOKUP:
                if self.verbose:
                    print(f">> Erro: Ação não reconhecida: {action_str}")
                # volte para o início do loop
                continue

            # 4. Execute action
            action_number = ACTION_LOOKUP[action_str]
            obs, reward, terminated, truncated, _ = env.step(action_number)
            episode_reward += reward

            # 5. Print the new resulting observation
            if self.verbose:
                print("-" * 20)
                print("NOVA OBSERVAÇÃO:")
                print(obs)
                print("-" * 20)
        
        if reward > 0.0 and self.verbose:
            print("SUCCESS!!!")
            print(f"Episode reward: {episode_reward}")
            #print(f"Last reward: {reward}")

        return episode_reward
