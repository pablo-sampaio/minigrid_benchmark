from collections import deque
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class SlidingWindowChatHistory:
    """
    Mantem um historico de mensagens multi-turno com janela deslizante.
    O historico interno e sempre completo; a janela e aplicada apenas
    nos metodos de leitura para envio ao modelo.
    """

    def __init__(self, system_prompt: str, window_size: int = 3):
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size deve ser um inteiro impar >= 1 (recebido: {window_size}). "
                "A janela precisa iniciar e terminar com mensagem do usuario."
            )

        self.system_prompt = system_prompt
        self.window_size = window_size  # total de mensagens na janela (impar: inicia e termina com 'human')
        self._messages = deque()  # historico completo de turnos user/assistant

    def add_message(self, role: str, content: str):
        """
        Adiciona uma mensagem ao historico mantendo alternancia user/assistant.
        Nao trunca o historico; a janela e aplicada apenas nos getters.
        """
        if role not in ("human", "ai"):
            raise ValueError(f"Role invalido: {role}")

        last_role = self._messages[-1]["role"] if self._messages else None

        if role == "human":
            if last_role not in (None, "ai"):
                raise ValueError("Esperava turno 'ai' antes de novo 'human'.")
        else:  # ai
            if last_role != "human":
                raise ValueError("Esperava turno 'human' antes de 'ai'.")

        self._messages.append({"role": role, "content": content})

    def _windowed_messages(self) -> list[dict]:
        """
        Retorna a janela para envio ao modelo:
        - A ultima mensagem precisa ser do usuario ('human').
        - Retorna as ultimas window_size mensagens no total,
          sendo a primeira e a ultima sempre do usuario.
        """
        if not self._messages:
            raise ValueError("O historico esta vazio. Precisa haver uma mensagem do usuario.")

        if self._messages[-1]["role"] != "human":
            raise ValueError("A ultima mensagem deve ser do usuario ('human') para montar a janela.")

        return list(self._messages)[-(self.window_size):]

    def get_messages(self) -> list[dict]:
        """Retorna system prompt + janela configurada do historico."""
        return [{"role": "system", "content": self.system_prompt}] + self._windowed_messages()

    def get_langchain_messages(self) -> list[SystemMessage | HumanMessage | AIMessage]:
        """Retorna mensagens tipadas do LangChain com a janela configurada."""
        messages = [SystemMessage(content=self.system_prompt)]
        for msg in self._windowed_messages():
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def get_full_history(self) -> list[dict]:
        """Retorna o historico completo de mensagens user/assistant."""
        return list(self._messages)

    def save_full_history_json(self, file_path: str) -> str:
        """
        Salva o historico completo em JSON (chamando get_full_history)
        e retorna o caminho salvo.
        """
        payload = {
            "system_prompt": self.system_prompt,
            "messages": self.get_full_history(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return file_path

    def reset(self):
        """Limpa o historico (novo episodio). Porém, não remove o system prompt."""
        self._messages.clear()

    def __len__(self):
        """Retorna o numero de mensagens totais do historico."""
        return len(self._messages)
