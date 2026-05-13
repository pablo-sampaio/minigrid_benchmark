import gymnasium as gym
import minigrid

# Requires
# !pip install -q minigrid==3.0.0

MAP_CELLS = {'wall': '#', 'floor': '.', 'goal': 'O', 'key': 'C', 'lava': 'L', 'open_door': '_', 'locked_dor': 'X', 'unlocked_closed_door': 'P'}
PLAYER_DIRECTIONS = ['>', 'v', '<', '^']

class MiniGridTextWrapper1(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.map_cells_repr = MAP_CELLS
        self.directions_repr = PLAYER_DIRECTIONS
        self.unknown = "?"

    def observation(self, obs):
        env = self.unwrapped
        agent_pos = env.agent_pos
        current_room = None
        if hasattr(env, 'rooms'):
            for room in env.rooms:
                top_x, top_y = room.top
                w, h = room.size
                if (top_x <= agent_pos[0] < top_x + w and top_y <= agent_pos[1] < top_y + h):
                    current_room = room
                    break

        x_s, y_s = (current_room.top, current_room.size) if current_room else ((0,0), (env.grid.width, env.grid.height))

        # Grid logic for the loop execution:
        grid_str = ""
        last_line = y_s[1] - 1
        for y in range(y_s[1]):
            line = ""
            for x in range(x_s[0], x_s[0] + y_s[0]):
                real_x, real_y = x, y + x_s[1]
                if [real_x, real_y] == list(agent_pos):
                    line += self.directions_repr[env.agent_dir]
                else:
                    tile = env.grid.get(real_x, real_y)
                    if tile is None:
                        line += self.map_cells_repr['floor']
                    elif tile.type == 'door':
                        line += (MAP_CELLS['open_door'] if tile.is_open else (MAP_CELLS['locked_dor'] if tile.is_locked else MAP_CELLS['unlocked_closed_door']))
                    else:
                        line += self.map_cells_repr.get(tile.type, self.unknown)
            grid_str += line + ("\n" if y != last_line else "")

        return grid_str

#MAP_CELLS = {'wall': '#', 'floor': '.', 'goal': 'O', 'key': 'C', 'lava': 'L', 'open_door': '_', 'locked_dor': 'X', 'unlocked_closed_door': 'P'}
#PLAYER_DIRECTIONS = ['>', 'v', '<', '^']

class MiniGridTextWrapper2(gym.ObservationWrapper):
    def __init__(self, env, show_numbers=False):
        super().__init__(env)
        self.map_cells_repr = MAP_CELLS
        self.directions_repr = PLAYER_DIRECTIONS
        self.unknown = "?"
        self.show_numbers = show_numbers

    def observation(self, obs):
        env = self.unwrapped
        agent_pos = env.agent_pos
        current_room = None
        if hasattr(env, 'rooms'):
            for room in env.rooms:
                top_x, top_y = room.top
                w, h = room.size
                if (top_x <= agent_pos[0] < top_x + w and top_y <= agent_pos[1] < top_y + h):
                    current_room = room
                    break

        x_s, y_s = (current_room.top, current_room.size) if current_room else ((0,0), (env.grid.width, env.grid.height))

        grid_str = ""

        # Add column letters header
        if self.show_numbers:
            header = "--"
            for x in range(y_s[0]):
                header += f"|{chr(65 + x)}"
            grid_str += header + "|\n"

        last_line = y_s[1] - 1
        for y in range(y_s[1]):
            line = ""
            if self.show_numbers:
                line += f"{y+1:02d}"

            for x in range(x_s[0], x_s[0] + y_s[0]):
                real_x, real_y = x, y + x_s[1]
                if [real_x, real_y] == list(agent_pos):
                    line += "|" + self.directions_repr[env.agent_dir]
                else:
                    tile = env.grid.get(real_x, real_y)
                    if tile is None:
                        line += "|" + self.map_cells_repr['floor']
                    elif tile.type == 'door':
                        line += "|" + (MAP_CELLS['open_door'] if tile.is_open else (MAP_CELLS['locked_dor'] if tile.is_locked else MAP_CELLS['unlocked_closed_door']))
                    else:
                        line += "|" + self.map_cells_repr.get(tile.type, self.unknown)
            grid_str += line + ("|\n" if y != last_line else "|")

        return grid_str

SYSTEM_PROMPT_WRAPPER_1 = """
Você é um agente ReAct que navega em um mapa quadriculado 2D (um "grid").
Seu objetivo é mover-se pelo mapa para alcançar a posição da célula objetivo.

# OBSERVAÇÃO:
- Você receberá a observação do mapa, em formato de texto.
- Cada linha do texto é uma linha do mapa.
- Cada célula é representada por um único caractere, seguindo esta representação:
    - `#` (parede)
    - `.` (chão, célula vazia)
    - `O` (posição do objetivo)
    - `L` (lava)
- Indicadores da orientação do agente: são colocados na representação do mapa para mostrar sua posição atual e a direção para a qual você está voltado.
    - Pode ser um dos seguintes: `^` (agente direcionado para cima), `v` (para baixo), `<` (para a esquerda) ou `>` (para a direita).

# AÇÕES DISPONÍVEIS:
- A cada passo, você deve escolher uma das seguintes ações:
    - GIRA_ANTI_HORARIO: gira 90 graus no sentido anti-horário.
    - GIRA_HORARIO: gira 90 graus no sentido horário.
    - FRENTE: move uma célula na direção para a qual você está voltado.

# FORMATO DA RESPOSTA:
Sua resposta deve ter apenas duas linhas e NADA MAIS. Não inclua texto conversacional.
THOUGHT: <Explique um breve raciocínio para justificar a escolha da próxima ação>
ACTION: <Sua próxima ação. Escolha uma das ações listadas acima>

## EXEMPLO DE RESPOSTA:
THOUGHT: Estou de frente para o objetivo, que está na próxima célula adiante.
ACTION: FRENTE
"""

SYSTEM_PROMPT_WRAPPER_2a = """
Você é um agente ReAct que navega em um mapa quadriculado 2D (um "grid").
Seu objetivo é mover-se pelo mapa para alcançar a posição da célula objetivo.

# OBSERVAÇÃO:
- Você receberá a observação do mapa, em formato de texto.
- Linhas são identificadas por números, e colunas são identificadas por letras maiúsculas.
- Cada célula é representada por um único caractere, seguindo esta representação:
    - `#` (parede, não pode ser ocupada pelo agente)
    - `.` (chão, célula vazia)
    - `O` (posição do objetivo)
    - `L` (lava, que é uma célula mortal)
- Um caractere `|` é usado entre células de uma mesma linha como separador.
- Indicadores da orientação do agente: são colocados na representação do mapa para mostrar sua posição atual e a direção para a qual você está voltado.
    - Pode ser um dos seguintes: `^` (agente direcionado para cima), `v` (para baixo), `<` (para a esquerda) ou `>` (para a direita).

# AÇÕES DISPONÍVEIS:
- A cada passo, você deve escolher uma das seguintes ações:
    - GIRA_ANTI_HORARIO: gira 90 graus no sentido anti-horário.
    - GIRA_HORARIO: gira 90 graus no sentido horário.
    - FRENTE: move uma célula na direção para a qual você está voltado.

# FORMATO DA RESPOSTA:
Sua resposta deve ter apenas duas linhas e NADA MAIS. Não inclua texto conversacional.
THOUGHT: <Explique um breve raciocínio para justificar a escolha da próxima ação>
ACTION: <Sua próxima ação. Escolha uma das ações listadas acima>

## EXEMPLO DE RESPOSTA:
THOUGHT: Estou de frente para o objetivo, que está na próxima célula adiante.
ACTION: FRENTE
"""

SYSTEM_PROMPT_WRAPPER_2d = """
You are a ReAct agent navigating a 2D grid map.
Your goal is to move through the map to reach the goal cell while avoiding lava cells.

# OBSERVATION:
- You will receive a text-based map observation from the user.
- Each line represents a row in the grid.
- Rows are identified by numbers, and columns by capital letters.
- Each cell is represented by a single character:
    * `#` (wall, a cell that you cannot ocupy)
    * `.` (floor, empty cell)
    * `O` (goal position)
    * `L` (lava, deadly cell)
    * orientation symbol (see below)
- An orientation symbol marks your current position and direction:
    * `^` (facing up)
    * `v` (facing down)
    * `<` (facing left)
    * `>` (facing right)
- The `|` character is used as a separator between cells.

# AVAILABLE ACTIONS:
- Choose one of the following actions at each step:
    * GIRA_HORARIO: rotate 90 degrees clockwise.
    * GIRA_ANTI_HORARIO: rotate 90 degrees counter-clockwise.
    * FRENTE: move one cell forward in the direction you are facing.

# YOUR MISSION
- Examine the observation carefully and output your next step using the XML tags.
- Think about you current orientation, the next cell you want to reach, what cells
are safe or not, and which ones lead you to the goal

# RESPONSE FORMAT:
Your response MUST follow this XML format exactly:
<thought>
Your reasoning for the next action.
</thought>
<action>
One of the actions listed above.
</action>

## EXAMPLES

# Observation (user message sent to you)

OBSERVAÇÃO ATUAL:
--|A|B|C|D|
01|#|#|#|#|
02|#|L|O|#|
03|#|.|.|#|
04|#|v|.|#|
05|#|#|#|#|

# Response (to be sent from you to the user)

<thought>
I am at cell row 04 column B and I am facing down (`v`). To reach the goal at row 02 column C, I need to change my orientation to face right `>`. To change my orientation from 'down' to 'right', I will turn counter-clockwise.
</thought>
<action>
GIRA_ANTI_HORARIO
</action>

# FINAL COMMENT
Examine the observation, think about it and output your next step using the XML tags.
"""


OBS_TEMPLATE = """
OBSERVAÇÃO ATUAL:
{SALA_ATUAL}
"""


OBS_TEMPLATE_ENG = """
CURRENT OBSERVATION:
{SALA_ATUAL}
"""
