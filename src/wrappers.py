import gymnasium as gym
from matplotlib.pyplot import grid
import minigrid
import numpy as np

# Requires
# !pip install -q minigrid==3.0.0

MAP_CELLS = {'wall': '#', 'floor': '.', 'goal': 'O', 'key': 'C', 'lava': 'L', 'open_door': '_', 'locked_dor': 'X', 'unlocked_closed_door': 'P'}
PLAYER_DIRECTIONS = ['>', 'v', '<', '^']
PLAYER_DIRECTIONS_DESCR = ['East', 'South', 'West', 'North']
OBJECT_IDX_TO_TYPE = {
    0: 'unseen',
    1: 'empty',
    2: 'wall',
    3: 'floor',
    4: 'door',
    5: 'key',
    6: 'ball',
    7: 'box',
    8: 'goal',
    9: 'lava',
    10: 'agent',
}

class MiniGridTextGlobalObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, show_numbers=False, separate_cells=True):
        super().__init__(env)
        self.map_cells_repr = MAP_CELLS
        self.directions_repr = PLAYER_DIRECTIONS
        self.unknown = "?"
        self.show_numbers = show_numbers
        self.separate_cells = separate_cells

    def _render_cell(self, env, x, y):
        if [x, y] == list(env.agent_pos):
            return self.directions_repr[env.agent_dir]

        tile = env.grid.get(x, y)
        if tile is None:
            return self.map_cells_repr['floor']
        if tile.type == 'door':
            return self.map_cells_repr['open_door'] if tile.is_open else (self.map_cells_repr['locked_dor'] if tile.is_locked else self.map_cells_repr['unlocked_closed_door'])

        return self.map_cells_repr.get(tile.type, self.unknown)

    def _get_view_bounds(self, env):
        agent_pos = env.agent_pos
        current_room = None
        if hasattr(env, 'rooms'):
            for room in env.rooms:
                top_x, top_y = room.top
                w, h = room.size
                if top_x <= agent_pos[0] < top_x + w and top_y <= agent_pos[1] < top_y + h:
                    current_room = room
                    break

        if current_room is None:
            return 0, 0, env.grid.width, env.grid.height

        start_x, start_y = current_room.top
        width, height = current_room.size
        return start_x, start_y, width, height

    def observation(self, obs):
        env = self.unwrapped
        start_x, start_y, width, height = self._get_view_bounds(env)
        output_lines = []

        if self.show_numbers:
            header = ['--']
            if self.separate_cells:
                header.append('|')
            for x in range(width):
                header.append(f"{chr(65 + x)}")
                if self.separate_cells:
                    header.append('|')
            output_lines.append(header)

        for row in range(height):
            if self.show_numbers:
                line_cells = [f"{(row + 1):02d}"]
            else:
                line_cells = []

            if self.separate_cells:
                line_cells.append('|')

            for col in range(width):
                line_cells.append(self._render_cell(env, start_x + col, start_y + row))
                if self.separate_cells:
                    line_cells.append('|')

            output_lines.append(line_cells)

        return "\n".join("".join(line_cells) for line_cells in output_lines)


class MiniGridTextWrapper1(MiniGridTextGlobalObsWrapper):
    def __init__(self, env):
        super().__init__(env, show_numbers=False, separate_cells=False)


class MiniGridTextWrapper2(MiniGridTextGlobalObsWrapper):
    def __init__(self, env, show_numbers=False):
        super().__init__(env, show_numbers=show_numbers, separate_cells=True)


class MiniGridTextLocalObsWrapper(gym.ObservationWrapper):
    
    def __init__(self, env, show_numbers=False, separate_cells=True, show_direction=True):
        super().__init__(env)
        self.map_cells_repr = MAP_CELLS
        self.unknown = "?"
        self.show_numbers = show_numbers
        self.separate_cells = separate_cells
        self.show_direction = show_direction

    def _decode_cell(self, encoded_cell):
        obj_idx, _, state_idx = int(encoded_cell[0]), int(encoded_cell[1]), int(encoded_cell[2])
        obj_type = OBJECT_IDX_TO_TYPE.get(obj_idx)

        if obj_type in (None, 'unseen'):
            return self.unknown
        if obj_type in ('empty', 'floor'):
            return self.map_cells_repr['floor']
        if obj_type == 'door':
            if state_idx == 0:
                return self.map_cells_repr['open_door']
            if state_idx == 2:
                return self.map_cells_repr['locked_dor']
            return self.map_cells_repr['unlocked_closed_door']

        return self.map_cells_repr.get(obj_type, self.unknown)

    def observation(self, obs):
        obs_image = obs['image']
        target_size = int(getattr(self.unwrapped, 'agent_view_size', obs_image.shape[0]))
        assert obs_image.shape[0] == target_size and obs_image.shape[1] == target_size, (
            f"Expected local observation shape ({target_size}, {target_size}, C), "
            f"got {obs_image.shape}"
        )
        # MiniGrid local observation uses [col, row, channel].
        view_width, view_height = obs_image.shape[0], obs_image.shape[1]

        output_lines = []
        
        # prepare header with column letters if show_numbers is True
        if self.show_numbers:
            header = ['--']
            if self.separate_cells:
                header.append('|')
            for x in range(view_width):
                header.append(f"{chr(65 + x)}")
                if self.separate_cells:
                    header.append('|')
            output_lines.append(header)

        # the lines describing the grid cells
        for row in range(view_height):
            if self.show_numbers:
                line_cells = [f"{(row + 1):02d}"]
            else:
                line_cells = []

            if self.separate_cells:
                line_cells.append('|')
            
            for col in range(view_width):
                # In MiniGrid partial observation, the agent is at the bottom-center of the view.
                if col == view_width // 2 and row == view_height - 1:
                    line_cells.append('^')
                else:
                    line_cells.append(self._decode_cell(obs_image[col, row]))
                
                if self.separate_cells:
                    line_cells.append('|')
            
            output_lines.append(line_cells)

        obs_str = "\n".join("".join(line_cells) for line_cells in output_lines)

        if self.show_direction:
            direction_idx = int(obs.get('direction', getattr(self.unwrapped, 'agent_dir', 0)))
            direction_str = PLAYER_DIRECTIONS_DESCR[direction_idx % len(PLAYER_DIRECTIONS_DESCR)]
            obs_str = f"direction: {direction_str}\n\n" + obs_str

        return obs_str


# for: global view, w/out separators or numbers
SYSTEM_PROMPT_GLOBAL_1 = """
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

# for: global view, with separators and numbers (in English)
SYSTEM_PROMPT_GLOBAL_2 = """
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

# TODO: FALTA informar a observação da direção (absoluta) do agente
# for: local view, with separators and numbers (in English) -- era 3a
SYSTEM_PROMPT_LOCAL_2 = """
You are a ReAct agent navigating a 2D grid, using a local view.
Your goal is to move through the map to reach the goal cell while avoiding lava cells.

# OBSERVATION:
- You will receive a text-based local map observation from the user.
- The observation is egocentric and centered on your current view.
- The agent is shown at the bottom-center of the view as `^`.
- Numbers are used to label rows, and letters are used to label columns.
- Each cell is represented by a single character:
    * `#` (wall, a cell that you cannot occupy)
    * `.` (floor, empty cell)
    * `O` (goal position)
    * `L` (lava, deadly cell)
    * `?` (unseen or unknown cell)
- The `|` character is used as a separator between cells.

# AVAILABLE ACTIONS:
- Choose one of the following actions at each step:
    * GIRA_HORARIO: rotate 90 degrees clockwise.
    * GIRA_ANTI_HORARIO: rotate 90 degrees counter-clockwise.
    * FRENTE: move one cell forward in the direction you are facing.

# YOUR MISSION
- Examine the observation carefully and output your next step using the XML tags.
- Think about your current local view, the next cell you want to reach, what cells
are safe or not, and which ones lead you to the goal.

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

CURRENT OBSERVATION:
--|A|B|C|D|E|F|G|
01|?|?|?|?|?|?|?|
02|?|?|?|?|?|?|?|
03|?|?|?|?|?|?|?|
04|?|?|#|#|#|#|#|
05|?|?|#|.|.|O|#|
06|?|?|#|L|.|L|#|
07|?|?|#|^|.|.|#|

# Response (to be sent from you to the user)

<thought>
The goal is visible at row 05 column E, three cells ahead and three cells to the right of my location. The cell in front of me is a deadly lava cell, so I cannot move forward. I will turn clockwise to head to the free neighbor cell at my right.
</thought>
<action>
GIRA_HORARIO
</action>

# FINAL COMMENT
Examine the observation, think about it and output your next step using the XML tags.
"""

# TODO: FALTA informar a observação da direção (absoluta) do agente
# for: local view, without separators or numbers (in English)
SYSTEM_PROMPT_LOCAL_1 = """
You are a ReAct agent navigating a 2D grid, using a local view.
Your goal is to move through the map to reach the goal cell while avoiding lava cells.

# OBSERVATION:
- You will receive a text-based local map observation from the user.
- The observation is egocentric and centered on your current view.
- The agent is shown at the bottom-center of the view as `^`.
- Each cell is represented by a single character:
    * `#` (wall, a cell that you cannot occupy)
    * `.` (floor, empty cell)
    * `O` (goal position)
    * `L` (lava, deadly cell)
    * `?` (unseen or unknown cell)

# AVAILABLE ACTIONS:
- Choose one of the following actions at each step:
    * GIRA_HORARIO: rotate 90 degrees clockwise.
    * GIRA_ANTI_HORARIO: rotate 90 degrees counter-clockwise.
    * FRENTE: move one cell forward in the direction you are facing.

# YOUR MISSION
- Examine the observation carefully and output your next step using the XML tags.
- Think about your current local view, the next cell you want to reach, what cells
are safe or not, and which ones lead you to the goal.

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

CURRENT OBSERVATION:
???????
???????
???????
??#####
??#..O#
??#L.L#
??#^..#

# Response (to be sent from you to the user)

<thought>
The goal O is visible three cells ahead and three cells to the right of my location. The cell in front of me is a deadly lava cell, so I cannot move forward. I will turn clockwise to head to the free neighbor cell at my right.
</thought>
<action>
GIRA_HORARIO
</action>

# FINAL COMMENT
Examine the observation, think about it and output your next step using the XML tags.
"""


OBS_TEMPLATE = """
CURRENT OBSERVATION:
{SALA_ATUAL}
"""
