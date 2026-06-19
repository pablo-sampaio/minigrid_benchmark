import gymnasium as gym
from matplotlib.pyplot import grid
import minigrid
import numpy as np

# Requires
# !pip install -q minigrid==3.0.0

MAP_CELLS = {'wall': '#', 'floor': '.', 'goal': 'G', 'key': 'K', 'lava': 'L', 'open_door': '_', 'unlocked_closed_door': 'D', 'locked_dor': 'X'}
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


class MiniGridTextLocalObsWrapper(gym.ObservationWrapper):
    
    def __init__(self, env, show_numbers=False, separate_cells=True, show_direction=False):
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


import wrappers_react_prompts as prompts
