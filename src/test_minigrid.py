from time import sleep
import gymnasium as gym
import minigrid
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

IDX_TO_STATE = {
    0: "open",
    1: "closed",
    2: "locked"
}

def explain(obs):
    '''
    Textually explains the object, color and state of each part of the observation grid, grouped by visual row.
    '''
    print("\n=== OBSERVATION ===")
    height = obs.shape[1]  # depth of agent's vision
    width = obs.shape[0]

    for row in range(height):
        print(f"Row {row}:")
        for col in range(width):
            cell = obs[col, row]
            obj_type = cell[0]
            color = cell[1]
            state = cell[2]
            print(
                f"  Column {col}: Object={IDX_TO_OBJECT[obj_type]},",
                f"Color={IDX_TO_COLOR[color]}, State={IDX_TO_STATE[state]}",
                ("[agent position]" if (row == height-1 and col == width // 2) else "")
            )

if __name__ == '__main__':
    #env = gym.make("MiniGrid-LavaGapS5-v0", render_mode="human", max_episode_steps=20)
    env = gym.make("MiniGrid-LavaCrossingS9N3-v0", render_mode="human", max_episode_steps=20)
    
    obs, _ = env.reset()
    explain(obs['image'])
    done = False
    
    # do some random actions
    while not done:
        #action = env.action_space.sample()
        action = int(input("Enter action (0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle): "))
        sleep(2)
        obs, reward, terminated, truncated, info = env.step(action)
        explain(obs['image'])
        done = terminated or truncated
        #done = True

    sleep(20)