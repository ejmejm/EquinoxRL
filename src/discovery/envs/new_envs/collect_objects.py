# Source: https://github.com/Somjit77/oc_gvfs/blob/main/pygame_gridworld.py

import time

import gymnasium as gym
import numpy as np

from .pygame_gridworld import PyGame


RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
GREY = (100, 100, 100)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)


class CollectObjects:
    def __init__(self, seed, breadth=7, length=7, turn_limit=40):
        np.random.seed(seed)
        # Environment Details
        self.breadth = breadth
        self.length = length
        self.step_reward = -0.01
        self.turn_limit = turn_limit
        self.actions = [0, 1, 2, 3]  # UP, DOWN, RIGHT, LEFT
        self.num_actions = len(self.actions)
        self.goals = [(1, self.breadth)]
        self.obstacles = []
        self.final_goal = (self.breadth, self.length)
        # Initialize State Space
        self.state_space = np.empty([self.length, self.breadth], dtype='<U1')
        self.colors = (RED, BLUE, YELLOW)
        self.final_limits=(self.length//2+2, self.length+1)
        self.reset()

    def add_goal(self):
        self.goals=[(1, self.breadth), (self.length, self.breadth)]
    
    def add_obstacle(self):
        self.obstacles = [(3*self.breadth//4+1, 3*self.length//4+1)]

    def convertxytoij(self, xy):
        return self.length - xy[1], xy[0] - 1

    def reset(self, random=False):
        self.state_space[:] = 'E'
        # Add Corridors
        corridor_position = (self.breadth // 2 + 1, self.length // 2 + 1)
        i, j = self.convertxytoij(corridor_position)
        self.state_space[:, j] = 'X'  # Vertical Corridor
        self.state_space[i, :] = 'X'  # Horizontal Corridor
        corridor_positions = [(self.breadth//4+1,self.length//2+1), (3*self.breadth//4+1,self.length//2+1), 
                              (self.breadth//2+1,self.length//4+1), (self.breadth//2+1,3*self.length//4+1)]
        for corr_pos in corridor_positions:
            i,j = self.convertxytoij(corr_pos)
            self.state_space[i, j] = 'E'
        #Staring Position
        self.starting_position = (1, 1)#(np.random.randint(1, self.breadth+1), np.random.randint(1, self.length+1))#(1, 1)  # Bottom left corner is (1,1)
        i, j = self.convertxytoij(self.starting_position)
        self.state_space[i, j] = 'P'
        #Final Goal
        if random:
            up_limit = self.breadth+1
            low_limit = self.breadth//2+2
            self.final_goal = (np.random.randint(low_limit, up_limit), np.random.randint(self.final_limits[0], self.final_limits[1])) # (self.breadth, 1)
        i, j = self.convertxytoij(self.final_goal)
        self.state_space[i, j] = 'G'
        # Red Goals
        cur_goals = [(0,0) for _ in range(len(self.goals))]
        for idx in range(len(self.goals)):
            if random:
                placed_obj = False
                while not placed_obj:
                    if self.goals[idx][0] == self.breadth:
                        up_limit = self.breadth+1
                        low_limit = self.breadth//2+2
                        cur_goals[idx] = (np.random.randint(low_limit, up_limit), np.random.randint(self.length//2+2, self.length+1))
                    else:
                        low_limit = 1
                        up_limit = self.breadth//2+1
                        cur_goals[idx] = (np.random.randint(low_limit, up_limit), np.random.randint(self.length//2+2, self.length+1))
                    i, j = self.convertxytoij(cur_goals[idx])
                    if self.state_space[i, j] == 'E':
                        self.state_space[i, j] = 'R'
                        placed_obj = True
            else:
                cur_goals[idx] = self.goals[idx]
            i, j = self.convertxytoij(cur_goals[idx])
            self.state_space[i, j] = 'R'
        # Obstacles
        for idx in range(len(self.obstacles)):
            placed_obj = False
            while not placed_obj:
                if random:
                    up_limit = self.breadth+1
                    low_limit = self.breadth//2+2
                    self.obstacles[idx] = (np.random.randint(low_limit, up_limit), np.random.randint(1, self.length//2+1))
                    i, j = self.convertxytoij(self.obstacles[idx])
                    if self.state_space[i,j] == 'E':
                        self.state_space[i,j] = 'B'
                        placed_obj = True
                else:
                    i, j = self.convertxytoij(self.obstacles[idx])
                    self.state_space[i, j] = 'B'
                    placed_obj = True
        self.steps = 0
        self.done = False
        self.visited_red_dot = [False for _ in range(len(self.goals))]
        self.state = self.starting_position + tuple(map(int, self.visited_red_dot))
        # Initialize PyGame
        self.pygame = PyGame(window_height=self.length, window_width=self.breadth, goals=cur_goals, 
                            final_goal=self.final_goal, obstacles = self.obstacles, agent_pos=self.starting_position)
        self.image = self.pygame.drawGrid(self.colors)
        self.observation_space = self.image.shape
        return {'image':self.image, 'position': self.state}

    def step(self, action):
        if self.done:
            raise Exception('Calling step after done=True')
        indices = self.convertxytoij([self.state[0], self.state[1]])
        next_obj, next_indices, next_state = self.next_position(action, indices)
        truncated = False
        if next_obj == 'E': # Empty
            self.state_space[indices[0], indices[1]] = 'E'
            self.state_space[next_indices[0], next_indices[1]] = 'P'
            self.image = self.pygame.move_agent(self.state, next_state)
            self.state = next_state + tuple(map(int, self.visited_red_dot))
            reward = self.step_reward
        elif next_obj == 'R': # 1st Goal
            self.state_space[indices[0], indices[1]] = 'E'
            self.state_space[next_indices[0], next_indices[1]] = 'P'
            self.image = self.pygame.move_agent(self.state, next_state)
            for idx in range(len(self.visited_red_dot)):
                if self.visited_red_dot[idx] == False:
                    self.visited_red_dot[idx] = True
            self.state = next_state + tuple(map(int, self.visited_red_dot))
            reward = 5.0 + self.step_reward
        elif next_obj == 'G': # Final Goal
            self.state_space[indices[0], indices[1]] = 'E'
            self.state_space[next_indices[0], next_indices[1]] = 'P'
            self.image = self.pygame.move_agent(self.state, next_state)
            reward = 0.0 + self.step_reward # Get no reward for visiting G without R
            if all(self.visited_red_dot):
                reward = 10.0 + self.step_reward
            self.state = next_state + tuple(map(int, self.visited_red_dot))
            self.done = True
        elif next_obj == 'B': # Obstacle
            self.state = self.state
            reward = -1.0 + self.step_reward
            self.done = True
        elif next_obj == 'X': # Wall
            self.state = self.state
            reward = self.step_reward
        else:
            raise Exception('Unknown Object')
        self.steps += 1
        if self.steps == self.turn_limit:
            truncated = True
            self.done = True
        
        return {'image':self.image, 'position': self.state}, reward, self.done, truncated

    def next_position(self, action, indices):
        '''Get Next Position for Empty Grid'''
        if action == 0:
            if self.state[1] + 1 > self.length:
                return 'X', indices, self.state
            next_state = (self.state[0], self.state[1] + 1)
            next_indices = self.convertxytoij([next_state[0], next_state[1]])
            return self.state_space[next_indices[0], next_indices[1]], next_indices, next_state
        elif action == 1:
            if self.state[1] - 1 < 1:
                return 'X', indices, self.state
            next_state = (self.state[0], self.state[1] - 1)
            next_indices = self.convertxytoij([next_state[0], next_state[1]])
            return self.state_space[next_indices[0], next_indices[1]], next_indices, next_state
        elif action == 2:
            if self.state[0] + 1 > self.breadth:
                return 'X', indices, self.state
            next_state = (self.state[0] + 1, self.state[1])
            next_indices = self.convertxytoij([next_state[0], next_state[1]])
            return self.state_space[next_indices[0], next_indices[1]], next_indices, next_state
        elif action == 3:
            if self.state[0] - 1 < 1:
                return 'X', indices, self.state
            next_state = (self.state[0] - 1, self.state[1])
            next_indices = self.convertxytoij([next_state[0], next_state[1]])
            return self.state_space[next_indices[0], next_indices[1]], next_indices, next_state
        else:
            raise Exception('Invalid Action')

    def render(self):
        for i in range(self.length):
            dash = "-"
            print(int(4 * self.breadth - 1) * dash)
            for j in range(self.breadth):
                print(self.state_space[i][j], end=" | ")
            print("")
        import matplotlib.pyplot as plt
        plt.imshow(self.image)
        plt.axis('off')
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.savefig('test0.png')


class CollectObjectsEnv(gym.Env):
    def __init__(self, seed=None, breadth=7, length=7):
        if seed is None:
            seed = int(time.time())
        self._env = CollectObjects(seed, breadth, length)
        self.tile_size = self._env.pygame.scale
        
        self.observation_space = gym.spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (self.tile_size * self._env.length, self.tile_size * self._env.breadth, 3),
            dtype = np.float32,
        )
        self.action_space = gym.spaces.Discrete(4)
    
    def reset(self, seed=None, random=False):
        obs = np.asarray(self._env.reset(random=random)['image'], dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs, reward, done, truncated = self._env.step(action)
        obs = np.asarray(obs['image'], dtype=np.float32)
        return obs, float(reward), done, truncated, {} # Position not implemented

    def render(self):
        return self._env.render()


# Simple test to instantiate and try the functions
if __name__ == '__main__':
    # Instantiate the environment
    env = CollectObjectsEnv(seed=0, breadth=7, length=7)
    
    # # Add gym wrappers to change to 0-1 range and move channels to first dimension
    # env = gym.wrappers.GrayScaleObservation(env)

    # Reset the environment and get initial observation
    obs = env.reset()[0] * 255.0
    print("Initial observation shape:", obs.shape)
    # Print range:
    print(f"Range: {obs.min()} to {obs.max()}")

    # Try a step
    action = 0  # Assuming 0 is a valid action
    next_obs, reward, done, truncated, info = env.step(action)
    next_obs = next_obs * 255.0
    print("Step result:")
    print("- Observation shape:", next_obs.shape)
    print("- Reward:", reward)
    print("- Done:", done)
    print("- Truncated:", truncated)

    # Render the environment
    env.render()

    # Display the image
    import matplotlib.pyplot as plt
    plt.imshow(next_obs)
    plt.title("Environment Observation")
    plt.show()

