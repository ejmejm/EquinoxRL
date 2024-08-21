# Source: https://github.com/Somjit77/oc_gvfs/blob/main/pygame_gridworld.py

import contextlib

with contextlib.redirect_stdout(None):  # Suppress Hello from Pygame community message
    import pygame  # Warm hello back to the Pygame community by the way.
import numpy as np
from math import sin, cos, pi, radians

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
GREY = (100, 100, 100)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)


class PyGame:
    def __init__(self, window_height, window_width, goals, final_goal, obstacles, agent_pos, scale=8):
        self.scale = scale
        self.window_height = window_height
        self.window_width = window_width
        self.WINDOW_HEIGHT_SCALED = window_height * scale
        self.WINDOW_WIDTH_SCALED = window_width * scale
        self.blockSize = scale * 1  # Set the size of the grid block
        self.goals = goals
        self.final_goal = final_goal
        self.agent_pos = agent_pos
        self.obstacles = obstacles
        self.screen = pygame.Surface((self.WINDOW_HEIGHT_SCALED, self.WINDOW_WIDTH_SCALED))

    def change_goals(self, goals, obstacles):
        self.goals = goals
        self.obstacles = obstacles

    def drawGrid(self, colors=(RED, BLUE, YELLOW)):
        self.screen.fill(BLACK)
        for goal in self.goals:
            #Sub Goal
            pos_x = goal[0] - 1
            pos_y = self.window_height-1 - (goal[1] - 1)
            pygame.draw.circle(self.screen, colors[0], ((pos_x+0.5) * self.blockSize, (pos_y+0.5) * self.blockSize), self.blockSize//2)
        for obstacle in self.obstacles:
            #Obstacles
            pos_x = obstacle[0] - 1
            pos_y = self.window_height-1 - (obstacle[1] - 1)
            pygame.draw.circle(self.screen, colors[2], ((pos_x+0.5) * self.blockSize, (pos_y+0.5) * self.blockSize), self.blockSize//2)
        
        #Final Goal
        pos_x = self.final_goal[0] - 1
        pos_y = self.window_height-1 - (self.final_goal[1] - 1)
        pygame.draw.circle(self.screen, colors[1], ((pos_x+0.5) * self.blockSize, (pos_y+0.5) * self.blockSize), self.blockSize//2)
        for y in range(self.WINDOW_HEIGHT_SCALED):
            rect = pygame.Rect(self.WINDOW_WIDTH_SCALED / 2 - 0.5 * self.blockSize, y, self.blockSize, self.blockSize)
            pygame.draw.rect(self.screen, GREY, rect, self.blockSize)
        for x in range(self.WINDOW_WIDTH_SCALED):
            rect = pygame.Rect(x, self.WINDOW_HEIGHT_SCALED / 2 - 0.5 * self.blockSize, self.blockSize, self.blockSize)
            pygame.draw.rect(self.screen, GREY, rect, self.blockSize)

        # Draw Corridors
        rect = pygame.Rect(self.WINDOW_WIDTH_SCALED / 2 - 0.5 * self.blockSize, self.window_width//4 * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, BLACK, rect, self.blockSize)
        rect = pygame.Rect(self.WINDOW_WIDTH_SCALED / 2 - 0.5 * self.blockSize, (3*self.window_width//4) * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, BLACK, rect, self.blockSize)

        rect = pygame.Rect(self.window_height//4 * self.blockSize, self.WINDOW_HEIGHT_SCALED / 2 - 0.5 * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, BLACK, rect, self.blockSize)
        rect = pygame.Rect((3*self.window_height//4) * self.blockSize, self.WINDOW_HEIGHT_SCALED / 2 - 0.5 * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, BLACK, rect, self.blockSize)


        # Agent
        agent_pos_x = self.agent_pos[0] - 1
        agent_pos_y = self.window_height-1 - (self.agent_pos[1] - 1)
        rect = pygame.Rect(agent_pos_x * self.blockSize, agent_pos_y * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, GREEN, rect, self.blockSize)

        # Convert to Image
        img = np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)/255.0, dtype=np.float32), axes=(1, 0, 2))
        return img


    def move_agent(self, prev_agent_pos, new_agent_pos):
        prev_agent_pos_x = prev_agent_pos[0] - 1
        prev_agent_pos_y = self.window_height - 1 -(prev_agent_pos[1] - 1)
        new_agent_pos_x = new_agent_pos[0] - 1
        new_agent_pos_y = self.window_height - 1 - (new_agent_pos[1] - 1)
        # Clear Previous position
        rect = pygame.Rect(prev_agent_pos_x * self.blockSize,
                        prev_agent_pos_y * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, BLACK, rect, self.blockSize)
        
        # Clear New Postition
        rect = pygame.Rect(new_agent_pos_x * self.blockSize,
                        new_agent_pos_y * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, BLACK, rect, self.blockSize)
        # Agent
        rect = pygame.Rect(new_agent_pos_x * self.blockSize, new_agent_pos_y * self.blockSize, self.blockSize, self.blockSize)
        pygame.draw.rect(self.screen, GREEN, rect, self.blockSize)
        img = np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)/255.0, dtype=np.float32), axes=(1, 0, 2))
        return img


if __name__=='__main__':
    from matplotlib import pyplot as plt
    env = PyGame(window_height=9, window_width=9, goals=[(3,7), (7,7)], final_goal=(9,1), obstacles=[], agent_pos=(1,1))
    img = env.drawGrid()
    print(img.shape)
    plt.imshow(img)
    plt.savefig('test0.png')