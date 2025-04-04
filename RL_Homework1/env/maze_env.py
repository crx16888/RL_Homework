import pygame
import numpy as np
from gym import spaces
import gym
from ppo.ppo import PPO

directions = ["上", "下", "左", "右"]

class MazeEnv(gym.Env):
    def __init__(self, maze, n=10, mode="single",render_mode="train", screen_size=400):
        super(MazeEnv, self).__init__()
        self.n = n  # 迷宫的大小 (n x n)
        self.maze = np.array(maze)
        self.reward_array = np.zeros_like(self.maze)  # 创建一个与maze大小相同的数组来存储动态奖励值
        # 路的初始奖励为0
        self.reward_array[self.maze == 2] = -1  # 起点奖励为-1
        self.reward_array[self.maze == 3] = -20 * n  # 终点奖励为-20*n
        self.reward_array[self.maze == 1] = -5  # 墙奖励为-5
        
        self.screen_size = screen_size  # 屏幕尺寸
        self.cell_size = screen_size // n  # 每个单元格的像素大小
        self.action_space = spaces.Discrete(
            4
        )  # 动作空间：上（0），下（1），左（2），右（3）
        self.observation_space = spaces.Box(
            low=0, high=n - 1, shape=(2,), dtype=np.int32
        )  # 观测空间为位置坐标

        # self.current_position == self.state
        self.state = np.array([0, 0])  # 初始化状态为左上角
        self.goal = np.array([n - 1, n - 1])  # 目标位置为右下角

        self.step_count = 0
        self.block_size = 40  # 设置每个格子的大小
        self.mode = mode  # 'single' or 'path'
        self.visited = set()  # 用于存储访问过的位置
        self.effective_movement = False  # 用于存储本次动作是否能产生有效的移动
        self.render_mode=render_mode
        if self.render_mode=="show":
            self.init_screen()

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.maze.shape[1] * self.block_size, self.maze.shape[0] * self.block_size)
        )
        pygame.display.set_caption("Maze Visualization")
        self.font = pygame.font.Font(None, 36)  # 字体初始化
        self.render_static()  # 首次渲染静态元素

    def render_static(self):
        """渲染不会变的部分，如墙和路径"""
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                rect = pygame.Rect(
                    y * self.block_size,
                    x * self.block_size,
                    self.block_size,
                    self.block_size,
                )
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # 墙体是黑色
                elif self.maze[x, y] == 2:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # 起点是蓝色
                elif self.maze[x, y] == 3:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # 终点是绿色
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)  # 路是白色

    def reset(self):
        self.state = np.array([0, 0])  # 每次开始重置到左上角
        self.step_count = 0
        self.visited = set([tuple(self.state)])  # 重置访问位置记录
        self.reward_array = np.zeros_like(self.maze)  # 重置奖励数组
        self.reward_array[self.maze == 2] = -1  # 起点奖励为-1
        self.reward_array[self.maze == 3] = -20 * self.n  # 终点奖励为-20*n
        self.reward_array[self.maze == 1] = -5  # 墙奖励为-5
        if self.render_mode=="show":
            self.render_static()  # 重新渲染静态元素
        return self.state

    def is_within_bounds(self, x, y):
        """检查给定位置是否在迷宫范围内"""
        return 0 <= x < self.n and 0 <= y < self.n
    
    def step(self, action):
        x, y = self.state
        # 根据动作更新状态
        if action == 0:  # 上
            x -= 1
            # self.effective_movement = True
        elif action == 1:  # 下
            x += 1
        elif action == 2:  # 左
            y -= 1
        elif action == 3:  # 右
            y += 1
        else:
            self.effective_movement = False
            print("错误的动作!")
            return self.state, -10000, False, {}  # 无效动作返回负奖励

        # 检查是否越界
        if not self.is_within_bounds(x, y):
            self.effective_movement = False
            return self.state, -1000000, False, {}  # 越界返回负奖励
        else:
            self.effective_movement = True

        if self.maze[x, y] == 1:
            # 撞墙,保持在原位置
            self.reward_array[x, y] -= 100  # 更新撞墙位置的惩罚值
            return self.state, self.reward_array[x, y], False, {}
        elif (x, y) == tuple(self.goal) and self.maze[x, y] == 3:
            # 到达终点
            return (x, y), 100000000000, True, {}  # 到达终点的奖励保持不变
        else:
            # 前面是路,正常前进
            self.state = np.array([x, y])
            self.visited.add(tuple(self.state))
            self.step_count += 1
            self.reward_array[x, y] -= 1  # 更新访问位置的惩罚值
            return self.state, self.reward_array[x, y], False, {}

    def render(self):
        if self.mode == "single":
            self.render_static()  # 重新渲染静态元素，覆盖旧路径
        # 绘制访问过的路径（如果在路径模式下）
        if self.mode == "path":
            for pos in self.visited:
                rect = pygame.Rect(
                    pos[1] * self.block_size,
                    pos[0] * self.block_size,
                    self.block_size,
                    self.block_size,
                )
                pygame.draw.rect(self.screen, (128, 128, 128), rect)

        # 绘制当前位置和步数
        x, y = self.state
        rect = pygame.Rect(
            y * self.block_size, x * self.block_size, self.block_size, self.block_size
        )
        pygame.draw.rect(self.screen, (128, 128, 128), rect)  # 当前位置用灰色标记
        text = self.font.render(str(self.step_count), True, (0, 0, 0))
        # self.font.render 方法用于创建包含文本的图像
        text_rect = text.get_rect(
            center=(
                y * self.block_size + self.block_size // 2,
                x * self.block_size + self.block_size // 2,
            )
        )
        # text.get_rect() 方法获取文本图像的矩形区域，用于定位。
        # center= 设置文本图像的中心点。计算方法是当前块的中心位置，确保文字居中于矩形。
        self.screen.blit(text, text_rect)

        pygame.display.flip()  # 更新整个屏幕

    def close(self):
        pygame.quit()