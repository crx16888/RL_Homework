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
        self.path_history = []  # 存储智能体的移动轨迹
        
        self.screen_size = screen_size  # 屏幕尺寸
        self.cell_size = screen_size // n  # 每个单元格的像素大小
        # 连续动作空间：二维向量，每个维度范围[-0.1,0.1]
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        # 连续状态空间：二维坐标，每个维度范围[-0.5,0.5]
        self.observation_space = spaces.Box(
            low=-0.5, high=0.5, shape=(2,), dtype=np.float32
        )

        # self.current_position == self.state
        # 初始化状态为左上角，映射到[-0.5,0.5]范围
        self.state = np.array([-0.5, -0.5], dtype=np.float32)
        # 目标位置为右下角，映射到[-0.5,0.5]范围
        self.goal = np.array([0.5, 0.5], dtype=np.float32)

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
        # 重置到连续空间的起点位置（左上角）
        self.state = np.array([-0.5, -0.5], dtype=np.float32)
        self.step_count = 0
        self.path_history = []  # 重置轨迹历史
        if self.render_mode=="show":
            self.render_static()
        return self.state

    def is_within_bounds(self, x, y):
        """检查给定位置是否在[-0.5,0.5]范围内"""
        return -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5
    
    def continuous_to_grid(self, x, y):
        """将连续坐标转换为网格坐标"""
        grid_x = int((x + 0.5) * (self.n - 1))
        grid_y = int((y + 0.5) * (self.n - 1))
        return grid_x, grid_y
    
    def step(self, action):
        x, y = self.state
        
        # 确保动作值在[-0.1,0.1]范围内
        dx, dy = np.clip(action, -0.1, 0.1)
        new_x = x + dx
        new_y = y + dy
        
        # 将新状态限制在[-0.5,0.5]范围内
        new_x = np.clip(new_x, -0.5, 0.5)
        new_y = np.clip(new_y, -0.5, 0.5)
        
        # 将连续坐标映射到离散网格以检查碰撞
        current_grid_x, current_grid_y = self.continuous_to_grid(x, y)
        new_grid_x, new_grid_y = self.continuous_to_grid(new_x, new_y)
        
        # 检查是否越界或撞墙
        if not self.is_within_bounds(new_x, new_y):
            return self.state, -10.0, False, {}
        
        # 计算网格位移
        dx_grid = new_grid_x - current_grid_x
        dy_grid = new_grid_y - current_grid_y
        
        # 检查移动路径上是否有墙
        # 使用更细致的路径检测，采样更多点进行检查
        steps = max(20, max(abs(dx_grid), abs(dy_grid)) * 4)  # 增加采样点数量
        for i in range(1, steps):
            # 使用线性插值计算路径上的点
            check_x_continuous = x + (new_x - x) * i / steps
            check_y_continuous = y + (new_y - y) * i / steps
            # 将连续坐标转换为网格坐标
            check_x, check_y = self.continuous_to_grid(check_x_continuous, check_y_continuous)
            # 确保坐标在有效范围内
            if 0 <= check_x < self.n and 0 <= check_y < self.n:
                if self.maze[check_x, check_y] == 1:
                    return self.state, -5.0, False, {}
        
        # 检查目标格子是否是墙
        if self.maze[new_grid_x, new_grid_y] == 1:
            return self.state, -5.0, False, {}
        
        # 更新状态
        self.state = np.array([new_x, new_y], dtype=np.float32)
        
        # 计算到目标的距离作为奖励的一部分
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        step_reward = -0.1 - dist_to_goal  # 基础移动成本加上距离惩罚
        
        # 检查是否到达终点区域
        goal_grid_x, goal_grid_y = self.continuous_to_grid(0.5, 0.5)  # 终点的网格坐标
        
        if (new_grid_x, new_grid_y) == (goal_grid_x, goal_grid_y) and self.maze[goal_grid_x, goal_grid_y] == 3:
            return self.state, 100.0, True, {}
        
        self.step_count += 1
        return self.state, step_reward, False, {}

    def render(self):
        if self.mode == "single":
            self.render_static()  # 重新渲染静态元素，覆盖旧路径

        # 将连续坐标转换为屏幕坐标
        x, y = self.state
        screen_x = int((y + 0.5) * (self.maze.shape[1] * self.block_size))
        screen_y = int((x + 0.5) * (self.maze.shape[0] * self.block_size))
        
        # 记录轨迹点
        self.path_history.append((screen_x, screen_y))
        
        # 绘制轨迹线和轨迹点
        if len(self.path_history) > 1:
            # 绘制轨迹线，使用渐变色和更细的线条
            if len(self.path_history) > 2:
                for i in range(len(self.path_history)-1):
                    # 计算渐变色，从橙色渐变到红色
                    progress = i / (len(self.path_history)-1)
                    color = (255, int(100 * (1-progress)), 0)
                    pygame.draw.line(self.screen, color, self.path_history[i], self.path_history[i+1], 3)
            # 在每个轨迹点绘制小圆点
            for i, point in enumerate(self.path_history[:-1]):
                progress = i / (len(self.path_history)-1)
                color = (255, int(100 * (1-progress)), 0)
                pygame.draw.circle(self.screen, color, point, 4)  # 轨迹点
                pygame.draw.circle(self.screen, (0, 0, 0), point, 5, 1)  # 轨迹点外圈

        # 绘制当前位置（使用更大更明显的圆点表示）
        pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), 15)
        # 绘制一个外圈使智能体更容易看见
        pygame.draw.circle(self.screen, (0, 0, 0), (screen_x, screen_y), 16, 2)

        # 绘制步数
        text = self.font.render(str(self.step_count), True, (0, 0, 0))
        text_rect = text.get_rect(center=(screen_x, screen_y - 25))
        self.screen.blit(text, text_rect)

        pygame.display.flip()  # 更新整个屏幕

    def close(self):
        pygame.quit()