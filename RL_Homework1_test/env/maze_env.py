import pygame
import numpy as np
from gym import spaces
import gym

class MazeEnv(gym.Env):
    def __init__(self, render_mode="train", screen_size=400):
        super(MazeEnv, self).__init__()
        
        # 迷宫定义（5x5简化版）
        self.maze = np.array([
            [2, 0, 0, 0, 0],  # 起点 (0,0)
            [0, 0, 1, 0, 0],  # 中间单列障碍
            [0, 0, 1, 0, 0],  # 中间单列障碍
            [0, 0, 1, 0, 0],  # 中间单列障碍
            [0, 0, 0, 0, 3]   # 终点 (4,4)
        ], dtype=np.int8)
        
        # 环境参数
        self.n = 5  # 固定迷宫尺寸
        self.reward_array = np.array([
            [-0.1 if cell==0 else -5 if cell==1 else 0 if cell==2 else 100 
            for cell in row] for row in self.maze
        ])
        self.path_history = []
        
        # 空间定义
        self.action_space = spaces.Box(
            low=-0.2, high=0.2, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-0.5, high=0.5, shape=(2,), dtype=np.float32
        )
        
        # 状态初始化
        self.state = np.array([-0.5, -0.5], dtype=np.float32)  # 起点坐标
        self.goal = np.array([0.5, 0.5], dtype=np.float32)     # 终点坐标
        self.step_count = 0
        
        # 渲染设置
        self.block_size = 80  # 增大格子尺寸便于观察
        self.render_mode = render_mode
        if self.render_mode == "show":
            self._init_render()

    # ----------------- 核心逻辑 -----------------
    def _continuous_to_grid(self, x, y):
        """连续坐标转离散网格坐标"""
        grid_x = int((x + 0.5) * (self.n - 1))
        grid_y = int((y + 0.5) * (self.n - 1))
        return (
            np.clip(grid_x, 0, self.n-1),
            np.clip(grid_y, 0, self.n-1)
        )

    def step(self, action):
        # 动作执行
        dx, dy = np.clip(action, -0.2, 0.2)
        new_x = np.clip(self.state[0] + dx, -0.5, 0.5)
        new_y = np.clip(self.state[1] + dy, -0.5, 0.5)
        
        # 碰撞检测
        grid_x, grid_y = self._continuous_to_grid(new_x, new_y)
        if self.maze[grid_x, grid_y] == 1:
            return self.state, -5.0, False, {}
        
        # 状态更新
        self.state = np.array([new_x, new_y], dtype=np.float32)
        self.step_count += 1
        
        # 奖励计算
        reward = self.reward_array[grid_x, grid_y]
        done = (self.maze[grid_x, grid_y] == 3)
        
        # 额外距离奖励
        if not done:
            dist = np.linalg.norm(self.state - self.goal)
            reward += 10 * (1 - dist)  # 距离越近奖励越高
            
        return self.state, reward, done, {}

    # ----------------- 渲染系统 -----------------
    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.n * self.block_size, self.n * self.block_size)
        )
        pygame.display.set_caption("Simple Maze")
        self.font = pygame.font.Font(None, 36)
        self._render_static()

    def _render_static(self):
        """绘制静态元素"""
        color_map = {
            0: (255,255,255),  # 路径
            1: (0,0,0),        # 障碍
            2: (0,0,255),       # 起点
            3: (0,255,0)       # 终点
        }
        for x in range(self.n):
            for y in range(self.n):
                rect = pygame.Rect(
                    y * self.block_size,
                    x * self.block_size,
                    self.block_size,
                    self.block_size
                )
                pygame.draw.rect(self.screen, color_map[self.maze[x,y]], rect)
        pygame.display.flip()

    def render(self):
        if self.render_mode != "show":
            return
        
        # 绘制动态元素
        self._render_static()
        
        # 绘制轨迹
        if len(self.path_history) > 1:
            for i in range(len(self.path_history)-1):
                # 检查路径是否穿过障碍物
                start_x, start_y = self.path_history[i]
                end_x, end_y = self.path_history[i+1]
                
                # 转换为网格坐标并进行边界检查
                start_grid_x = min(max(int(start_x / self.block_size * self.n), 0), self.n-1)
                start_grid_y = min(max(int(start_y / self.block_size * self.n), 0), self.n-1)
                end_grid_x = min(max(int(end_x / self.block_size * self.n), 0), self.n-1)
                end_grid_y = min(max(int(end_y / self.block_size * self.n), 0), self.n-1)
                
                # 检查坐标是否有效
                if not (0 <= start_grid_x < self.n and 0 <= start_grid_y < self.n and
                        0 <= end_grid_x < self.n and 0 <= end_grid_y < self.n):
                    continue
                
                # 检查路径两端点是否在障碍物上
                if (self.maze[start_grid_y, start_grid_x] == 1 or 
                    self.maze[end_grid_y, end_grid_x] == 1):
                    continue
                
                # 根据路径长度渐变颜色
                progress = i / (len(self.path_history) - 1)
                color = (
                    int(255 * (1 - progress)),  # R
                    int(100 * progress),         # G
                    int(200 * (1 - progress))    # B
                )
                
                # 绘制带有渐变透明度的粗线条
                alpha = int(255 * (0.3 + 0.7 * progress))
                line_width = max(2, int(8 * (1 - progress)))
                
                # 创建临时surface来支持透明度
                line_surface = pygame.Surface((self.n * self.block_size, self.n * self.block_size), pygame.SRCALPHA)
                pygame.draw.line(
                    line_surface,
                    (*color, alpha),
                    self.path_history[i],
                    self.path_history[i+1],
                    line_width
                )
                self.screen.blit(line_surface, (0, 0))
        
        # 转换当前坐标并进行边界检查
        x, y = self.state
        screen_x = min(max(int((y + 0.5) * self.block_size * (self.n-1)), 0), self.n * self.block_size - 1)
        screen_y = min(max(int((x + 0.5) * self.block_size * (self.n-1)), 0), self.n * self.block_size - 1)
        self.path_history.append((screen_x, screen_y))
        
        # 绘制智能体
        pygame.draw.circle(self.screen, (255,0,0), (screen_x, screen_y), 12)
        pygame.display.flip()

    def reset(self):
        self.state = np.array([-0.5, -0.5], dtype=np.float32)
        self.step_count = 0
        self.path_history = []
        if self.render_mode == "show":
            self._render_static()
        return self.state

    def close(self):
        pygame.quit()

# 测试用例
if __name__ == "__main__":
    env = MazeEnv(render_mode="show")
    state = env.reset()
    
    # 手动控制测试
    running = True
    while running:
        action = np.random.uniform(-0.2, 0.2, 2)  # 随机动作
        next_state, reward, done, _ = env.step(action)
        env.render()
        
        if done:
            print("到达终点！奖励:", reward)
            break
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()