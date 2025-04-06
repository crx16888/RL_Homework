import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class ContinuousMazeEnv(gym.Env):
    """
    连续状态空间的迷宫环境
    状态空间: s ∈ R^2, s ∈ [-0.5, 0.5]^2
    动作空间: a ∈ R^2, a ∈ [-0.1, 0.1]^2
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    } #定义了渲染模式（给人类观看、彩色图）和帧率

    def __init__(self, render_mode=None):
        super().__init__()
        
        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        # 目标位置
        self.target_position = np.array([0.4, 0.4])
        
        # 障碍物定义 (x_min, y_min, width, height)
        self.obstacles = [
            [-0.5, 0.0, 0.4, 0.05],   # 水平障碍物1（缩短宽度，给左下角留出通道）
            [0.0, -0.3, 0.05, 0.5],    # 垂直障碍物1（上移起点，给左下角留出通道）
            [0.2, 0.2, 0.2, 0.05],     # 小障碍物1
            [0.1, -0.3, 0.3, 0.05],    # 水平障碍物2
            [0.2, 0.0, 0.05, 0.4],     # 垂直障碍物2
            [-0.3, 0.3, 0.4, 0.05],    # 水平障碍物3
        ]
        
        # 当前位置
        self.current_position = None
        
        # 轨迹记录
        self.trajectory = []
        
        # 渲染相关
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        
        # 重置环境
        self.reset()

    def reset(self, seed=None, options=None):
        # 处理seed参数，但不传递给super().reset()
        if seed is not None:
            np.random.seed(seed) # 当随机数种子不为空时，使用seed作为随机数种子保证实验数据相同
        
        # 固定初始位置在左下角
        self.current_position = np.array([-0.4, -0.4], dtype=np.float32)
        
        # 确保初始位置不在障碍物内（对于固定位置，应该预先确认）
        if self._is_collision(self.current_position):
            raise ValueError("固定的初始位置与障碍物发生碰撞，请调整初始位置或障碍物布局")
        
        # 清空轨迹
        self.trajectory = [self.current_position.copy()]
        
        # 返回观测
        observation = self.current_position.copy()
        info = {}
        
        return observation, info

    # def step(self, action):
    #     # 确保动作在动作空间内
    #     action = np.clip(action, self.action_space.low, self.action_space.high)
        
    #     # 计算新位置
    #     new_position = self.current_position + action
        
    #     # 检查是否超出边界
    #     if np.any(new_position < self.observation_space.low) or np.any(new_position > self.observation_space.high):
    #         # 如果超出边界，位置不变，给予负奖励
    #         reward = -1.0
    #         # 不更新位置
    #     # 检查路径是否与障碍物碰撞
    #     elif self._is_path_collision(self.current_position, new_position):
    #         # 如果碰撞，位置不变，给予更大的负奖励
    #         reward = -1.0
    #         # 不更新位置
    #     else:
    #         # 确保新位置在观测空间内
    #         new_position = np.clip(new_position, self.observation_space.low, self.observation_space.high)
            
    #         # 更新位置
    #         self.current_position = new_position
            
    #         # 计算到目标的距离
    #         distance = np.linalg.norm(self.current_position - self.target_position)
    #         prev_distance = np.linalg.norm(self.current_position - action - self.target_position)
    #         reward = -0.1 * distance
    #         # # 1. 距离奖励：基于距离变化的奖励，鼓励朝目标方向移动
    #         # # 增加距离奖励的权重，使其成为主要的引导信号
    #         # distance_reward = 1.0 * (prev_distance - distance)  # 如果距离减小，给予正奖励
    #         # reward = distance_reward
            
    #         # # 2. 障碍物避让奖励：距离障碍物越近，惩罚越大
    #         # min_obstacle_distance = self._get_min_obstacle_distance(self.current_position)
    #         # obstacle_threshold = 0.1  # 增大安全距离阈值
    #         # if min_obstacle_distance < obstacle_threshold:
    #         #     # 非线性惩罚，距离越近惩罚越大
    #         #     obstacle_penalty = -3.0 * (1.0 - min_obstacle_distance/obstacle_threshold)**2
    #         #     reward += obstacle_penalty
            
    #         # # 3. 方向引导奖励：鼓励智能体朝着目标方向移动，但避开障碍物
    #         # if not self._is_obstacle_between(self.current_position, self.target_position):
    #         #     # 如果当前位置和目标之间没有障碍物，给予额外奖励
    #         #     # 增加此奖励以鼓励找到无障碍路径
    #         #     reward += 0.5
            
    #         # # 4. 目标接近奖励：使用平滑的指数奖励函数
    #         # # 使用更平缓的指数衰减，避免奖励突变
    #         # goal_reward = 2.0 * np.exp(-5.0 * distance)
    #         # reward += goal_reward
            
    #         # 如果非常接近目标，给予额外奖励
    #         if distance < 0.1:
    #             reward += 10.0
            
    #         # 5. 成功到达目标的奖励
    #         # 使用单一的、明确的成功奖励，避免多层次的奖励叠加
    #         if distance < 0.05:
    #             reward += 100.0  # 减小终点奖励，使其与其他奖励更加平衡    
        
    #     # 记录轨迹
    #     self.trajectory.append(self.current_position.copy())
        
    #     # 检查是否到达目标
    #     done = np.linalg.norm(self.current_position - self.target_position) < 0.05
        
    #     # 返回观测、奖励、终止标志和信息
    #     observation = self.current_position.copy()
    #     info = {}
        
    #     # 渲染
    #     if self.render_mode == "human":
    #         self.render()
        
    #     return observation, reward, done, False, info


    def step(self, action):
        # 确保动作在动作空间内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 计算新位置
        new_position = self.current_position + action
        
        # 保存当前位置到目标的距离，用于后续计算
        prev_distance = np.linalg.norm(self.current_position - self.target_position)
        
        # 检查是否超出边界
        if np.any(new_position < self.observation_space.low) or np.any(new_position > self.observation_space.high):
            # 如果超出边界，位置不变，给予较小的负奖励，不直接结束回合
            # 减小边界碰撞惩罚，避免过度惩罚
            reward = -0.5
            done = False
            # 位置不变
            new_position = self.current_position.copy()
        # 检查路径是否与障碍物碰撞
        elif self._is_path_collision(self.current_position, new_position):
            # 如果碰撞，位置不变，给予较小的负奖励，不直接结束回合
            # 减小障碍物碰撞惩罚，避免过度惩罚
            reward = -0.5
            done = False
            # 位置不变
            new_position = self.current_position.copy()
        else:
            # 确保新位置在观测空间内
            new_position = np.clip(new_position, self.observation_space.low, self.observation_space.high)
            
            # 更新位置
            self.current_position = new_position
            
            # 计算到目标的距离
            distance = np.linalg.norm(self.current_position - self.target_position)
            
            # 初始化奖励
            reward = 0.0
            
            # 1. 基于距离的连续奖励
            distance_change = prev_distance - distance
            # 使用平滑的距离奖励函数
            if distance_change > 0:
                # 朝目标移动给予适度的正向奖励
                distance_reward = 2.0 * distance_change
            else:
                # 远离目标给予较小的负向奖励
                distance_reward = 1.0 * distance_change
            reward += distance_reward
            
            # 2. 距离目标的全局奖励
            # 使用平滑的反比例函数
            proximity_reward = 0.5 / (1.0 + 2.0 * distance)
            reward += proximity_reward
            
            # 3. 避障奖励
            min_obstacle_distance = self._get_min_obstacle_distance(self.current_position)
            obstacle_threshold = 0.1
            if min_obstacle_distance < obstacle_threshold:
                # 使用较小的避障惩罚
                obstacle_penalty = -0.1 * (1.0 - min_obstacle_distance/obstacle_threshold)
                reward += obstacle_penalty
            
            # 4. 到达目标奖励
            if distance < 0.05:
                reward += 50.0  # 适度的终点奖励
        
        # 记录轨迹
        self.trajectory.append(self.current_position.copy())
        
        # 检查是否到达目标（如果没有碰到障碍物）
        if 'done' not in locals():
            done = np.linalg.norm(self.current_position - self.target_position) < 0.05
        
        # 返回观测、奖励、终止标志和信息
        observation = self.current_position.copy()
        info = {}
        
        # 渲染
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, done, False, info
    def _is_collision(self, position):
        """
        检查位置是否与障碍物碰撞
        """
        for obstacle in self.obstacles:
            x_min, y_min, width, height = obstacle
            x_max, y_max = x_min + width, y_min + height
            
            if (x_min <= position[0] <= x_max and 
                y_min <= position[1] <= y_max):
                return True
        
        return False
        
    def _is_path_collision(self, start_position, end_position):
        # 真实调用的时候：self._is_path_collision(self.current_position, new_position)，用于判断当前位置和新位置之间是否有障碍物
        """
        检查从起点到终点的路径是否与障碍物碰撞
        使用线段采样检测，将路径分成多个点进行检测
        """
        # 采样点数量，越多检测越精确，但计算量越大
        num_samples = 10
        
        # 生成路径上的采样点
        for i in range(num_samples + 1):
            t = i / num_samples  # 插值参数，从0到1
            sample_position = start_position * (1 - t) + end_position * t
            
            # 检查采样点是否与障碍物碰撞
            if self._is_collision(sample_position):
                return True
                
        return False
        
    def _get_min_obstacle_distance(self, position):
        """
        计算位置到最近障碍物的距离
        """
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            x_min, y_min, width, height = obstacle
            x_max, y_max = x_min + width, y_min + height
            
            # 计算位置到矩形障碍物的最短距离
            dx = max(x_min - position[0], 0, position[0] - x_max)
            dy = max(y_min - position[1], 0, position[1] - y_max)
            distance = np.sqrt(dx**2 + dy**2)
            
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _is_obstacle_between(self, start_position, end_position):
        """
        检查从起点到终点的直线路径上是否有障碍物
        """
        # 使用更多的采样点来提高精度
        num_samples = 20
        
        # 生成路径上的采样点
        for i in range(1, num_samples):  # 跳过起点和终点
            t = i / num_samples
            sample_position = start_position * (1 - t) + end_position * t
            
            # 检查采样点是否与障碍物碰撞
            if self._is_collision(sample_position):
                return True
                
        return False

    def render(self):
        if self.render_mode == "human" and self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.show()
        
        if self.fig is not None:
            self.ax.clear()
            
            # 设置坐标轴范围
            self.ax.set_xlim([-0.6, 0.6])
            self.ax.set_ylim([-0.6, 0.6])
            self.ax.set_aspect('equal')
            self.ax.set_title('Continuous Maze Environment')
            
            # 绘制边界
            boundary = Rectangle((-0.5, -0.5), 1.0, 1.0, linewidth=2, edgecolor='black', facecolor='none')
            self.ax.add_patch(boundary)
            
            # 绘制障碍物
            for obstacle in self.obstacles:
                x_min, y_min, width, height = obstacle
                rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='black', facecolor='gray')
                self.ax.add_patch(rect)
            
            # 绘制目标
            self.ax.plot(self.target_position[0], self.target_position[1], 'ro', markersize=10)
            
            # 绘制当前位置
            self.ax.plot(self.current_position[0], self.current_position[1], 'bo', markersize=8)
            
            # 绘制轨迹
            if len(self.trajectory) > 1:
                trajectory = np.array(self.trajectory)
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=1, alpha=0.5)
            
            plt.draw()
            plt.pause(0.01)
            
            return self.fig
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

# 测试环境
if __name__ == "__main__":
    env = ContinuousMazeEnv(render_mode="human")
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, truncated, info = env.step(action)
        print(f"Position: {obs}, Reward: {reward}")
        
        if done:
            print("Goal reached!")
            break
    
    env.close()