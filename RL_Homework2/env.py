import gymnasium as gym
import torch

# 环境交互模块
# 相当于直接从gym里面调HalfCheetah-v4环境和奖励函数了，不需要自己写；要改的话wrappers改写
class CheetahEnv:
    def __init__(self, render_mode=None):
        """
        初始化HalfCheetah环境
        Args:
            render_mode: 渲染模式，None表示不渲染，'human'表示渲染
        """
        self.env = gym.make('HalfCheetah-v4', render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
    
    def reset(self):
        """
        重置环境
        Returns:
            state: 初始状态
        """
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        """
        执行动作
        Args:
            action: 动作
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            truncated: 是否截断
            info: 额外信息
        """
        return self.env.step(action)
    
    def render(self):
        """
        渲染环境
        """
        self.env.render()
    
    def close(self):
        """
        关闭环境
        """
        self.env.close()
    
    @staticmethod
    def get_device():
        """
        获取设备
        Returns:
            device: 设备
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")