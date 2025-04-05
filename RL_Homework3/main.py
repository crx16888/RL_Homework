import torch
import torch.nn as nn
import gym
import numpy as np
import mujoco
from gym.envs.mujoco.mujoco_env import MujocoEnv
import matplotlib.pyplot as plt
import os

# ================== 自定义PointMaze环境 ==================
class PointMazeEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 500,
    }
    def __init__(self, render_mode="human"):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,))
        super().__init__(
            model_path=r'C:\Users\95718\Desktop\vscode\Program\RL_Homework\RL_Homework3\point_maze.xml',
            frame_skip=1,
            observation_space=observation_space,
            render_mode=render_mode
        )
        self.action_space = gym.spaces.Box(-0.1, 0.1, shape=(2,))
        self._target = np.array([0.5, 0.5])
        self.trajectory = []
        if render_mode == "human":
            self.render()



    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        self.trajectory.append(obs[:2].copy())
        reward = -np.linalg.norm(obs[:2] - self._target)
        done = False
        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:2],  # x, y位置
            self.data.qvel.flat[:2]   # x, y速度
        ])

    def reset(self):
        return self.reset_model()

    def reset_model(self):
        qpos = np.zeros(self.model.nq)
        qvel = np.zeros(self.model.nv)
        qpos[:2] = np.random.uniform(low=-0.1, high=0.1, size=2)
        qvel[:2] = np.random.uniform(low=-0.1, high=0.1, size=2)
        self.set_state(qpos, qvel)
        self.trajectory = []
        return self._get_obs()

# ================== 策略网络 ==================
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, act_dim * 2)
        )
    
    def forward(self, x):
        mu, log_std = self.net(x).chunk(2, -1)
        log_std = torch.clamp(log_std, -20, 2)
        return torch.distributions.Normal(mu, log_std.exp())
    
    def get_kl(self, x, old_dist):
        dist = self(x)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist).sum(-1).mean()
        return kl
    
    def get_log_prob(self, x, actions):
        dist = self(x)
        return dist.log_prob(actions).sum(-1)
    
    def get_flat_params(self):
        return torch.cat([param.data.view(-1) for param in self.parameters()])
    
    def set_flat_params(self, flat_params):
        start = 0
        for param in self.parameters():
            end = start + param.numel()
            param.data.copy_(flat_params[start:end].view_as(param))
            start = end

# ================== 工具函数 ==================
def conjugate_gradient(mvp_function, b, n_iters=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    
    for i in range(n_iters):
        Ap = mvp_function(p)
        alpha = r.dot(r) / p.dot(Ap)
        x += alpha * p
        r_new = r - alpha * Ap
        beta = r_new.dot(r_new) / r.dot(r)
        r = r_new
        if r.norm() < residual_tol:
            break
        p = r + beta * p
    return x

# ================== 训练主函数 ==================
def test(policy, env, num_episodes=100):
    """测试并实时渲染轨迹"""
    import time
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step_idx = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                dist = policy(state_tensor)
                action = dist.mean.numpy()
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 强制每帧渲染
            env.render()
            time.sleep(0.01)  # 控制渲染速度
            state = next_state
            step_idx += 1
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")

def train(logger=None):
    if ENV_NAME == "point_maze":
        env = PointMazeEnv()
    else:
        env = gym.make(ENV_NAME)
    
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    episode_returns = []
    
    for epoch in range(10000):
        states, actions, rewards = [], [], []
        state = env.reset()
        for step_idx in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state)
            dist = policy(state_tensor)
            action = dist.sample().numpy()
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # 训练时每5步渲染一次以平衡性能
            if step_idx % 5 == 0:
                env.render()
            
            state = next_state if not done else env.reset()
            if done: break
        
        # 计算折扣回报
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        old_dist = policy(states_tensor)
        old_log_probs = old_dist.log_prob(actions_tensor).sum(1).detach()
        
        # 计算优势函数
        advantages = returns.detach()
        
        # PPO更新逻辑
        if ENV_NAME == "point_maze":
            for _ in range(4):
                new_dist = policy(states_tensor)
                log_probs = new_dist.log_prob(actions_tensor).sum(1)
                ratio = (log_probs - old_log_probs).exp()
                clip_loss = torch.min(ratio * advantages,
                                    torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantages)
                loss = -clip_loss.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        episode_return = sum(rewards)
        episode_returns.append(episode_return)
        print(f"Epoch {epoch}, Return: {episode_return:.1f}")
        
        if logger is not None:
            logger.log(ALGORITHM, ENV_NAME, epoch, episode_return)
            if (epoch + 1) % 10 == 0:
                model_path = logger.save_model(policy, ALGORITHM, ENV_NAME, epoch + 1)
                print(f"Model saved to {model_path}")

# ================== 全局配置 ==================
ENV_NAME = "point_maze"
ALGORITHM = "PPO"
HIDDEN_SIZE = 64
GAMMA = 0.99
CLIP_EPS = 0.2
MAX_STEPS = 2048
MAX_KL = 0.01
CG_DAMPING = 0.1
CG_ITERS = 10

if __name__ == "__main__":
    from utils import ExperimentLogger
    logger = ExperimentLogger()
    
    print(f"\nTraining {ALGORITHM} on {ENV_NAME}...")
    train(logger)
    
    # 加载模型测试
    if ENV_NAME == 'point_maze':
        env = PointMazeEnv(render_mode="human")
    else:
        env = gym.make(ENV_NAME)
    
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    latest_model = max([f for f in os.listdir(logger.model_dir) if f.startswith(f'{ALGORITHM}_{ENV_NAME}')],
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
    model_path = os.path.join(logger.model_dir, latest_model)
    policy = logger.load_model(policy, model_path)
    
    print(f"\nTesting model from {model_path}...")
    test(policy, env)
    
    logger.close()