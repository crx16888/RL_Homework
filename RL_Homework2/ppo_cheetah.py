import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
from torch.utils.tensorboard.writer import SummaryWriter

# 神经网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习的对数标准差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))  # 输出在[-1,1]之间
        return mu, self.log_std.exp()

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)

# PPO算法实现
class PPO:
    def __init__(self, state_dim, action_dim, device='cpu',
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, ent_coef=0.01,
                 batch_size=64, n_epochs=10):
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            value = self.critic(state)
        return action.cpu().numpy(), log_prob.item(), value.item()

    def compute_gae(self, rewards, masks, values):
        returns = []
        gae = 0
        next_value = 0
        
        for step in reversed(range(len(rewards))):
            if step < len(rewards) - 1:
                next_value = values[step + 1]
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, buffer):
        states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(buffer.log_probs)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(buffer.rewards)).unsqueeze(1).to(self.device)
        masks = torch.FloatTensor(np.array(buffer.masks)).unsqueeze(1).to(self.device)
        values = torch.FloatTensor(np.array(buffer.values)).unsqueeze(1).to(self.device)

        # 计算GAE和returns
        returns = self.compute_gae(rewards.cpu().numpy().flatten(),
                                  masks.cpu().numpy().flatten(),
                                  values.cpu().numpy().flatten())
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = returns - values

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 创建数据集
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 多epoch更新
        for _ in range(self.n_epochs):
            for batch in dataloader:
                state_b, action_b, old_log_prob_b, return_b, advantage_b = batch

                # 计算新策略的概率和熵
                mu, std = self.actor(state_b)
                dist = Normal(mu, std)
                new_log_prob = dist.log_prob(action_b).sum(dim=1, keepdim=True)
                entropy = dist.entropy().mean()

                # 计算策略损失
                ratio = (new_log_prob - old_log_prob_b).exp()
                surr1 = ratio * advantage_b
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantage_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                current_value = self.critic(state_b)
                value_loss = F.mse_loss(current_value, return_b)

                # 总损失
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

                # 更新网络
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

        buffer.clear()

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, mask, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

# 主训练循环
def main():
    env = gym.make('HalfCheetah-v4')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    writer = SummaryWriter()
    ppo = PPO(state_dim, action_dim, device=device)
    buffer = ReplayBuffer()
    
    max_steps = 1_000_000
    update_interval = 2048
    total_steps = 0
    episode = 0
    
    while total_steps < max_steps:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, log_prob, value = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            mask = 0.0 if done else 1.0
            
            buffer.add(state, action, reward, mask, log_prob, value)
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if total_steps % update_interval == 0:
                ppo.update(buffer)
        
        # 记录训练数据
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        print(f"Episode {episode} | Reward: {episode_reward:.1f} | Steps: {total_steps}")
        episode += 1
    
    env.close()
    writer.close()

if __name__ == "__main__":
    main()