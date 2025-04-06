import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 均值和标准差输出层
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(hidden_dim, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 使用正交初始化，但增加gain值以增加初始探索
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=1.414)  # 使用sqrt(2)作为gain
            nn.init.constant_(layer.bias, 0.0)
        
        # 均值层使用较小的初始化，避免一开始就有强烈的方向偏好
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        
        # 初始化log_std层，使初始策略的标准差较大，增加初始探索
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0.5)  # 初始标准差约为1.65
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # 限制log_std的范围，防止数值不稳定
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state)
        mean, log_std = self.forward(state)
        
        if deterministic:
            return mean.detach().numpy()
        
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        action = dist.sample()
        return action.detach().numpy()
    
    def evaluate(self, state, action):
        mean, log_std = self(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy

# 价值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# 经验回放缓冲区
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.logprobs = []
        self.values = []
    
    def add(self, state, action, reward, done, next_state, logprob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.logprobs.append(logprob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()
        self.logprobs.clear()
        self.values.clear()
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        # 计算广义优势估计(GAE)
        returns = []
        advantages = []
        
        next_value = last_value
        next_advantage = 0
        
        for i in reversed(range(len(self.rewards))):
            # 计算TD误差
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            
            # 计算GAE
            advantage = delta + gamma * gae_lambda * (1 - self.dones[i]) * next_advantage
            returns.insert(0, advantage + self.values[i])
            advantages.insert(0, advantage)
            
            next_value = self.values[i]
            next_advantage = advantage
        
        return returns, advantages

# PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, target_kl=0.01, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, update_epochs=10, hidden_dim=64):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.buffer = RolloutBuffer()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            mean, log_std = self.policy(state)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.value(state)
            
            return action.numpy(), log_prob.numpy(), value.numpy().squeeze()
    
    def update(self):
        # 将缓冲区数据转换为张量
        states = torch.FloatTensor(np.array(self.buffer.states))
        actions = torch.FloatTensor(np.array(self.buffer.actions))
        old_logprobs = torch.FloatTensor(np.array(self.buffer.logprobs))
        old_values = torch.FloatTensor(np.array(self.buffer.values))
        
        # 计算回报和优势
        with torch.no_grad():
            last_value = self.value(torch.FloatTensor(self.buffer.next_states[-1])).squeeze()
        
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value.numpy(), self.gamma, self.gae_lambda)
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 执行多个epoch的更新
        for _ in range(self.update_epochs):
            # 重新评估动作
            logprobs, entropy = self.policy.evaluate(states, actions)
            values = self.value(states).squeeze()
            
            # 计算比率 r(θ) = π_θ(a|s) / π_θ_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # 计算裁剪的目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(values, returns)
            
            # 计算熵奖励
            entropy_loss = -entropy.mean()
            
            # 总损失
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 执行梯度更新
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # 计算KL散度，如果过大则提前停止更新
            with torch.no_grad():
                mean_new, log_std_new = self.policy(states)
                std_new = torch.exp(log_std_new)
                mean_old, log_std_old = self.policy(states)
                std_old = torch.exp(log_std_old)
                
                dist_new = Normal(mean_new, std_new)
                dist_old = Normal(mean_old, std_old)
                
                kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).sum(dim=-1).mean().item()
                
                if kl > 1.5 * self.target_kl:
                    break
        
        # 清空缓冲区
        self.buffer.clear()
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])