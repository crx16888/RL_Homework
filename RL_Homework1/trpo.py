import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence

# 策略网络 - 与PPO中相同
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
        entropy = dist.entropy().sum(dim=-1) # 策略的熵，表示动作选择的不确定性
        
        return log_prob, entropy

# 价值网络 - 与PPO中相同
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

# 经验回放缓冲区 - 与PPO中相同
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

# TRPO算法
class TRPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 max_kl=0.01, damping=0.1, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, update_epochs=10, hidden_dim=64):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        
        # 只为价值网络设置优化器，策略网络使用TRPO更新
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.buffer = RolloutBuffer()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl  # KL散度约束
        self.damping = damping  # 共轭梯度法的阻尼系数
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
    
    def flat_grad(self, grads):
        flat_grads = []
        for grad in grads:
            flat_grads.append(grad.view(-1))
        return torch.cat(flat_grads)
    
    def get_kl(self, states):
        # 计算当前策略与旧策略之间的KL散度
        mu, log_std = self.policy(states)
        mu_old, log_std_old = mu.detach(), log_std.detach()
        
        std = torch.exp(log_std)
        std_old = torch.exp(log_std_old)
        
        dist = Normal(mu, std)
        dist_old = Normal(mu_old, std_old)
        
        kl = kl_divergence(dist_old, dist).sum(1, keepdim=True).mean()
        return kl
    
    def conjugate_gradient(self, states, b, nsteps=10, residual_tol=1e-10):
        # 使用共轭梯度法求解Ax=b，其中A是Fisher信息矩阵
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(nsteps):
            Ap = self.fisher_vector_product(states, p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def fisher_vector_product(self, states, v):
        # 计算Fisher信息矩阵与向量v的乘积
        kl = self.get_kl(states)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = self.flat_grad(grads)
        
        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad_grad_kl = self.flat_grad(grads)
        
        return flat_grad_grad_kl + self.damping * v
    
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
        
        # 记录损失值
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        # 执行多个epoch的更新
        for _ in range(self.update_epochs):
            # 更新价值网络
            values = self.value(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            # 计算策略梯度
            logprobs, entropy = self.policy.evaluate(states, actions)
            ratio = torch.exp(logprobs - old_logprobs)
            surrogate = (ratio * advantages).mean()
            
            # 计算策略梯度
            grads = torch.autograd.grad(surrogate, self.policy.parameters())
            policy_gradient = self.flat_grad(grads)
            
            # 使用共轭梯度法计算搜索方向
            search_dir = self.conjugate_gradient(states, policy_gradient.detach())
            
            # 计算步长
            gHg = (self.fisher_vector_product(states, search_dir) * search_dir).sum(0, keepdim=True)
            step_size = torch.sqrt(2 * self.max_kl / (gHg + 1e-8))
            
            # 获取当前参数
            old_params = torch.cat([param.view(-1) for param in self.policy.parameters()])
            
            # 线搜索找到满足KL约束的最大步长
            for i in range(10):
                # 计算新参数
                new_params = old_params + step_size * search_dir
                
                # 将新参数应用到策略网络
                index = 0
                for param in self.policy.parameters():
                    param_size = param.numel()
                    param.data.copy_(new_params[index:index + param_size].view(param.size()))
                    index += param_size
                
                # 计算KL散度
                kl = self.get_kl(states)
                
                # 如果KL散度超过约束，减小步长
                if kl > 1.5 * self.max_kl:
                    step_size *= 0.5
                else:
                    # 计算新的策略损失
                    new_logprobs, new_entropy = self.policy.evaluate(states, actions)
                    new_ratio = torch.exp(new_logprobs - old_logprobs)
                    new_surrogate = (new_ratio * advantages).mean()
                    
                    # 如果策略改进，接受更新
                    if new_surrogate > surrogate:
                        policy_loss = -new_surrogate
                        entropy_loss = -new_entropy.mean()
                        break
                    else:
                        step_size *= 0.5
            
            # 如果线搜索失败，恢复旧参数
            if i == 9:
                index = 0
                for param in self.policy.parameters():
                    param_size = param.numel()
                    param.data.copy_(old_params[index:index + param_size].view(param.size()))
                    index += param_size
        
        # 清空缓冲区
        self.buffer.clear()
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])