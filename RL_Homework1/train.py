import os
import time
import numpy as np
import torch
import shutil
from torch.nn.modules.fold import F
from torch.utils.tensorboard.writer import SummaryWriter
from maze_env import ContinuousMazeEnv
from ppo import PPO
from trpo import TRPO
import matplotlib.pyplot as plt

# 创建日志和模型保存目录
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
model_dir = os.path.join(os.path.dirname(__file__), 'models')

# 清除之前的TensorBoard日志
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print(f"已清除旧的TensorBoard日志: {log_dir}")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 训练参数
num_episodes = 1000  # 训练回合数
max_steps = 200      # 每个回合的最大步数
update_frequency = 1024  # 更新频率（收集多少步数据后更新一次策略）- 减小更新频率以提高样本利用率

# 环境和算法参数
gamma = 0.99        # 折扣回报，智能体对于远期回报的权重
gae_lambda = 0.98   # GAE参数 - 增加以更好地平衡偏差和方差
clip_ratio = 0.2    # PPO裁剪参数
value_coef = 0.5    # 价值损失系数
entropy_coef = 0.05 # 熵奖励系数 - 增加以鼓励更多探索
lr = 5e-4           # 网络参数更新的步长
hidden_dim = 128    # 隐藏层维度 - 增加网络容量

def train(algorithm='ppo', max_episodes=num_episodes):
    # 初始化环境
    # render_mode="human" 表示使用实时可视化模式
    # env = ContinuousMazeEnv(render_mode="human")
    env = ContinuousMazeEnv(render_mode="None")
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 根据选择的算法初始化智能体
    if algorithm.lower() == 'ppo':
        agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            hidden_dim=hidden_dim
        )
    elif algorithm.lower() == 'trpo':
        agent = TRPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            max_kl=0.01,  # TRPO特有参数
            damping=0.1,  # TRPO特有参数
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}，请选择 'ppo' 或 'trpo'")
    
    # 初始化TensorBoard - 为不同算法创建不同的日志目录
    algorithm_log_dir = os.path.join(log_dir, algorithm)
    if os.path.exists(algorithm_log_dir):
        shutil.rmtree(algorithm_log_dir)
    os.makedirs(algorithm_log_dir, exist_ok=True)
    writer = SummaryWriter(algorithm_log_dir)
    
    # 训练统计
    episode_rewards = []
    best_reward = -float('inf')
    
    # 收集的步数计数
    steps_collected = 0
    
    # 开始训练
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            
            # 存储经验
            agent.buffer.add(state, action, reward, done, next_state, log_prob, value)
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            step += 1
            steps_collected += 1
            
            # 达到更新频率时更新策略
            if steps_collected >= update_frequency:
                policy_loss, value_loss, entropy_loss = agent.update()
                
                # 记录损失到TensorBoard
                writer.add_scalar('Loss/policy', policy_loss, episode)
                writer.add_scalar('Loss/value', value_loss, episode)
                writer.add_scalar('Loss/entropy', entropy_loss, episode)
                
                steps_collected = 0
        
        # 记录回合奖励
        episode_rewards.append(episode_reward)
        writer.add_scalar('Reward/episode', episode_reward, episode)
        
        # 打印训练信息
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {step}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = os.path.join(model_dir, f'{algorithm}_maze_best.pth')
            agent.save(model_path)
            print(f"New best model saved with reward: {best_reward:.2f}")
        
        # 定期保存模型
        if (episode + 1) % 100 == 0:
            model_path = os.path.join(model_dir, f'{algorithm}_maze_episode_{episode+1}.pth')
            agent.save(model_path)
            print(f"Model saved at episode {episode+1}")
            
            # 绘制奖励曲线
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'{algorithm.upper()} Training Reward')
            plt.savefig(os.path.join(algorithm_log_dir, 'reward_curve.png'))
            plt.close()
    
    # 关闭环境和TensorBoard
    env.close()
    writer.close()
    
    return episode_rewards

def test(model_path, num_episodes=5, algorithm='ppo'):
    # 初始化环境
    env = ContinuousMazeEnv(render_mode="human")
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 根据选择的算法初始化智能体
    if algorithm.lower() == 'ppo':
        agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            hidden_dim=hidden_dim
        )
    elif algorithm.lower() == 'trpo':
        agent = TRPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            max_kl=0.01,  # TRPO特有参数
            damping=0.1,  # TRPO特有参数
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}，请选择 'ppo' 或 'trpo'")
    
    # 加载模型
    agent.load(model_path)
    print(f"模型已加载: {model_path}")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n开始测试回合 {episode+1}/{num_episodes} ({algorithm.upper()})")
        print(f"初始位置: [{state[0]:.3f}, {state[1]:.3f}]")
        
        while not done and step < max_steps:
            # 选择动作（确定性策略）
            # 使用确定性策略进行测试，避免随机性导致在终点附近徘徊
            action = agent.policy.get_action(state, deterministic=True)
            
            # 输出当前状态和选择的动作
            print(f"步骤 {step+1}: 位置 [{state[0]:.3f}, {state[1]:.3f}], 动作 [{action[0]:.3f}, {action[1]:.3f}]")
            
            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            
            # 强制渲染每一步
            env.render()
            plt.pause(0.01)  # 确保图形更新并处理事件循环
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            step += 1
            
            # 控制渲染速度
            time.sleep(0.05)
            
            # 如果到达目标，显示成功信息
            if done:
                print(f"成功到达目标! 位置: [{state[0]:.3f}, {state[1]:.3f}]")
        
        print(f"测试回合 {episode+1}/{num_episodes} 完成, 奖励: {episode_reward:.2f}, 步数: {step}")
        episode_rewards.append(episode_reward)
        
        # 回合结束后暂停一下，让用户有时间查看最终状态
        time.sleep(1.0)
    
    return episode_rewards

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Maze Training') 
    parser.add_argument('--test', action='store_true', help='Test mode') 
    parser.add_argument('--model', type=str, default=None, help='Model path for testing') 
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'trpo'], 
                        help='RL algorithm to use (ppo or trpo)')
    parser.add_argument('--compare', action='store_true', help='Compare PPO and TRPO algorithms')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较模式 - 调用compare.py中的函数
        from compare import compare_algorithms
        print(f"比较 PPO 和 TRPO 算法性能...")
        compare_algorithms(max_episodes=args.episodes)
    elif args.test: # 测试模式
        if args.model is None:
            # 如果没有指定模型，使用最新的模型
            model_pattern = f"{args.algorithm}_maze_*.pth"
            model_files = [f for f in os.listdir(model_dir) 
                          if f.startswith(f"{args.algorithm}_maze_") and f.endswith('.pth')]
            if not model_files:
                print(f"No {args.algorithm.upper()} model found for testing. Please train first.")
                exit(1)
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            model_path = os.path.join(model_dir, latest_model)
        else:
            model_path = args.model
        
        print(f"Testing {args.algorithm.upper()} model: {model_path}")
        test(model_path, algorithm=args.algorithm)
    else:  # 训练模式（默认）
        print(f"Starting {args.algorithm.upper()} training...")
        rewards = train(algorithm=args.algorithm, max_episodes=args.episodes)
        
        # 绘制最终的奖励曲线
        algorithm_log_dir = os.path.join(log_dir, args.algorithm)
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'{args.algorithm.upper()} Training Reward')
        plt.savefig(os.path.join(algorithm_log_dir, 'final_reward_curve.png'))
        plt.show()