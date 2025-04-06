import os
import time
import numpy as np
import torch
import shutil
from torch.nn.modules.fold import F
from torch.utils.tensorboard.writer import SummaryWriter
from maze_env import ContinuousMazeEnv
from ppo import PPO
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
update_frequency = 2048  # 更新频率（收集多少步数据后更新一次策略）

# 环境和算法参数
gamma = 0.99        # 折扣因子
gae_lambda = 0.98   # GAE参数 - 增加以更好地平衡偏差和方差
clip_ratio = 0.2    # PPO裁剪参数
value_coef = 0.5    # 价值损失系数
entropy_coef = 0.05 # 熵奖励系数 - 增加以鼓励更多探索
lr = 5e-4           # 学习率 - 略微增加以加快学习
hidden_dim = 128    # 隐藏层维度 - 增加网络容量

def train():
    # 初始化环境
    # render_mode="human" 表示使用实时可视化模式
    env = ContinuousMazeEnv(render_mode="None")
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化PPO算法
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
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练统计
    episode_rewards = []
    best_reward = -float('inf')
    
    # 收集的步数计数
    steps_collected = 0
    
    # 开始训练
    for episode in range(num_episodes):
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
            model_path = os.path.join(model_dir, f'ppo_maze_best.pth')
            agent.save(model_path)
            print(f"New best model saved with reward: {best_reward:.2f}")
        
        # 定期保存模型
        if (episode + 1) % 100 == 0:
            model_path = os.path.join(model_dir, f'ppo_maze_episode_{episode+1}.pth')
            agent.save(model_path)
            print(f"Model saved at episode {episode+1}")
            
            # 绘制奖励曲线
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Reward')
            plt.savefig(os.path.join(log_dir, 'reward_curve.png'))
            plt.close()
    
    # 关闭环境和TensorBoard
    env.close()
    writer.close()
    
    return episode_rewards

def test(model_path, num_episodes=5):
    # 初始化环境
    env = ContinuousMazeEnv(render_mode="human")
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化PPO算法
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
    
    # 加载模型
    agent.load(model_path)
    print(f"模型已加载: {model_path}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n开始测试回合 {episode+1}/{num_episodes}")
        print(f"初始位置: [{state[0]:.3f}, {state[1]:.3f}]")
        
        while not done and step < max_steps:
            # 选择动作（确定性策略）
            # 疑似现在PPO训练出来的模型均值一直是往右上走，所以效果很差；能走到终点完全是靠右上的趋势+一定的随机性
            # 此前奖励函数主要基于智能体到目标的距离，没有考虑障碍物的影响。这导致智能体倾向于直接朝右上方（目标位置）移动，而不考虑避障
            action = agent.policy.get_action(state, deterministic=True)
            
            # # 选择动作（使用与训练相同的随机策略）
            # action, log_prob, value = agent.select_action(state)
            
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
        
        # 回合结束后暂停一下，让用户有时间查看最终状态
        time.sleep(1.0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Maze Training') # 使用 argparse 库处理命令行参数
    parser.add_argument('--test', action='store_true', help='Test mode') # 使用模型进行测试（最新为默认）
    parser.add_argument('--model', type=str, default=None, help='Model path for testing') # 使用指定模型进行测试
    
    args = parser.parse_args()
    
    if args.test: # 测试模式
        if args.model is None:
            # 如果没有指定模型，使用最新的模型
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if not model_files:
                print("No model found for testing. Please train first.")
                exit(1)
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            model_path = os.path.join(model_dir, latest_model)
        else:
            model_path = args.model
        
        print(f"Testing model: {model_path}")
        test(model_path)
    else:  # 训练模式（默认）
        print("Starting training...")
        rewards = train()
        
        # 绘制最终的奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward')
        plt.savefig(os.path.join(log_dir, 'final_reward_curve.png'))
        plt.show()