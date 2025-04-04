import os
import torch
import numpy as np
from ppo.ppo import PPO
from env.maze_env import MazeEnv
from datetime import datetime, timedelta
from torch.utils.tensorboard.writer import SummaryWriter
import shutil

if __name__ == "__main__":
    # 环境配置
    maze = [
        [2, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3],
    ]

    # 训练参数配置
    config = {
        "has_continuous_action_space": True,
        "action_std_init": 0.2,
        "max_training_timesteps": int(1e5),
        "max_ep_len": 200,
        "n_episodes": 200,
        "update_timestep": 4000,
        "K_epochs": 10,
        "eps_clip": 0.2,
        "gamma": 0.99,
        "lr_actor": 0.0003,
        "lr_critic": 0.0005,
        "log_freq": 800,
        "print_freq": 4000,
        "save_model_freq": 20000,
        "random_seed": 0,
    }

    # 初始化环境
    env = MazeEnv(maze)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 初始化PPO
    ppo_agent = PPO(
        state_dim,
        action_dim,
        config["lr_actor"],
        config["lr_critic"],
        config["gamma"],
        config["K_epochs"],
        config["eps_clip"],
        config["has_continuous_action_space"],
        config["action_std_init"],
    )

    # 创建日志目录
    log_dir = r"C:\Users\95718\Desktop\vscode\Program\RL_Homework\RL_Homework1\PPO_logs\Maze_v1"
    
    # 确保日志目录被清空
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化日志系统
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    log_file = open(os.path.join(log_dir, "training_log.csv"), "w+")
    log_file.write('episode,timestep,episode_reward,episode_length,average_reward_10\n')

    # 训练历史记录
    episode_rewards = []
    episode_lengths = []

    # 训练主循环
    start_time = datetime.now().replace(microsecond=0) + timedelta(hours=8)
    print(f"Training Start Time (CST): {start_time}")
    print("="*80)
    
    time_step = 0
    i_episode = 0

    try:
        while i_episode < config["n_episodes"]: # 不设置训练的时间步限制
            i_episode += 1
            state = env.reset()
            episode_reward = 0
            episode_length = 0

            for _ in range(config["max_ep_len"]):
                time_step += 1
                
                # 选择并执行动作
                action = ppo_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # 存储经验到buffer
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                # 更新统计量
                episode_reward += reward
                episode_length += 1

                # 策略更新
                if time_step % config["update_timestep"] == 0:
                    ppo_agent.update()

                # 保存模型
                if time_step % config["save_model_freq"] == 0:
                    model_path = os.path.join("C:\\Users\\95718\\Desktop\\vscode\\Program\\RL_Homework\\RL_Homework1\\PPO_preTrained\\Maze_v1", f"PPO_model_{time_step}.pth")
                    ppo_agent.save(model_path)
                    print(f"\nModel saved at timestep {time_step}")

                # 更新状态
                state = next_state
                if done:
                    break

            # 记录训练指标
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 计算最近10个episode的平均奖励
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            
            # TensorBoard记录
            writer.add_scalar("Episode/Reward", episode_reward, i_episode)
            writer.add_scalar("Episode/Length", episode_length, i_episode)
            writer.add_scalar("Training/Average_Reward_10", avg_reward_10, i_episode)
            
            # CSV日志记录
            log_file.write(f"{i_episode},{time_step},{episode_reward},{episode_length},{avg_reward_10}\n")
            log_file.flush()

            # 打印训练信息
            if i_episode % (config["print_freq"] // config["max_ep_len"]) == 0:
                print(f"\nEpisode: {i_episode} | Timestep: {time_step}")
                print(f"Current Reward: {episode_reward:.1f} | Length: {episode_length}")
                print(f"Average Reward (last 10): {avg_reward_10:.1f}")
                print("-"*60)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        # 训练结束处理
        end_time = datetime.now().replace(microsecond=0) + timedelta(hours=8)
        training_duration = end_time - start_time
        
        print("\n" + "="*80)
        print(f"Training Completed")
        print(f"Start Time: {start_time} | End Time: {end_time}")
        print(f"Total Duration: {training_duration}")
        print(f"Total Episodes: {i_episode} | Total Timesteps: {time_step}")
        
        if len(episode_rewards) > 0:
            print(f"\nPerformance Summary:")
            print(f"Final Episode Reward: {episode_rewards[-1]:.1f}")
            print(f"Best Episode Reward: {max(episode_rewards):.1f}")
            print(f"Average Reward (last 10): {np.mean(episode_rewards[-10:]):.1f}")
        print("="*80)
        
        # 保存最终模型
        final_model_path = os.path.join("C:\\Users\\95718\\Desktop\\vscode\\Program\\RL_Homework\\RL_Homework1\\PPO_preTrained\\Maze_v1", "PPO_final_model.pth")
        ppo_agent.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # 关闭资源
        log_file.close()
        writer.close()
        env.close()
