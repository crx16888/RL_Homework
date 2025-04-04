import pygame
from ppo.ppo import PPO
from env.maze_env import MazeEnv

# 加载训练模型
if __name__ == "__main__":
    # 环境选择和初始化
    # maze = [
    #     [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    #     [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    #     [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    #     [1, 1, 1, 1, 0, 1, 0, 0, 0, 3],
    # ]
    maze = [
        [2, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 3],
    ]

    # env = gym.make(env_name)
    env = MazeEnv(maze, n=5, mode="single", render_mode="show")
    state = env.reset()

    # 获取环境状态维度和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间的维度

    print("state_dim = ", state_dim)
    print("action_dim = ", action_dim)

    # 初始化PPO
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=True,  # 使用连续动作空间
    )
    ppo_agent.load(r"C:\Users\95718\Desktop\vscode\Program\RL_Homework\RL_Homework1\PPO_preTrained\Maze_v1\PPO_Maze_v1_0_0.pth")

    done = False
    while not done:
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        pygame.time.wait(300)  # 控制每步的显示时间，增加等待时间使动作更容易观察
    pygame.quit()
