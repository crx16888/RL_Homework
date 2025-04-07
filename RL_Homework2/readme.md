# 环境
mujoco虚拟环境
gym、torch、tensorboard等
mujoco一定是2.3.7支持win

# 检测数据
tensorboard --logdir='C:/Users/95718/Desktop/vscode/Program/RL_Homework/RL_Homework2/runs'
亦可在runs文件夹下查看.png

# 代码
ppo_cheetah.py是原始的PPO算法运行文件，后面模块化了这个留着备用

# 运行
python main.py --mode train --max_episodes 1000 --save_interval 100 # 训练代码
python main.py --mode test # 测试代码，可以指定模型路径测试
python main.py --mode compare # 比较代码：比较PPO和TRPO性能并记录

# 待改进
当前奖励函数策略似乎只和前进相关？后续看一看里面有没有关于智能体摔倒的奖励，训了1000回合以后发现智能体还经常摔倒前进。