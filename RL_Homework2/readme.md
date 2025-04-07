# 检测数据
tensorboard --logdir='C:/Users/95718/Desktop/vscode/Program/RL_Homework/RL_Homework2/runs'
亦可在runs文件夹下查看.png

# 代码
ppo_cheetah.py是原始的PPO算法运行文件，后面模块化了这个留着备用

# 运行
python main.py --mode train --max_episodes 1000 --save_interval 100 # 训练代码
python main.py --mode test #测试代码

# 待改进