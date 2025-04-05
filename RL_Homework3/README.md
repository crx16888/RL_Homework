# 强化学习作业3：策略梯度算法实现

## 项目概述
本项目实现了多种策略梯度算法，并在两个不同的环境中进行了测试和比较：
1. Point Maze导航问题（连续状态-动作空间）
2. MuJoCo HalfCheetah机器人控制问题

## 算法实现
项目实现了以下策略梯度算法：
- 原始策略梯度（Vanilla Policy Gradient）
- 自然策略梯度（Natural Policy Gradient, NPG）
- 信任域策略优化（Trust Region Policy Optimization, TRPO）
- 近端策略优化（Proximal Policy Optimization, PPO）

## 环境设置
1. Point Maze环境
   - 状态空间：2D位置和速度 (s ∈ ℝ⁴)
   - 动作空间：2D速度控制 (a ∈ [-0.1, 0.1]²)
   - 目标：导航到目标点(0.5, 0.5)

2. HalfCheetah环境
   - 使用MuJoCo物理引擎
   - 目标：使机器人向前运动

## 代码结构
- `main.py`: 主程序，包含算法实现和训练逻辑
- `point_maze.xml`: Point Maze环境的MuJoCo模型文件
- `utils.py`: 实验记录和可视化工具

## 使用方法
1. 安装依赖：
```bash
pip install torch gym mujoco numpy matplotlib
```

2. 运行训练：
```bash
python main.py
```

3. 查看结果：
训练完成后会生成：
- `training_curves.png`: 训练曲线可视化
- `training_results.npy`: 详细训练数据

## 实验结果
- 不同算法在两个环境中的表现比较
- 训练曲线展示了算法的收敛性和稳定性
- 可以通过修改超参数进行进一步调优

## 参考
- PPO论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- TRPO论文：[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)