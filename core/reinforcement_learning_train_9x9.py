import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from mine_sweeper_game import MineSweeper

# 在文件开头添加设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 残差块1
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        
        # 残差块2 
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self.relu = nn.ReLU()

    def forward(self, obs, state=None, info=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to(device)
        batch = obs.shape[0] if len(obs.shape) > 1 else 1
        x = obs.view(batch, -1)
        
        # 特征提取
        x = self.feature_extractor(x)
        
        # 残差连接1
        identity = x
        x = self.res_block1(x)
        x = self.relu(x + identity)
        
        # 残差连接2
        identity = x  
        x = self.res_block2(x)
        x = self.relu(x + identity)
        
        # 价值预测
        logits = self.value_head(x)
        
        return logits, state

class MinesweeperPolicy(DQNPolicy):
    def __init__(self, model, optim, discount_factor, estimation_step, target_update_freq, action_space):
        super().__init__(
            model=model.to(device),  # 将模型移动到指定设备
            optim=optim,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            is_double=True,
            action_space=action_space
        )
        
    def forward(self, batch, state=None, **kwargs):
        """
        重写 forward 方法来实现自定义的动作选择逻辑
        Args:
            batch: Batch 对象，包含观察值等信息
            state: 环境状态
            **kwargs: 其他参数
        """
        obs = batch.obs
        q_values, hidden = self.model(obs, state=state)
        
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
            
        # 获取有效动作
        valid_actions = []
        for state in obs:
            valid = [i for i in range(len(state)) if state[i] == 7]
            valid_actions.append(valid)
            
        # 修改 Q 值，使无效动作的 Q 值为负无穷
        for i, valid in enumerate(valid_actions):
            invalid_mask = torch.ones_like(q_values[i], dtype=torch.bool)
            invalid_mask[valid] = False
            q_values[i][invalid_mask] = float('-inf')
        
        # 计算最优动作
        acts = q_values.max(dim=1)[1]
        
        return Batch(
            logits=q_values,
            act=acts,
            state=hidden
        )

class MinesweeperEnv(gym.Env):
    def __init__(self, grid_size=9, num_mines=10):
        super(MinesweeperEnv, self).__init__()
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.action_space = spaces.Discrete(self.state_size)
        self.observation_space = spaces.Box(low=0, high=9, shape=(self.state_size,), dtype=np.float32)
        
        # 创建扫雷游戏实例
        self.game = MineSweeper(grid_size=grid_size, num_mines=num_mines)

        self.state = np.array(self.game.cur_grid).flatten()

    def reset(self, seed=None):
        # 重置游戏状态
        self.game = MineSweeper(grid_size=self.grid_size, num_mines=self.game.num_mines)
        self.state = np.array(self.game.cur_grid).flatten()
        return self.state, {}

    def step(self, action):
        # 将动作转换为网格坐标
        row = action // self.grid_size
        col = action % self.grid_size
        
        # 记录之前的状态用于计算奖励
        prev_state = np.array(self.game.cur_grid).flatten()
        
        # 执行动作
        result = self.game.play_game((row, col))
        
        # 获取新状态
        new_state = np.array(self.game.cur_grid).flatten()
        self.state = new_state
        
        # 根据游戏返回值判断游戏状态和奖励
        if result == 1:  # 胜利
            reward = 100
            done = True
        elif result == -1:  # 踩雷
            reward = -50
            done = True
        elif result == -2:  # 重复点击
            reward = -1
            done = False
        else:  # 正常揭开格子
            # 计算新揭开的格子数量
            new_revealed = sum([1 for i, j in zip(prev_state, new_state) if i == 7 and j != 7])
            reward = new_revealed if new_revealed > 0 else -0.1
            done = False
        
        return self.state, reward, done, False, {}
    
    def check_done(self, grid_state):
        """
        检查游戏是否结束
        """
        return any([cell == 9 for row in grid_state for cell in row])
    
    def check_win(self, grid_state):
        """
        检查游戏是否胜利
        """
        return all([cell != 9 or cell != 7 for row in grid_state for cell in row])


    def render(self):
        self.game.show_grid(self.game.cur_grid)

class MinesweeperAgent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = Net(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 使用自定义的 MinesweeperPolicy
        self.policy = MinesweeperPolicy(
            model=self.model,
            optim=self.optimizer,
            discount_factor=self.gamma,
            estimation_step=3,
            target_update_freq=320,
            action_space=spaces.Discrete(action_size)
        )

def train_agent(agent, env, num_envs=5, buffer_size=50000, batch_size=128, epoch=100, step_per_epoch=2000, step_per_collect=20, update_per_step=0.2):
    # 设置环境
    # train_env = DummyVectorEnv([lambda: env for _ in range(num_envs)])
    # test_env = DummyVectorEnv([lambda: env for _ in range(num_envs)])
    # buffer = VectorReplayBuffer(buffer_size, buffer_num=num_envs)
    train_env = DummyVectorEnv([lambda: env])
    test_env = DummyVectorEnv([lambda: env])
    buffer = ReplayBuffer(buffer_size)

    # 设置收集器
    collector = Collector(
        policy=agent.policy,
        env=train_env,
        buffer=buffer,
        exploration_noise=True
    )
    
    test_collector = Collector(
        policy=agent.policy,
        env=test_env,
        exploration_noise=True
    )

    # 创建日志目录和writer
    log_path = os.path.join('log', 'minesweeper_9x9')
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold if env.spec and env.spec.reward_threshold else False

    def train_fn(epoch, env_step):
        # 记录训练信息
        writer.add_scalar('train/env_step', env_step, global_step=env_step)
        if hasattr(collector.buffer, "rew"):
            writer.add_scalar('train/reward', collector.buffer.rew.mean(), global_step=env_step)
        print(f"Training epoch {epoch}, step {env_step}")

    def test_fn(epoch, env_step):
        # 记录测试信息
        writer.add_scalar('test/env_step', env_step, global_step=env_step)
        if hasattr(test_collector.buffer, "rew"):
            writer.add_scalar('test/reward', test_collector.buffer.rew.mean(), global_step=env_step)
        print(f"Testing epoch {epoch}, step {env_step}")

    # 创建 logger
    logger = TensorboardLogger(
        writer,
        train_interval=1,
        update_interval=1,
        save_interval=1
    )

    # 创建训练器
    trainer = OffpolicyTrainer(
        policy=agent.policy,
        train_collector=collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=10,
        batch_size=batch_size,
        update_per_step=update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )

    # 运行训练
    result = trainer.run()
    
    # 关闭writer
    writer.close()
    return result

def load_model(agent, model_path):
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path} to {device}")
    else:
        print("Model not found in ", model_path)

def main():
    # 初始化环境和智能体
    grid_size = 9
    num_mines = 10
    env = MinesweeperEnv(grid_size=grid_size, num_mines=num_mines)
    agent = MinesweeperAgent(state_size=grid_size*grid_size, action_size=grid_size*grid_size)

    # 加载模型
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, '../log/minesweeper_9x9/policy.pth')
    load_model(agent, model_path)

    # 训练智能体
    result = train_agent(agent, env, epoch=1000)

if __name__ == '__main__':
    main()