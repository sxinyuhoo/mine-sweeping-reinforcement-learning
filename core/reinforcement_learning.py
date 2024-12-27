import os
import cv2
import signal
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import parse_game_state as pgs
import pyautogui
from gymnasium import spaces
from tianshou.data import Collector, ReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, obs, state=None, info=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0] if len(obs.shape) > 1 else 1
        logits = self.model(obs.view(batch, -1))
        return logits, state

class MinesweeperPolicy(DQNPolicy):
    def __init__(self, model, optim, discount_factor, estimation_step, target_update_freq, action_space):
        super().__init__(
            model=model,
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
            batch: Batch 对象，包含��察值等信息
            state: 环境状态
            **kwargs: 其他参数
        """
        obs = batch.obs
        q_values, hidden = self.model(obs, state=state)
        
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
            
        # 获取有效动作
        valid_actions = []
        for state in obs:
            valid = [i for i in range(len(state)) if state[i] == 0]
            if not valid:
                valid = [i for i in range(len(state)) if state[i] == 88]
            if not valid:
                valid = [i for i in range(len(state)) if state[i] != 99]
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

    def __init__(self, templates, grid_size=9):
        super(MinesweeperEnv, self).__init__()
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.action_space = spaces.Discrete(self.state_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        _screen_width, _screen_height = pyautogui.size()
        self.region = {'left': 0, 'top': 0, 'width': _screen_width, 'height': _screen_height}
        self.templates = templates
        _init_img = pgs.capture_screen(self.region)
        self.grid_structure = pgs.parse_grid_structure(_init_img, self.templates, grid_size=self.grid_size)
        self.grid_pos = pgs.parse_subgrid_pos(self.grid_structure[0], self.grid_structure[1], self.grid_structure[2], self.grid_structure[3])
        self.reset_pos = pgs.parse_reset_pos(_init_img, self.templates['reset_button_template'])
        
        self.reset()

    def reset(self):

        # 重置游戏状态
        self.state = np.zeros(self.state_size, dtype=np.int32)
        
        # 点击重置按钮
        pgs.click_position(*self.reset_pos, num_click=2)

        return self.state, {}  # 返回状态和字典

    def step(self, action):

        # 动作转换为网格坐标
        row = action // self.grid_size
        col = action % self.grid_size

        click_type = 'left' if self.state[action] == 0 else 'right'

        # 点击指定位置
        pgs.click_position(*self.grid_pos[(row, col)], num_click=2, button=click_type)

        # 获取新的屏幕截图并解析网格状态
        img = pgs.capture_screen(self.region)
        prev_state = self.state
        new_grid_state = pgs.parse_grid_state(img, *self.grid_structure)
        # print(new_grid_state, "\n == \n")

        # 更新状态
        self.state = np.array(new_grid_state).reshape(-1)

        # 检查游戏是否结束
        done = self.check_done(self.state)

        # 计算即时奖励
        reward = self.calculate_reward(prev_state=prev_state, cur_state=self.state, done=done) 

        return self.state, reward, done, False, {} # 返回 (observation, reward, terminated, truncated, info)

    def check_done(self, grid_state):
        # 检查是否有雷被点击
        if 99 in grid_state:  # 99表示失败
            return True
        return False
    
    def check_victory(self, state):
        # 检查是否胜利，如果没有点到雷且所有格子都被点击，则胜利
        if 0 not in state and 99 not in state:
            return True
        return False

    def calculate_reward(self, prev_state, cur_state, done):
        if done:
            if self.check_victory(cur_state):
                return 100  # 胜利奖励可以设置更高
            else:
                return -50  # 失败惩罚
        else:
            reward = 0
            # 判断是否有新的格子被揭开
            new_opened = sum([1 for i,j in zip(prev_state, cur_state) if i == 0 and j > 0])
            if new_opened > 0:
                reward += 1.0 * new_opened  # 增加揭开新格子的奖励
            else:
                reward -= 0.5  # 减小重复点击的惩罚

            # 对点击已揭开的格子进行惩罚
            already_opened = sum([1 for i,j in zip(prev_state, cur_state) if i==j and i > 0])
            if already_opened > 0:
                reward -= 0.5

            return reward



class MinesweeperAgent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = Net(state_size, action_size, hidden_size)
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

def train_agent(agent, env, buffer_size=50000, batch_size=128, epoch=100, step_per_epoch=2000, step_per_collect=20, update_per_step=0.2):
    # 设置环境
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
    log_path = os.path.join('log', 'minesweeper')
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

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def load_model(agent, model_path):
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path))
        print("Model loaded from ", model_path)

    else:
        print("Model not found in ", model_path)

def main():
    # signal.signal(signal.SIGINT, signal_handler)

    base_path = os.path.dirname(os.path.abspath(__file__))

    templates = {
        # unexploded cell
        0: cv2.imread(os.path.join(base_path,'../config/screenshot_template/0_template.png'), cv2.IMREAD_GRAYSCALE),

        # numbers of mines
        1: cv2.imread(os.path.join(base_path,'../config/screenshot_template/1_template.png'), cv2.IMREAD_GRAYSCALE),
        2: cv2.imread(os.path.join(base_path,'../config/screenshot_template/2_template.png'), cv2.IMREAD_GRAYSCALE),
        3: cv2.imread(os.path.join(base_path,'../config/screenshot_template/3_template.png'), cv2.IMREAD_GRAYSCALE),
        4: cv2.imread(os.path.join(base_path,'../config/screenshot_template/4_template.png'), cv2.IMREAD_GRAYSCALE),
        5: cv2.imread(os.path.join(base_path,'../config/screenshot_template/5_template.png'), cv2.IMREAD_GRAYSCALE),

        # mine
        99: cv2.imread(os.path.join(base_path,'../config/screenshot_template/mine_template.png'), cv2.IMREAD_GRAYSCALE),
        # flagged mine
        88: cv2.imread(os.path.join(base_path,'../config/screenshot_template/flag_template.png'), cv2.IMREAD_GRAYSCALE),
        # exploded but empty cell
        77: cv2.imread(os.path.join(base_path,'../config/screenshot_template/empty_template.png'), cv2.IMREAD_GRAYSCALE),

        # grid
        'grid_template': cv2.imread(os.path.join(base_path,'../config/screenshot_template/grid_template.png'), cv2.IMREAD_GRAYSCALE),
        # reset button
        'reset_button_template': cv2.imread(os.path.join(base_path,'../config/screenshot_template/reset_button_template.png'), cv2.IMREAD_GRAYSCALE)
    }

    # 初始化环境和智能体
    grid_size = 9
    env = MinesweeperEnv(templates=templates, grid_size=grid_size)
    agent = MinesweeperAgent(state_size=grid_size*grid_size, action_size=grid_size*grid_size)

    # 将 agent 传递给环境
    env.agent = agent

    # 加载模型
    model_path = os.path.join(base_path, '../log/minesweeper/policy.pth')
    load_model(agent, model_path)

    result = train_agent(agent, env, epoch=100)


if __name__ == '__main__':
    main()