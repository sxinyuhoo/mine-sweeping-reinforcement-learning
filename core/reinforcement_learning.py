import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer

class MinesweeperAgent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.policy = DQNPolicy(model=self.model, optim=self.optimizer, discount_factor=self.gamma, estimation_step=3, target_update_freq=320)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            return q_values.argmax().item()

def train_agent(agent, env, buffer_size=20000, batch_size=64, epoch=10, step_per_epoch=1000, step_per_collect=10, update_per_step=0.1):
    train_envs = DummyVectorEnv([lambda: env for _ in range(8)])
    test_envs = DummyVectorEnv([lambda: env for _ in range(8)])
    buffer = ReplayBuffer(buffer_size)
    collector = Collector(agent.policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(agent.policy, test_envs, exploration_noise=True)

    result = offpolicy_trainer(
        policy=agent.policy,
        train_collector=collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        batch_size=batch_size,
        train_fn=None,
        test_fn=None,
        stop_fn=None,
        save_fn=None,
        logger=None
    )
    return result


# Example usage:
# env = YourMinesweeperEnv()
# agent = MinesweeperAgent(state_size=64, action_size=64)
# result = train_agent(agent, env)
# print(result)