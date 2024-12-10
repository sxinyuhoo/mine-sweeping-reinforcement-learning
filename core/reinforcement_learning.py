import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MinesweeperAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)

        # 计算目标Q值
        with torch.no_grad():
            target = reward
            if not done:
                target = reward + 0.95 * torch.max(self.model(next_state))

        # 计算当前Q值
        current = self.model(state)[0][action]

        # 计算损失并更新
        loss = self.criterion(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values[0]).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

# Example usage:
# agent = MinesweeperAgent(state_size=64, action_size=64)
# state = np.zeros((1, 64))
# action = agent.act(state)
# print(action)