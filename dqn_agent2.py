import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, device="cuda"):
        self.device = device
        self.model = DQN(obs_dim, action_dim).to(self.device)
        self.target = DQN(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

    def preprocess_obs(self, obs):
        # Flatten dice (5), scorecard (13), rerolls (1), yahtzeeZero (1), upperSectionScore (1)
        return np.concatenate([
            obs["dice"].astype(np.float32),
            obs["scorecard"].astype(np.float32),
            [obs["rerolls"]],
            [obs["yahtzeeZero"]],
            [obs["upperSectionScore"]]
        ])

    def act(self, obs, action_mask):
        state = torch.tensor(self.preprocess_obs(obs), dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.model(state).detach().cpu().numpy().squeeze()
        masked_q = np.where(action_mask, q_values, -np.inf)

        if random.random() < self.epsilon:
            valid_actions = np.flatnonzero(action_mask)
            return np.random.choice(valid_actions)
        return int(np.argmax(masked_q))

    def store(self, s, a, r, s_next, done, mask_next):
        self.memory.append((s, a, r, s_next, done, mask_next))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s_batch, a_batch, r_batch, s_next_batch, d_batch, mask_next_batch = zip(*batch)

        np_s_batch = np.array([self.preprocess_obs(s) for s in s_batch])
        s_batch = torch.tensor(np_s_batch, dtype=torch.float32).to(self.device)

#        s_batch = torch.tensor([self.preprocess_obs(s) for s in s_batch], dtype=torch.float32).to(self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.int64).to(self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float32).to(self.device)
        s_next_batch = torch.tensor([self.preprocess_obs(s) for s in s_next_batch], dtype=torch.float32).to(self.device)
        d_batch = torch.tensor(d_batch, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_next = self.target(s_next_batch)
            for i in range(len(q_next)):
                q_next[i][mask_next_batch[i] == 0] = -float("inf")
            max_q_next = q_next.max(1)[0]
            target_q = r_batch + (1 - d_batch) * self.gamma * max_q_next

        q_vals = self.model(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
