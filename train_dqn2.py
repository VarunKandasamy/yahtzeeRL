import gymnasium as gym
from yahtzee_env import YahtzeeEnv
from dqn_agent2 import DQNAgent
import numpy as np
import torch

env = YahtzeeEnv()
agent = DQNAgent(obs_dim=21, action_dim=44, device="cuda" if torch.cuda.is_available() else "cpu")

episodes = 10000
for episode in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    score=0

    while not done:
        action_mask = obs["actionMask"]
        action = agent.act(obs, action_mask)
        next_obs, reward, done, _, info = env.step(action)
        next_mask = next_obs["actionMask"]
        agent.store(obs, action, reward, next_obs, done, next_mask)
        obs = next_obs
        total_reward += reward
        agent.train()

    if episode % 100 == 0:
        agent.update_target()
        
        valid_scores = [int(v) for v in obs["scorecard"] if v != -1]
        scorecard_sum = sum(valid_scores)
        
        print(f"\nEpisode {episode}")
        print(f"Scorecard: {obs['scorecard']}")
        print(f"Scorecard Sum: {scorecard_sum}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}\n")

print("\nEvaluating trained agent over 1000 games")
num_eval_games = 1000
scores = []

agent.epsilon = 0.0
min_score = 1000

for _ in range(num_eval_games):
    obs, _ = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.tensor(agent.preprocess_obs(obs), dtype=torch.float32).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.model(state_tensor)
            mask = torch.tensor(obs["actionMask"], dtype=torch.bool)
            masked_q = q_values[0].masked_fill(~mask, -1e10)
            action = torch.argmax(masked_q).item()

        obs, reward, done, _, _ = env.step(action)

    final_score = sum(int(v) for v in obs["scorecard"])
    min_score = min(min_score, final_score)
    scores.append(final_score)

avg_score = sum(scores) / len(scores)
print(f"\nAverage final score over {num_eval_games} games: {avg_score:.2f}")
print(f"Lowest Score: {min_score}")