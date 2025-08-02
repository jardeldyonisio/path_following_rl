import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from environment.simple import SimplePathFollowingEnv
from agent.ddpg import DDPGAgent
from torch.utils.tensorboard import SummaryWriter
from utils.utils import DrQv2Noise
from datetime import datetime

'''
This code it's the main file to run the environment.
'''

# Set fixed seed
SEED = 42

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"runs/ddpg_path_following/{timestamp}"
writer = SummaryWriter(log_dir=log_dir)

env = SimplePathFollowingEnv()
obs_dim = 4 + env.num_goals_window
agent = DDPGAgent(observation_dim=obs_dim, action_dim=2, max_action=1.0, seed=SEED)
noise = DrQv2Noise(action_dim=env.action_space.shape[0])

batch_size = 256
max_episodes = 100000
max_steps = 1000
seed_steps = 2000

step = 0

rewards = []
avg_rewards = []



for episode in range(max_episodes):
    # Get the initial state
    observation = env.reset(seed=SEED + episode)
    episode_reward = 0
    episode_steps = 0

    for _ in range(max_steps):
        # Collect experience
        if step > seed_steps:
            action = agent.get_action(observation)
            action_noise = noise.sample(step)
            action = action + action_noise
            # 3. Clip the noisy action to be within the valid action range
            action = np.clip(action, -agent.actor.max_action, agent.actor.max_action)

        else:
            # Takes random action before training
            print("Random action before training")
            action = env.rand_action()

        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.memory.add(observation, action, reward, next_observation, terminated, truncated, info)

        if step >= seed_steps:
            critic_loss_val, actor_loss_val = agent.update(batch_size)
            writer.add_scalar('Loss/Critic', critic_loss_val, step)
            writer.add_scalar('Loss/Actor', actor_loss_val, step)

        step += 1
        episode_steps += 1

        observation = next_observation
        episode_reward += reward

        if terminated or truncated:
            break
        env.render()
    rewards.append(episode_reward)
    avg_reward_last_10 = np.mean(rewards[-10:]) if len(rewards) > 0 else episode_reward

    writer.add_scalar('Reward/Episode', episode_reward, episode)
    writer.add_scalar('Reward/Average_reward_last_10_episodes', avg_reward_last_10, episode)
    writer.add_scalar('Reward/Total_reward', episode_steps, episode)

    sys.stdout.write("episode: {}, reward: {}, average_reward {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))

writer.close()

agent.save_model(f"models/{timestamp}.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()