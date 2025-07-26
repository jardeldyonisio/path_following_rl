import sys
import numpy as np
import matplotlib.pyplot as plt

from environment.simple import SimplePathFollowingEnv
from agent.ddpg import DDPGAgent
from utils.utils import OUNoise
from torch.utils.tensorboard import SummaryWriter

'''
This code it's the main file to run the environment.
'''

writer = SummaryWriter('runs/ddpg_path_following')

# env = SimplePathFollowingEnv()
# agent = DDPGAgent(observation_dim=4, action_dim=2, max_action=1.0)
# noise = OUNoise(env.action_space)

# batch_size = 128
# max_episodes = 100000
# max_steps = 1000
# seed_steps = 5000

# step = 0

# rewards = []
# avg_rewards = []

# for episode in range(max_episodes):
#     # Get the initial state
#     observation = env.reset()
#     episode_reward = 0
#     episode_steps = 0

#     for _ in range(max_steps):
#         # Collect experience
#         if step > seed_steps:
#             action = agent.get_action(observation)
#             action = noise.get_action_from_raw_action(action, max_steps)

### DrQ-v2 Style Noise Scheduler ###
class DrQv2Noise:
    """
    A noise scheduler that linearly anneals the standard deviation of Gaussian noise.
    This is a common practice in modern DRL algorithms like DrQ-v2.
    """
    def __init__(self, action_dim, initial_noise=1.0, final_noise=0.1, anneal_steps=500000):
        self.action_dim = action_dim
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.anneal_steps = anneal_steps
        # Calculate the linear decay slope
        self.slope = -(self.initial_noise - self.final_noise) / self.anneal_steps

    def get_stddev(self, step):
        """Calculates the current standard deviation based on the training step."""
        if step >= self.anneal_steps:
            return self.final_noise
        return self.initial_noise + self.slope * step

    def sample(self, step):
        """Samples noise from a Gaussian distribution with the current stddev."""
        stddev = self.get_stddev(step)
        return np.random.normal(0, stddev, size=self.action_dim).astype(np.float32)

env = SimplePathFollowingEnv()
agent = DDPGAgent(observation_dim=4, action_dim=2, max_action=1.0)
# noise = OUNoise(env.action_space)

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
    observation = env.reset()
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
            # if step == seed_steps:
            #     num_updates = seed_steps
            # else:
            #     num_updates = 1
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

agent.save_model("models/ddpg_model.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()