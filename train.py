import sys
import numpy as np
import matplotlib.pyplot as plt

from environment.simple import SimplePathFollowingEnv
from agent.ddpg import DDPGAgent
from utils.utils import OUNoise

'''
This code it's the main file to run the environment.
'''

env = SimplePathFollowingEnv()
agent = DDPGAgent(observation_dim=4, action_dim=2, max_action=1.0)
noise = OUNoise(env.action_space)

batch_size = 128
max_episodes = 10000
max_steps = 1000
seed_steps = 10000

step = 0

rewards = []
avg_rewards = []

for episode in range(max_episodes):
    # Get the initial state
    observation = env.reset()
    # print("observation: ", observation)
    episode_reward = 0
    # noise.reset()

    for _ in range(max_steps):
        # Collect experience
        if step > seed_steps:
            action = agent.get_action(observation)
            action = noise.get_action_from_raw_action(action, max_steps)
        else:
            # Takes random action before training
            print("Random action before training")
            action = env.rand_action()

        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.memory.add(observation, action, reward, next_observation, terminated, truncated, info)

        # if step >= seed_steps:
        #     if step == seed_steps:
        #         num_updates = seed_steps
        #     else:
        #         num_updates += 1
        #     for _ in range(num_updates):
        #         agent.update(batch_size)

        if step >= seed_steps:
            # Update the agent
            agent.update(batch_size)

        step += 1

        observation = next_observation
        episode_reward += reward

        if terminated or truncated:
            sys.stdout.write("episode: {}, reward: {}, average_reward {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break
        env.render()
    rewards.append(episode_reward)

agent.save_model("models/ddpg_model.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()