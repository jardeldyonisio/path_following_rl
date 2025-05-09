import sys
import numpy as np
import matplotlib.pyplot as plt

# from multi_agent_env import MultiAgentPathFollowingEnv
from simple_env import SimplePathFollowingEnv
from agent.ddpg import DDPGAgent
from utils.utils import OUNoise, ReplayBuffer, GaussianStrategy

'''
This code it's the main file to run the environment.
'''

env = SimplePathFollowingEnv()
agent = DDPGAgent(state_dim=4, action_dim=2, max_action=1.0)
noise = OUNoise(env.action_space)
buffer_size = 128
rewards = []
avg_rewards = []
episodes = 10000
step = 1000

memory = ReplayBuffer(buffer_size)

for episode in range(episodes):
    # Get the initial state
    observation = env.reset()
    episode_reward = 0

    for step in range(1000):
        action = agent.get_action(observation)
        # print("action: ", action)
        action = noise.get_action_from_raw_action(action, step)
        # print("action noise: ", action)
        next_observation, reward, terminated, truncated, info = env.step(action)
        memory.add(observation, action, reward, next_observation, terminated, truncated, info)

        state = next_observation
        episode_reward += reward

        if terminated or truncated:
            sys.stdout.write("episode: {}, reward: {}, average_reward {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break
        env.render()
    rewards.append(episode_reward)
    # avg_rewards.append(np.mean(reward[-10:]))

agent.save_model("ddpg_model.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()