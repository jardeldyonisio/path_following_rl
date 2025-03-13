import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

from multi_agent_env import MultiAgentPathFollowingEnv
from ddpg import DDPGAgent
from utils import *

'''
This code it's the main file to run the environment.
'''

# Creating the environment
# env = NormalizedEnv(gym.make("CarRacing-v2"))
env = MultiAgentPathFollowingEnv(num_agents=1)

agent = DDPGAgent(state_dim=6, action_dim=2, max_action=1.0)
noise = OUNoise(env.action_space)
batch_size = 128
# agent = DDPGAgent(env)

# Número de episódios para rodar
rewards = []
avg_rewards = []
episodes = 10000

for episode in range(episodes):
    # Get the initial state
    state = env.reset()
    episode_reward = 0

    for step in range(500):
        # Get an action from the current state
        action = agent.get_action(state)

        # Add noise to the action
        action = noise.get_action(action, step)
        new_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average_reward {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break
        env.render()
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(reward[-10:]))

agent.save_model("ddpg_model.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()