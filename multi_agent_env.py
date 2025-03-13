#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

'''
This class implements a multi-agent environment for path following.
'''

class MultiAgentPathFollowingEnv(gym.Env):
    def __init__(self, num_agents=20, buffer_size=1000):
        super(MultiAgentPathFollowingEnv, self).__init__()
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        self.action_space = gym.spaces.Box(
            low=np.array([[-0.5, 0.1]] * num_agents),
            high=np.array([[0.5, 1.0]] * num_agents),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_agents, 4),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        self.positions = np.zeros((self.num_agents, 2))
        self.orientations = np.zeros(self.num_agents)
        self.path = self.generate_path()
        self.time = 0
        self.r_speed = 0.0
        self.r_forward = 0.0
        self.r_distance = 0.0
        self.r_angle = 0.0
        self.failed = 0.0
        self.success = 0.0
        self.active_agents = np.ones(self.num_agents, dtype=bool)
        return self.get_state(), {}

    def get_state(self):
        '''
        Get the current state of the environment.
        '''
        distances, angles = self.calculate_path_metrics()
        return np.column_stack((self.positions, distances, angles))

    def step(self, actions):
        '''
        Step the environment forward using the given actions.
        '''
        self.update_positions(actions)
        distances, angles = self.calculate_path_metrics()
        
        speeds = actions[:, 1]  
        rewards = self.calculate_rewards(distances, angles, speeds)
        
        self.active_agents = self.active_agents & (distances <= 5.0)
        
        self.failed = (self.active_agents == 0)
        self.success = (np.linalg.norm(self.positions - self.path[-1], axis=1) < 0.2)

        if self.success:
            rewards += 100.0
        elif self.failed:
            rewards -= 100.0

        next_state = self.get_state()
        for i in range(self.num_agents):
            self.replay_buffer.append((self.get_state(), actions[i], rewards[i], next_state[i]))
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.time += 1
        done = self.time > 100 or self.success or self.failed
        return next_state, rewards, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            # Create the plot
            self.fig, self.ax = plt.subplots()

            # Create agent markers
            self.path_plot, = self.ax.plot([], [], label='Path')

            # Create front indicators (small lines)
            self.agent_plots = [self.ax.plot([], [], 'o', label=f'Agent {i}')[0] for i in range(self.num_agents)]
            self.agent_fronts = [self.ax.plot([], [], '-', color='red')[0] for _ in range(self.num_agents)]
            
            # Set plot limits
            self.ax.set_xlim(-10, 110)
            self.ax.set_ylim(-10, 10)
            self.ax.legend()
            plt.ion()
            plt.show(block=False)

        # Update path
        path_x, path_y = zip(*self.path)
        self.path_plot.set_data(path_x, path_y)

        # Update agent positions and front indicators
        for i, agent_plot in enumerate(self.agent_plots):
            if self.active_agents[i]:
                x, y = self.positions[i]

                x = np.array([x]) if np.isscalar(x) else np.array(x)
                y = np.array([y]) if np.isscalar(y) else np.array(y)
                agent_plot.set_data(x, y)  # Update agent position
                
                # Compute front indicator position (short line in the direction of orientation)
                front_x = x + 0.5 * np.cos(self.orientations[i])
                front_y = y + 0.5 * np.sin(self.orientations[i])
                self.agent_fronts[i].set_data([x, front_x], [y, front_y])  # Draw front line

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_positions(self, actions):
        '''
        Update the positions of the agents based on the given actions.
        '''
        actions = actions.reshape((self.num_agents, 2))
        steerings, speeds = actions[:, 0], actions[:, 1]
        self.orientations += steerings * 0.1
        dx = speeds * np.cos(self.orientations)
        dy = speeds * np.sin(self.orientations)
        self.positions += np.column_stack((dx, dy))

    def calculate_path_metrics(self):
        '''
        This function calculates the distance and angle between each agent and the path.
        
        Obs: In the first version we are considering and returning only the distance and
        angle of the closest point in the path. Maybe it's necessary to consider more 
        points in the path to follow the path better..
        '''
        # Calculate the euclidean distance between each agent and EVERY point in the path
        distances = np.linalg.norm(self.path[:, None, :] - self.positions[None, :, :], axis=2)

        # Find the index of the closest point in the path for each agent
        closest_indices = np.argmin(distances, axis=0)

        # Find the coordinates of the closest points
        closest_points = self.path[closest_indices]
        
        # Calculate the distance for the closest point
        distance_to_path = distances[closest_indices, np.arange(self.num_agents)]

        # Calculate the angle between the closest point and the agent's orientation
        path_directions = np.arctan2(closest_points[:, 1] - self.positions[:, 1],
                                     closest_points[:, 0] - self.positions[:, 0])
        
        # The difference between the agent's orientation and the path direction
        angle_to_path = path_directions - self.orientations

        distance_to_closest_point = distance_to_path
        angle_to_closest_point = angle_to_path 
        
        return distance_to_closest_point, angle_to_closest_point

    def calculate_rewards(self, distances, angles, speeds):
        '''
        Calculate the reward for each agent based on the distance and angle to the path.
        '''

        # Reward based on the distance to the path.
        # k_distance is the coefficient of the distance reward.
        k_distance = 0.5
        self.r_distance = k_distance * (-distances)

        # Reward based on the angle to the path.
        # k_angle is the coefficient of the angle reward.
        k_angle = 0.3
        self.r_angle = k_angle * (-angles)

        # # Bônus por estar muito próximo do caminho
        # reward[distances < 0.1] += 1.0  

        if speeds < 0.05:
            self.r_speed -= 5.0

        '''
        Reward based on the progress made along the path
        By: Path Following Optimization for an Underactuated 
        USV Using Smoothly-Convergent Deep Reinforcement Learning
        '''
        
        # Maximum coefficient of the navigation reward
        k_N = 1.0
        
        # Full path length
        total_path_length = np.linalg.norm(self.path[0] - self.path[-1])  
        
        # Distance traveled from start
        progress_param = np.linalg.norm(self.positions - self.path[0], axis=1)  

        # Normalize progress
        navigational_reward = k_N * (progress_param / total_path_length)  
        
        # Add to total reward
        self.r_forward += navigational_reward

        # print("-----")
        # print("r_distance: ", self.r_distance)
        # print("r_forward: ", self.r_forward)
        # print("r_speed: ", self.r_speed)
        # print("r_angle: ", self.r_angle)
        reward = self.r_distance + self.r_forward + self.r_speed + self.r_angle

        # # Alta recompensa para agentes que chegaram ao fim do caminho
        # final_position = self.path[-1]  # Última posição do caminho
        # reached_goal = np.linalg.norm(self.positions - final_position, axis=1) < 0.2  # Verifica se o agente chegou perto do fim
        # reward[reached_goal] += 100.0  # Alta recompensa para incentivar alcançar o objetivo

        # reward = np.maximum(reward, 0.0)

        return reward

    def generate_path(self):
        return np.array([[i, 0.0] for i in range(100)])

    def sample_from_replay_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) >= batch_size else []
