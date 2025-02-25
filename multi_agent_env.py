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
        self.previous_steering = np.zeros(self.num_agents)

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        self.positions = np.zeros((self.num_agents, 2))
        self.orientations = np.zeros(self.num_agents)

        # If will be created a new path for each episode, this line should be moved
        self.path = self.generate_path()
        self.time = 0
        self.r_speed = 0.0
        self.r_forward = 0.0
        self.r_distance = 0.0
        self.r_angle = 0.0
        self.r_steering = 0.0
        self.failed = False
        self.success = False
        self.r_window = np.zeros(self.num_agents)

        self.goals = np.zeros((self.num_agents, 2))
        self.goal_counter = 0
        self.calculate_goal()

        self.window_size = 5  
        self.active_goal_indices = list(range(self.window_size))  # Start with first 5 path points
        self.active_agents = np.ones(self.num_agents, dtype=bool)
        return self.get_state(), {}

    def get_state(self):
        '''
        Get the current state of the environment.
        '''
        distances, angles, _ = self.calculate_path_metrics()
        return np.column_stack((self.positions, distances, angles))

    def step(self, actions):
        '''
        Step the environment forward using the given actions.
        '''
        self.update_positions(actions)
        self.calculate_goal()
        distances, angles, closest = self.calculate_path_metrics()
        
        speeds = actions[:, 1]  
        rewards = self.calculate_rewards(distances, angles, speeds, actions[:, 0], closest)
        
        # self.active_agents = self.active_agents & (distances <= 10.0)
        
        # Will be pena
        self.failed = distances > 5.0 or self.time > 150
        # self.success = (np.linalg.norm(self.positions - self.path[-1], axis=1) < 0.2)
        self.success = self.goal_counter == len(self.path) * 0.7

        # if self.success:
        #     rewards += 100.0
        if self.failed:
            print("Penalized")
            rewards += -10.0

        next_state = self.get_state()
        for i in range(self.num_agents):
            self.replay_buffer.append((self.get_state(), actions[i], rewards[i], next_state[i]))
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.time += 1
        done = self.success or self.failed
        return next_state, rewards, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            # Create the plot
            self.fig, self.ax = plt.subplots()

            # Create path and agent markers
            self.path_plot, = self.ax.plot([], [], linestyle='--', label='Path')
            self.agent_plots = [self.ax.plot([], [], 'o', label=f'Agent {i}')[0] for i in range(self.num_agents)]
            self.agent_fronts = [self.ax.plot([], [], '-', color='red')[0] for _ in range(self.num_agents)]

            # Current goal marker
            self.current_goal_marker, = self.ax.plot([], [], 'rx', markersize=5, label='Current Goal')

            # Title text placeholder for Speed, Steering, and Goal Distance
            self.title_text = self.ax.set_title("Speed: 0.00 m/s | Steering: 0.00 rad | Goal Dist: 0.00m")

            # Set plot limits
            self.ax.set_xlim(-10, 110)
            self.ax.set_ylim(-5, 5)
            self.ax.legend()
            plt.ion()
            plt.show(block=False)

        # Update path
        path_x, path_y = zip(*self.path)
        self.path_plot.set_data(path_x, path_y)

        # Update agent positions and front indicators
        goal_xs, goal_ys = [], []
        for i, agent_plot in enumerate(self.agent_plots):
            if self.active_agents[i]:
                x, y = self.positions[i]
                # Update agent position
                agent_plot.set_data(x, y)  

                # Compute front indicator position
                front_x = x + 0.5 * np.cos(self.orientations[i])
                front_y = y + 0.5 * np.sin(self.orientations[i])
                self.agent_fronts[i].set_data([x, front_x], [y, front_y])  # Red line for orientation

                # Show current goal
                goal_x, goal_y = self.goals
                goal_xs.append(goal_x)
                goal_ys.append(goal_y)

        # Update goal marker
        self.current_goal_marker.set_data(goal_xs, goal_ys)

        # Update title dynamically with speed, steering, and goal distance
        speed = self.last_speed if hasattr(self, 'last_speed') else 0.0
        steering = self.last_steering if hasattr(self, 'last_steering') else 0.0
        self.title_text.set_text(f"Speed: {speed:.2f} m/s | Steering: {steering:.2f} rad | Goal Dist: {self.goals_distances[0][0]:.2f}m")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_positions(self, actions):
        '''
        Update the positions of the agents based on the given actions.
        '''
        actions = actions.reshape((self.num_agents, 2))
        steerings, speeds = actions[:, 0], actions[:, 1]
        
        # In the original code the steering limitation is defined here
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
        # distances = np.linalg.norm(self.path[:, None, :] - self.positions[None, :, :], axis=2)
        self.goals_distances = np.linalg.norm(self.goals - self.positions[None, :, :], axis=2)

        # Find the index of the closest point in the path for each agent
        self.goals_indices = np.argmin(self.goals_distances, axis=0)

        # Find the coordinates of the closest points
        closest_points = self.path[self.goals_indices]
        
        # Calculate the distance for the closest point
        distance_to_path = self.goals_distances[self.goals_indices, np.arange(self.num_agents)]

        # Calculate the angle between the closest point and the agent's orientation
        path_directions = np.arctan2(closest_points[:, 1] - self.positions[:, 1],
                                     closest_points[:, 0] - self.positions[:, 0])
        
        # The difference between the agent's orientation and the path direction
        angle_to_path = path_directions - self.orientations

        distance_to_closest_point = distance_to_path
        angle_to_closest_point = angle_to_path 
        
        return distance_to_closest_point, angle_to_closest_point, self.goals_indices
    
    def calculate_goal(self):
        '''
        Calculate the goal for each agent based on the distance and angle to the path.
        '''
        if self.goal_counter == 0:
            self.goals = self.path[0]
            self.goal_counter += 1
        elif self.goals_distances < 0.2:
            self.goals = self.path[self.goal_counter]
            self.goal_counter += 1
            self.r_forward += 10.0

    # def calculate_angle(self):
    #     '''
    #     Calculate the angle between the agent's orientation and the path.
    #     '''
    #     path_directions = np.arctan2(self.goals[1] - self.positions[:, 1],
    #                                  self.goals[0] - self.positions[:, 0])
        
    #     angle_to_path = path_directions - self.orientations
    #     return angle_to_path

    def calculate_rewards(self, distances, angles, speeds, steerings, goals_indices):
        '''
        Calculate the reward for each agent based on the distance and angle to the path.
        '''
        
        '''
        I tried to implement a succcess and failed reward, but I don't if it's working well.
        The failed is basically when the agent is too far from the path or the episode time
        is over. The success is when the agent is threshold under of the last path point.
        '''

        '''
        The distance reward is a commum choice for path following tasks. There are some 
        variations of this reward, but the most common is the reward based on the distance
        to the path. How many far from the path the agent is, less penality.
        '''

        k_d = 1.0
        # self.r_distance = 2 * np.exp(-k_d * abs(distances)) - 1
        # self.r_distance = -k_d * abs(distances)
        self.r_distance = -abs(distances)
        print("r_distance: ", self.r_distance)
        # self.r_distance += -distances

        # Reward based on the angle to the path.
        # k_angle is the coefficient of the angle reward.
        k_a = 0.1
        # self.r_angle += k_a * (-angles)
        # print("angles: ", angles)
        self.r_angle = np.exp(-k_a * abs(angles))
        # print("r_angle: ", self.r_angle)

        '''
        Based on Reinforcement Learning-Based High-Speed Path Following Control for Autonomous
        Vehicles

        This reward function has the ide
        '''
        # k_s = 0.01
        # self.r_speed = k_s * speeds

        # steering_change = np.abs(self.previous_steering - steerings)  # How much steering changed

        # if steering_change > 0.2:
        #     self.r_steering -= 20.0

        '''
        The reward below, steering_change < 0.02, is not working well. The agent is
        spinning in circles. Maybe it's a problem try to define a threshold to the 
        steering_change.
        '''
        # if steering_change < 0.02:
        #     self.r_steering += 1.0

        # # Bônus por estar muito próximo do caminho
        # reward[distances < 0.1] += 1.0  

        '''
        Reward based on the progress made along the path
        By: Path Following Optimization for an Underactuated 
        USV Using Smoothly-Convergent Deep Reinforcement Learning

        Notes: The issue of this reward is that the agent is rewarded
        for moving forward even if it is not following the path. A solution
        can be add a thereshold to the distance to the path to receive the 
        reward.
        '''
        
        # # Maximum coefficient of the navigation reward
        # k_N = 10.0
        
        # # # Full path length
        # # total_path_length = np.linalg.norm(self.path[0] - self.path[-1])  
        
        # # # Distance traveled from start
        # # progress_param = np.linalg.norm(self.positions - self.path[0], axis=1)  

        # total_path_length = len(self.path)

        # # # Normalize progress
        # navigational_reward = k_N * (self.goal_counter / total_path_length)  
        
        # # # Add to total reward
        # self.r_forward = navigational_reward
        # print("r_forward: ", self.r_forward)

        # k_N = 0.8  # Maximum navigation reward
        # total_path_length = len(self.path)  # Full path length (total points in path)

        # # Where we still having the issue of the agent moving forward without following the path
        # # but it's a new try to solve this problem. Another issue is the agent trying to reach always
        # # the same closest point in the path.
        # progress_param = goals_indices / total_path_length  
        # navigational_reward = k_N * progress_param

        # self.r_forward += navigational_reward

        # # print("window_rewards: ", window_rewards)
        w_d = 1.0
        w_a = 1.0

        reward = w_d * self.r_distance + w_a * self.r_angle + self.r_forward

        '''
        Some developers use the code below to avoid negative rewards.
        '''
        # if reward[0] < 0.0:
        #     reward[0] = 0.0

        '''
        My actual best result was using a very big reward for the success (be close to the last point
        of the path)
        '''
        # final_position = self.path[-1]  # Última posição do caminho
        # reached_goal = np.linalg.norm(self.positions - final_position, axis=1) < 0.2  # Verifica se o agente chegou perto do fim
        # reward[reached_goal] += 100.0  # Alta recompensa para incentivar alcançar o objetivo

        # reward = np.maximum(reward, 0.0)

        self.previous_steering = steerings

        return reward

    def generate_path(self):
        return np.array([[i, 0.0] for i in range(100)])

    def sample_from_replay_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) >= batch_size else []
