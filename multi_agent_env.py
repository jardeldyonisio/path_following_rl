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
        self.current_goal_index = 0
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
        self.success = self.current_goal_index == len(self.path) * 0.7

        # if self.success:
        #     rewards += 100.0
        # if self.failed:
        #     print("Penalized")
        #     rewards += -10.0

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
                # Garante que x e y sejam arrays NumPy com pelo menos um elemento
                x = np.array([x]) if np.isscalar(x) else np.array(x)
                y = np.array([y]) if np.isscalar(y) else np.array(y)

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
        self.title_text.set_text(f"Speed: {speed:.2f} m/s | Steering: {steering:.2f} rad | Goal Dist: {self.current_goal_distance[0][0]:.2f}m")

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
        This function calculates the distance and angle between the agent and the path.
        
        The first version only considered the distance and angle to the closest point.
        Now, we also compute the **path tangent angle** to help the agent align better.
        '''
    
        current_goal, current_goal_index, previus_goal, previus_goal_index = self.calculate_goal()

        # Compute Euclidean distance between each agent and EVERY goal point
        self.current_goal_distance = np.linalg.norm(current_goal - self.positions[None, :, :], axis=2)

        goal_direction = np.arctan2(
            self.positions[None, :, 1] - previus_goal[:, 1],
            self.positions[None, :, 0] - previus_goal[:, 0]
        )

        # Find the index of the closest goal in the path for each agent
        self.goals_indices = np.argmin(self.current_goal_distance, axis=0)

        # Find the coordinates of the closest goal points
        goals_points = self.path[self.goals_indices]
        
        # Calculate the distance for the closest goal
        distance_to_goal = self.current_goal_distance[self.goals_indices, np.arange(self.num_agents)]

        # Compute path direction at each goal point
        previous_indices = np.maximum(self.goals_indices - 1, 0)  # Ensure valid indices
        previous_points = self.path[previous_indices]

        # Calculate **path tangent angle** (χ_path)
        path_direction = np.arctan2(
            goals_points[:, 1] - previous_points[:, 1], 
            goals_points[:, 0] - previous_points[:, 0]
        )

        # Calculate the angle difference between the agent's orientation and path direction
        angle_to_path = path_direction - self.orientations

        return distance_to_goal, angle_to_path, path_direction, self.goals_indices

    
    def calculate_goal(self):
        '''
        Calculate the goal for each agent based on the distance and angle to the path.
        '''
        # goal_step = 0
        # path_goals = self.path

        if self.current_goal_index == 0:
            # current_goal = self.path[0]
            self.goals = self.path[0]
            previus_goal = np.zeros((self.num_agents, 2))
            self.current_goal_index += 1
        elif self.current_goal_distance < 0.2:
            self.goals = self.path[self.current_goal_index]
            self.current_goal_index += 1
            # current_goal = path_goals[goal_step + self.current_goal_index]
            # self.r_forward += 1.0
        # return current_goal, current_goal_index, previus_goal, previous_goal_index

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
        
        k_d = 0.2
        self.r_distance = np.exp(-k_d * abs(distances))

        # Reward based on the angle to the path.
        # k_angle is the coefficient of the angle reward.
        k_a = 0.1
        self.r_angle = np.exp(-k_a * abs(angles))

        w_d = 0.5
        w_a = 0.4

        reward = w_d * self.r_distance + w_a * self.r_angle + self.r_forward

        return reward

    def generate_path(self):
        return np.array([[i, 0.0] for i in range(100)])

    def sample_from_replay_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) >= batch_size else []
