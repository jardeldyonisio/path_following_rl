#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

'''
This class implements a **single-agent** environment for path following.
'''

class PathFollowingEnv(gym.Env):
    def __init__(self, buffer_size=1000):
        super(PathFollowingEnv, self).__init__()
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, 0.1]),  # [steering, speed] limits
            high=np.array([0.5, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),  # [x, y, distance, angle]
            dtype=np.float32
        )
        
        self.reset()
        self.previous_steering = 0.0

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.path = self.generate_path()
        self.time = 0
        self.failed = False
        self.success = False
        self.r_speed = 0.0
        self.r_forward = 0.0
        self.r_distance = 0.0
        self.r_angle = 0.0
        self.r_steering = 0.0
        self.r_window = 0.0

        # ✅ **Define initial target (first point in the path)**
        self.target_index = 0  # First target point

        return self.get_state(), {}

    def get_state(self):
        '''
        Get the current state of the environment.
        '''
        distance, angle = self.calculate_path_metrics()
        return np.array([self.position[0], self.position[1], distance, angle])

    def step(self, action):
        '''
        Step the environment forward using the given action.
        '''
        self.update_position(action)
        distance, angle = self.calculate_path_metrics()
        
        speed = action[1]  
        reward = self.calculate_reward(distance, angle, speed, action[0])
        
        if distance > 10.0:
            self.failed = True
            reward -= 100.0

        next_state = self.get_state()
        self.replay_buffer.append((self.get_state(), action, reward, next_state))
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.time += 1
        done = self.time > 150 or self.failed
        return next_state, reward, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            # Create the plot
            self.fig, self.ax = plt.subplots()

            # Create path and agent markers
            self.path_plot, = self.ax.plot([], [], linestyle='--', label='Path')
            self.agent_plot, = self.ax.plot([], [], 'o', label='Agent')
            self.agent_front, = self.ax.plot([], [], '-', color='red')  # Red line for orientation

            # ✅ Target marker
            self.target_marker, = self.ax.plot([], [], 'bs', markersize=8, label='Target')  # 🔷 Show target

            self.ax.set_xlim(-10, 110)
            self.ax.set_ylim(-10, 10)
            self.ax.legend()
            plt.ion()
            plt.show(block=False)

        # Update path
        path_x, path_y = zip(*self.path)
        self.path_plot.set_data(path_x, path_y)

        # ✅ Update agent position
        x, y = self.position
        self.agent_plot.set_data(x, y)

        # Compute front indicator position
        front_x = x + 0.5 * np.cos(self.orientation)
        front_y = y + 0.5 * np.sin(self.orientation)
        self.agent_front.set_data([x, front_x], [y, front_y])

        # ✅ Show the target
        target_x, target_y = self.path[self.target_index]
        self.target_marker.set_data(target_x, target_y)  # 🔷 Show the current target

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_position(self, action):
        '''
        Update the position of the agent based on the given action.
        '''
        print("action: ", action[0][0])
        steering, speed = action[0][0], action[0][1]
        
        # Steering changes orientation
        self.orientation += steering * 0.1
        dx = speed * np.cos(self.orientation)
        dy = speed * np.sin(self.orientation)
        self.position += np.array([dx, dy])

    def calculate_path_metrics(self):
        '''
        This function calculates:
        - Distance to the assigned target.
        - Angle to the target.
        '''
        target_position = self.path[self.target_index]
        distance_to_target = np.linalg.norm(target_position - self.position)
        target_direction = np.arctan2(target_position[1] - self.position[1],
                                      target_position[0] - self.position[0])
        angle_to_target = target_direction - self.orientation
        return distance_to_target, angle_to_target

    def calculate_reward(self, distance, angle, speed, steering):
        '''
        Calculate the reward based on:
        - Distance to the assigned target
        - Angle to the target
        - Speed consistency
        '''

        k_d = 0.5
        self.r_distance = k_d * (-distance)

        k_a = 0.3
        self.r_angle = k_a * (-angle)

        k_s = 0.01
        self.r_speed = k_s * speed

        # ✅ **Reward for reaching the target**
        if distance < 0.2:  # ✅ If agent is close enough to the target
            self.r_window += 1.0  # Full reward for reaching the goal
            if self.target_index < len(self.path) - 1:  
                self.target_index += 1  # ✅ Move to the next point on the path!

        reward = self.r_distance + self.r_forward + self.r_speed + self.r_angle + self.r_steering + self.r_window

        self.previous_steering = steering

        return reward

    def generate_path(self):
        return np.array([[i, 0.0] for i in range(100)])

    def sample_from_replay_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) >= batch_size else []
