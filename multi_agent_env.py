#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

'''
This class implements a multi-agent environment for path following.
'''

class MultiAgentPathFollowingEnv(gym.Env):
    def __init__(self, num_agents=20, buffer_size=1000):
        super(MultiAgentPathFollowingEnv, self).__init__()
        self.num_agents = num_agents
        self.num_goal = 1.0
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        self.action_space = gym.spaces.Box(
            low=np.array([[0.1, -0.5]] * num_agents),
            high=np.array([[0.3, 0.5]] * num_agents),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_agents, 4),
            dtype=np.float32
        )
        
        self.path = self.generate_path()
        self.reset()
        self.r_ey = 0.0

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        self.current_goal_index = 0
        self.current_position = np.zeros((self.num_agents, 2))
        self.current_angular_velocity = None
        self.update()
        self.angular_velocity_dt = np.zeros(self.num_agents)

        self.time = 0
        self.failed = False
        self.success = False
        self.linear_velocity = None
        self.desired_angular_velocity = np.zeros(self.num_agents)

        self.active_agents = np.ones(self.num_agents, dtype=bool)
        return self.get_state()

    def get_state(self):
        '''
        Get the current state of the environment.
        '''
        # Here we call the path_metrics, this function use goal metrics but this data can
        # be delayed, so we need to fix it
        self.update()

        if self.linear_velocity is None:
            self.linear_velocity = np.array([0.0])
        if self.current_angular_velocity is None:
            self.current_angular_velocity = np.array([0.0])

        return np.column_stack((self.current_goal_distance, self.linear_velocity, self.current_angular_velocity, self.desired_angular_velocity))

    def step(self, action):
        '''
        Step the environment forward using the given action.
        '''
        self.update_position(action)
        self.update()
        
        step_reward = self.calculate_rewards()
        
        self.failed = self.current_goal_distance > 15.0 or self.time > 500
        self.success = self.current_goal_index == len(self.path) - 1

        if self.failed:
            step_reward = np.array([0.0]).reshape(1, 1)

        next_state = self.get_state()
        for i in range(self.num_agents):
            self.replay_buffer.append((self.get_state(), action[i], step_reward[i], next_state[i]))
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.time += 1
        done = self.success or self.failed
        return next_state, step_reward, done

    def render(self):
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
                x, y = self.current_position[i]
                # Update agent position

                # Garante que x e y sejam arrays NumPy com pelo menos um elemento
                x = np.array([x]) if np.isscalar(x) else np.array(x)
                y = np.array([y]) if np.isscalar(y) else np.array(y)
                agent_plot.set_data(x, y)  

                # Compute front indicator position
                front_x = x + 0.5 * np.cos(self.current_angular_velocity[i])
                front_y = y + 0.5 * np.sin(self.current_angular_velocity[i])
                self.agent_fronts[i].set_data([x, front_x], [y, front_y])  # Red line for orientation

                # Show current goal
                goal_x, goal_y = self.current_goal_position
                goal_xs.append(goal_x)
                goal_ys.append(goal_y)

        # Update goal marker
        self.current_goal_marker.set_data(goal_xs, goal_ys)

        # Update title dynamically with speed, steering, and goal distance
        self.title_text.set_text(f"Goal Dist: {self.current_goal_distance[0][0]:.2f}m")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_position(self, action, dt = 0.01):
        '''
        Update the positions of the agents based on the given action.
        '''
        action = action.reshape((self.num_agents, 2))
        self.linear_velocity, self.angular_velocity = action[:, 0], action[:, 1]
        
        self.current_angular_velocity += self.angular_velocity * dt
        # print("self.current_angular_velocity: ", self.current_angular_velocity)
        # self.linear_velocity = self.linear_velocity * 0.1
        dx = self.linear_velocity * np.cos(self.current_angular_velocity)
        dy = self.linear_velocity * np.sin(self.current_angular_velocity)
        self.current_position += np.column_stack((dx, dy))

    def update(self, goal_step=1):
        '''
        Update the goal and calculate the path metrics for each agent.
        '''
        # Update goal
        if self.current_goal_index == 0:
            self.current_goal_position = self.path[0]
            self.previus_goal_position = self.current_goal_position
            self.current_goal_index += goal_step
            self.future_goal_position = self.path[self.current_goal_index + goal_step]
        elif self.current_goal_distance < 0.2:
            self.current_goal_index += goal_step
            self.current_goal_position = self.path[self.current_goal_index]
            self.previus_goal_position = self.path[self.current_goal_index - goal_step]
            self.future_goal_position = self.path[self.current_goal_index + goal_step]
            self.r_forward = np.array([10.0]).reshape(1, 1)

        # Calculate path metrics
        self.current_goal_distance = np.linalg.norm(self.current_position[None, :, :] - self.current_goal_position, axis=2)

        self.current_goal_direction = np.arctan2(
            self.future_goal_position[1] - self.current_goal_position[1],
            self.future_goal_position[0] - self.current_goal_position[0]
        )

        self.agent_to_goal_direction = np.arctan2(
            self.current_goal_position[1] - self.current_position[:, 1],
            self.current_goal_position[0] - self.current_position[:, 0]
        )

        if self.current_angular_velocity is None:
            self.desired_angular_velocity = np.zeros(self.num_agents)
        else:
            # self.desired_angular_velocity = self.current_goal_direction - self.current_angular_velocity
            self.desired_angular_velocity = self.current_goal_direction - self.agent_to_goal_direction

        self.yaw_angle_error = self.current_goal_direction - self.agent_to_goal_direction
        # print("self.yaw_angle_error: ", self.yaw_angle_error)

        self.lateral_error = np.sin(self.yaw_angle_error) * self.current_goal_distance
        # print("self.lateral_error: ", self.lateral_error)

    def calculate_rewards(self):
        '''
        Calculate the reward for each agent based on the distance and angle to the path.
        '''
        
        if abs(self.lateral_error) <= 0.5:
            k_ey = 1.0
            r_ey = np.exp(-k_ey * abs(self.lateral_error))
        elif abs(self.lateral_error) > 0.5:
            k_ey = 10.0
            # r_ey = np.array([0.0]).reshape(1, 1)
            r_ey = np.exp(-k_ey * abs(self.lateral_error))
            # print("r_ey: ", r_ey)
            # print("Lateral error too high, no reward")

        print("abs(self.yaw_angle_error): ", abs(self.yaw_angle_error))
        if abs(self.yaw_angle_error) < (np.pi / 2)/2:
            k_a = 1.0
            r_a = np.exp(-k_a * abs(self.yaw_angle_error))
        elif abs(self.yaw_angle_error) >= (np.pi / 2)/2:
            k_a = 10.0
            r_a = -k_a * abs(self.yaw_angle_error)
            # print("r_a: ", r_a)
            # print("Yaw angle error too high, no reward")
        # elif abs(self.yaw_angle_error) 

        w_ey = 1.0
        w_a = 1.0
        reward = (w_ey * r_ey) + (w_a * r_a) + self.r_forward

        return reward

    def generate_path(self):
        return np.array([[i, 0.0] for i in range(100)])

    def sample_from_replay_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) >= batch_size else []
