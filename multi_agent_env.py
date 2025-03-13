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
            low=np.array([[0.1, -1.0]] * num_agents),
            high=np.array([[1.0, 1.0]] * num_agents),
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
        self.previous_steering = np.zeros(self.num_agents)

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        self.current_position = np.zeros((self.num_agents, 2))
        self.angular_velocity_dt = np.zeros(self.num_agents)

        # If will be created a new path for each episode, this line should be moved
        self.time = 0
        self.r_speed = 0.0
        self.r_forward = 0.0
        self.r_distance = 0.0
        self.r_angle = 0.0
        self.r_steering = 0.0
        self.failed = False
        self.success = False
        self.linear_velocity = None
        self.current_angular_velocity = None
        self.desired_angular_velocity = np.zeros(self.num_agents)

        self.current_goal_index = 0
        self.update_goal()

        self.active_agents = np.ones(self.num_agents, dtype=bool)
        return self.get_state()

    def get_state(self):
        '''
        Get the current state of the environment.
        '''
        # Should return the goal_distance, goal_angle, agent_current_position,
        # agent_current_orientation
        self.path_metrics()

        if self.linear_velocity is None:
            self.linear_velocity = np.array([0.0])
        if self.current_angular_velocity is None:
            self.current_angular_velocity = np.array([0.0])

        return np.column_stack((self.current_position, self.current_goal_distance, self.linear_velocity, self.current_angular_velocity, self.desired_angular_velocity))

    def step(self, action):
        '''
        Step the environment forward using the given action.
        '''

        # Maybe there is a issue where, because the position is updated and only
        # afterthat the goal is updated. Like a delay in the goal update. The all
        # tree functions should be called in the same order.
        self.update_position(action)
        self.update_goal()
        self.path_metrics()
        
        step_reward = self.calculate_rewards()
        
        self.failed = self.current_goal_distance > 5.0 or self.time > 150
        self.success = self.current_goal_index == len(self.path) - 1

        if self.failed:
            step_reward = np.array([0.0])

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
        dx = self.linear_velocity * np.cos(self.current_angular_velocity)
        dy = self.linear_velocity * np.sin(self.current_angular_velocity)
        self.current_position += np.column_stack((dx, dy))

    def path_metrics(self):
        '''
        This function calculates the distance and angle between each agent and the path.
        '''
        
        self.current_goal_distance = np.linalg.norm(self.current_position[None, :, :] - self.current_goal_position, axis=2)

        current_goal_direction = np.arctan2(self.current_position[:, 1] - self.current_goal_position[1],
                                            self.current_position[:, 0] - self.current_goal_position[0])

        if self.current_angular_velocity is None:
            self.current_angular_velocity = np.zeros(self.num_agents)
        else:
            self.desired_angular_velocity = current_goal_direction - self.current_angular_velocity
    
    def update_goal(self, goal_step = 1):
        '''
        Calculate the goal for each agent based on the distance and angle to the path.
        '''
        if self.current_goal_index == 0:
            self.current_goal_position = self.path[0]
            previus_goal_index = self.current_goal_index
            self.current_goal_index += goal_step
        elif self.current_goal_distance < 0.2:
            self.current_goal_index += goal_step
            self.current_goal_position = self.path[self.current_goal_index]
            self.r_forward += 1.0

    def calculate_rewards(self):
        '''
        Calculate the reward for each agent based on the distance and angle to the path.
        '''
        
        k_d = 0.5
        # self.r_distance = -abs(distances)
        # print("self.current_goal_distance: ", self.current_goal_distance)
        self.r_distance = np.exp((-k_d) * abs(self.current_goal_distance[0]))
        # print("r_distance: ", self.r_distance)

        # self.r_angle += k_a * (-angles)
        # print("angles: ", angles)
        # self.r_angle = np.exp(-k_a * abs(angles))
        # print("r_angle: ", self.r_angle)

        w_d = 5.0
        w_a = 1.0

        reward = w_d * self.r_distance + w_a * self.r_angle + self.r_forward

        return reward

    def generate_path(self):
        return np.array([[i, 0.0] for i in range(100)])

    def sample_from_replay_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) >= batch_size else []
