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
        self.wheelbase = 1.0
        
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
        self.path = self.generate_path()
        self.positions = np.zeros((self.num_agents, 2))
        self.orientations = np.zeros(self.num_agents)

        self.time = 0
        self.failed = False
        self.success = False

        self.current_goal_index = 0

        self.active_agents = np.ones(self.num_agents, dtype=bool)
        return self.get_state(), {}

    def get_state(self):
        '''
        Get the current state of the environment.
        '''
        _, _, _, _, _, _, _, goal_distance, angle_goal = self.update_goal_metrics()
        print("Goal distance: ", goal_distance)
        print("Angle goal: ", angle_goal)
        print("Positions: ", self.positions)
        return np.column_stack((self.positions, goal_distance, angle_goal))

    def step(self, actions):
        '''
        Step the environment forward using the given actions.
        '''
        self.update_positions(actions)
        _, _, _, _, _, _, _, current_goal_distance, _ = self.update_goal_metrics()
        
        rewards = self.calculate_rewards()
        
        self.failed = current_goal_distance > 5.0 or self.time > 150
        self.success = self.current_goal_index == len(self.path) - 1

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

    def calculate_wheels_velocities(self, actions):
        '''
        Calculate the velocity of the agents based on the given actions.
        From the DDPG model this function receive the speeds and steering 
        angles and convert them to left and right wheel speeds.
        '''
        steerings, speeds = actions[:, 0], actions[:, 1]

        left_wheel_speed =  speeds - steerings * self.wheelbase / 2
        right_wheel_speed = speeds + steerings * self.wheelbase / 2

        linear_speed = (left_wheel_speed + right_wheel_speed) / 2.0
        angular_speed = (right_wheel_speed - left_wheel_speed) / self.wheelbase

        return linear_speed, angular_speed

    def calculate_heading_angle(self, angular_speed, angle_step = 0.1):
        '''
        Update the orientation of the agents based on the given angular speed.
        '''
        self.orientations += angular_speed * angle_step
        return self.orientations 

    def update_positions(self, actions):
        '''
        Update the positions of the agents based on the given actions for the simulation.
        '''
        actions = actions.reshape((self.num_agents, 2))
        linear_speed, angular_speed = self.calculate_wheels_velocities(actions)
        heading_angle = self.calculate_heading_angle(angular_speed)
        
        # The small change in the robot's x-position after a timestep.
        dx = linear_speed * np.cos(heading_angle)

        # The small change in the robot's y-position after a timestep.
        dy = linear_speed * np.sin(heading_angle)

        self.positions += np.column_stack((dx, dy))

    def calculate_path_metrics(self):
        '''
        This function calculates the distance and angle between the agent and the path.
        '''
        distances_to_all_points = np.linalg.norm(self.path[:, None, :] - self.positions[None, :, :], axis=2)
        closest_point_index = np.argmin(distances_to_all_points, axis=0)
        closest_point_position = self.path[closest_point_index]
        
        if closest_point_index == 0:
            previous_point_position = closest_point_position
            current_point_position = self.path[closest_point_index + 1]
        else:
            previous_point_position = self.path[closest_point_index - 1]
            current_point_position = closest_point_position

        path_direction = np.arctan2(
            previous_point_position[:, 1] - current_point_position[:, 1], 
            previous_point_position[:, 0] - current_point_position[:, 0]
            )

        return path_direction, previous_point_position

    def calculate_error(self):
        '''
        Calculate the progress error and lateral error.
        '''
        path_direction = self.calculate_path_metrics()
        _, _, _, _, previus_goal_direction, previus_goal_distance = self.update_goal_metrics()

        error_x = np.cos(path_direction - previus_goal_direction) * previus_goal_distance
        error_y = np.sin(path_direction - previus_goal_direction) * previus_goal_distance

        return error_x, error_y

    def calculate_distance_between_points(self, p1, p2):
        '''
        Calculate the distance between two points.
        '''
        return np.absolute(np.linalg.norm(p1 - p2, axis=2))

    def update_goal_metrics(self, goal_step = 1):
        '''
        Calculate the goal for each agent based on the distance and angle to the path.
        '''

        # error_x, error_y = self.calculate_error()

        if self.current_goal_index == 0:
            current_goal_position = self.path[0]
            current_goal_index = 0

            previus_goal_position = current_goal_position
            previus_goal_index = current_goal_index
            
            self.current_goal_index += goal_step
        elif np.linalg.norm(previus_goal_position, current_goal_position) - error_x < 0.2:
            self.current_goal_index += goal_step
            current_goal_position = self.path[self.current_goal_index]
            current_goal_index = self.current_goal_index

            previus_goal_position = self.path[self.current_goal_index - goal_step]
            previus_goal_index = self.current_goal_index - goal_step

        previus_goal_direction = np.arctan2(
            self.positions[:, 1] - previus_goal_position[1],
            self.positions[:, 0] - previus_goal_position[0]
        )

        current_goal_direction = np.arctan2(
            self.positions[None, :, 1] - current_goal_position[1],
            self.positions[None, :, 0] - current_goal_position[0]
        )
        
        angle_to_current_goal = current_goal_direction - self.orientations

        distance_between_goals = np.linalg.norm(previus_goal_position - current_goal_position, axis=0)
        previus_goal_distance = np.linalg.norm(self.positions - previus_goal_position, axis=0)
        current_goal_distance = np.linalg.norm(self.positions - current_goal_position, axis=0)

        return current_goal_position, current_goal_index, previus_goal_position, previus_goal_index, previus_goal_direction, previus_goal_distance, distance_between_goals, current_goal_distance, angle_to_current_goal

    def calculate_rewards(self, distances, angles):
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
