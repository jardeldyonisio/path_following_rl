#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from typing import Optional 

'''
This class implements a single-agent environment for path following.
'''

class SimplePathFollowingEnv(gym.Env):
    def __init__(self):
        '''
        
        '''
        super().__init__()
        self.goal_step = 1
        self.num_goals_window = 15
        self.out_of_bound_threshold = (self.goal_step * self.num_goals_window) + self.goal_step
        self.goal_threshold = 0.2

        min_linear_velocity = 0.05
        max_linear_velocity = 1.0

        min_angular_velocity = -1.0
        max_angular_velocity = 1.0

        self.min_goal_distance = 0.0
        self.max_goal_distance = self.out_of_bound_threshold

        min_yaw_error = -2*np.pi
        max_yaw_error = 2*np.pi


        self.terminated_counter : int = 0
        
        self.action_space = gym.spaces.Box(
            low=np.array([min_linear_velocity, min_angular_velocity]),
            high=np.array([max_linear_velocity, max_angular_velocity]),
            dtype=np.float32
        )

        # State: [distance_to_goal, linear_vel, angular_vel, yaw_error]
        self.observation_space = gym.spaces.Box(
            low=np.array([self.min_goal_distance, min_linear_velocity, min_angular_velocity, min_yaw_error]),
            high=np.array([self.max_goal_distance, max_linear_velocity, max_angular_velocity, max_yaw_error]),
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Reset the environment to the initial state.
        '''
        super().reset(seed=seed)
        self.path = self._create_path()

        self.tick : int = 0
        self.linear_velocity : float = 0.0
        self.yaw_angle_error : float = 0.0
        self.angular_velocity : float = 0.0
        self.current_goal_index : int = 1
        self.current_goal_distance : float = 0.0
        self.current_goal_position : float = self.path[0]
        self.current_position : np.ndarray = np.array([-0.3, 0.0])
        self.current_goals_window_position = self.path[1:self.goal_step * self.num_goals_window: self.goal_step]
        self.is_minor_goal_reached : bool = False
        self.goal_reached_counter : int = 0
        self.minor_goal_reacher_counter : int = 0

        self.arrived : bool = False
        self.timeout : bool = False
        self.out_of_bound : bool = False

        observation = self._get_obs()
        info = self._get_info()

        return observation
        # return observation, info

    def step(self, action, dt : float=0.5):
        '''
        Step the environment with the given action.
        '''
        self.linear_velocity, self.angular_velocity = action
        self.angular_velocity = self.angular_velocity * dt 

        self._is_goal_reached()
        # calculate lateral error
        terminated = self._is_success()

        if self.terminated_counter:
            self.terminated_counter += 1

        truncated = self._is_truncated()

        reward = self._rewards()
        observation = self._get_obs()
        info = self._get_info()
        self.tick += 1

        # return observation, reward, truncated
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        '''
        Render the environment.
        '''

        # Verify if fig is already created and create it if not
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()

            # Create markers
            self.path_marker, = self.ax.plot([], [], linestyle='-', color='blue', label='Path')
            self.agent_marker, = self.ax.plot([], [], marker='>', color='red', label='Agent')
            self.current_goal_marker, = self.ax.plot([], [], marker='x', color='green', label='Current Goal')
            self.current_goals_window_marker, = self.ax.plot([], [], marker='x', color='magenta', label='Goals Window')

            self.distance_title = self.ax.set_title('Distance to Goal: 0.00m')

            self.ax.set_xlim(-10, 110)
            self.ax.set_ylim(-5, 5)
            self.ax.legend()
            plt.ion()
            plt.show(block = False)
        
        # if not done
        path_x, path_y = zip(*self.path)
        self.path_marker.set_data(path_x, path_y)

        # Show current goal
        goal_x, goal_y = self.current_goal_position
        self.current_goal_marker.set_data([goal_x], [goal_y])

        # Show current agent position
        agent_x, agent_y = self._update_agent_position()
        self.agent_marker.set_data([agent_x], [agent_y])

        # print("self.current_goals_window_position: ", self.current_goals_window_position)
        if self.current_goals_window_position.size > 0:
            multi_goals_x, multi_goals_y = zip(*self.current_goals_window_position)
            self.current_goals_window_marker.set_data(multi_goals_x, multi_goals_y)

        self.distance_title.set_text(f'Goal distance: {self._goal_distance():.2f}m')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        '''
        Close the environment.
        '''
        pass
    
    def _get_info(self):
        # How many goals reached
        # How many minor goals reached
        pass

    def _update_agent_position(self):
        '''
        Update the agent position based on the current state.
        '''
        dx = self.linear_velocity * np.cos(self.angular_velocity)
        dy = self.linear_velocity * np.sin(self.angular_velocity)
        self.current_position += np.array([dx, dy])

        return self.current_position

    def _get_obs(self):
        '''
        Get the current state of the environment.
        '''
        return np.array([
            self._goal_distance(),
            self.linear_velocity,
            self.angular_velocity,
            self._get_angle_error(),
        ])
    
    def _rewards(self):
        '''
        Calculate the reward based on the current state.
        '''
        reward_goal_reached = 0.0
        reward_minor_goal_reached = 0.0
        sucess_reward = 0.0

        k_d = 1.0

        # reward_lateral_error

        if self.is_goal_reached:
            reward_goal_reached = 10.0
        if self.is_minor_goal_reached:
            reward_minor_goal_reached = 5.0
        if self._is_success():
            sucess_reward = 100.0
        rewards = reward_goal_reached + sucess_reward + reward_minor_goal_reached

        return rewards
    
    def _goal_distance(self):
        '''
        Calculate the distance to the goal.
        '''
        return np.linalg.norm(self._update_agent_position() - self.current_goal_position)
    
    def _is_goal_reached(self):
        '''
        Check if the goal is reached and update if in this case.
        '''
        self.reward_goal_reached = 0.0
        if self._goal_distance() < self.goal_threshold:
            self.current_goal_index += self.goal_step
            self.goal_reached_counter += 1
            print("self.goal_reached_counter: ", self.goal_reached_counter)
            if self.current_goal_index < len(self.path):
                self.current_goal_position = self.path[self.current_goal_index]
            else:
                self.current_goal_position = self.path[-1]  # Set to the last goal if index exceeds bounds
            self._generate_goals_window()
            self.is_goal_reached = True
            return True
        self._update_minor_goal()
        self.is_goal_reached = False
        return False
    
    def _update_minor_goal(self):
        '''
        Update the minor goal for the agent.
        '''
        # print("self.current_goals_window_position: ", self.current_goals_window_position)
        # Calculate distances between current position and all goals in window
        distances = np.array([self._get_distance(self.current_position, goal) 
                         for goal in self.current_goals_window_position])
        
        if np.any(distances < 0.2):
            self.minor_goal_reacher_counter += 1
            print("self.minor_goal_reacher_counter: ", self.minor_goal_reacher_counter)
            closest_goal_index = np.argmin(distances)
            self.current_goal_index = closest_goal_index + self.current_goal_index + 1
            self.current_goal_position = self.path[self.current_goal_index]
            self._generate_goals_window()
            self.is_minor_goal_reached = True
            return True
        self.is_minor_goal_reached = False
        return False
    
    def _is_success(self):
        '''
        Check if the agent is in the goal.
        '''
        if self._get_distance(self.current_position, self.path[-1]) < self.goal_threshold:
            print("Final goal reached")
            return True
        return False
    
    def _get_distance(self, p1 : np.ndarray, p2: np.ndarray):
        '''
        Calculate the distance between two points.
        '''
        return np.linalg.norm(p1 - p2)
    
    def _is_truncated(self):
        '''
        Verify if the episode is done.
        '''
        self.timeout = self.tick > 1000
        self.out_of_bound = self._goal_distance() > self.out_of_bound_threshold

        if not self._is_success() and self.current_goal_position[1] > self.path[-1][1]:
            self.out_of_bound = True 

        return (self.timeout or self.out_of_bound)
    
    def _create_path(self):
        '''
        Create path for the agent to follow.
        '''
        return np.array([[i, 0.0] for i in range(100)])

        # base_x = np.arange(100)
        # base_y = np.zeros(100)

        # return np.column_stack((base_x, 3 * np.sin(base_x / 3)))

        # x = np.arange(100)
    
        # # Ajuste crítico: deslocamos a lombada para começar em x=0
        # y = 3.0 * np.exp(-0.5 * ((x - 80) / 10)**2)  # Pico em x=30, largura 10
        
        # return np.column_stack((x, y))
    
    def _get_angle_error(self):
        '''
        Calculate the angle error between the agent and the goal.
        '''
        yaw_angle_error = np.arctan2(
            self.current_goal_position[1] - self.current_position[1],
            self.current_goal_position[0] - self.current_position[0]
        )
        return yaw_angle_error
    
    def _generate_goals_window(self):
        '''
        Generate a window of goals for the agent to follow.
        '''
        self.current_goals_window_position = self.path[self.current_goal_index + 1:self.current_goal_index + self.num_goals_window: self.goal_step]