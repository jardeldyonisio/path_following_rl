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
        self.max_tick: int = 800
        self.goal_step: int = 1
        self.num_goals_window: int = 15
        self.out_of_bound_threshold: int = (self.goal_step * self.num_goals_window) + self.goal_step
        self.goal_threshold: float = 0.2

        self.min_linear_velocity: float = 0.01
        self.max_linear_velocity: float = 0.25

        self.min_angular_velocity: float = -0.5
        self.max_angular_velocity: float = 0.5

        self.min_goal_distance: float = 0.0
        self.max_goal_distance: float = self.out_of_bound_threshold

        self.min_yaw_error: float = -np.pi * 2
        self.max_yaw_error: float = np.pi * 2

        self.terminated_counter: int = 1
        
        self.action_space = gym.spaces.Box(
            low=np.array([self.min_linear_velocity, self.min_angular_velocity]),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity]),
            dtype=np.float32
        )

        # State: [distance_to_goal, linear_vel, angular_vel, yaw_error]
        self.observation_space = gym.spaces.Box(
            low=np.array([self.min_goal_distance, self.min_linear_velocity, self.min_angular_velocity, self.min_yaw_error]),
            high=np.array([self.max_goal_distance, self.max_linear_velocity, self.max_angular_velocity, self.max_yaw_error]),
            dtype=np.float32
        )

        self.path = self._create_path()
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Reset the environment to the initial state.
        '''
        super().reset(seed=seed)
        if self.terminated_counter % 5 == 0:
            self.path = self._switch_path()
            self.terminated_counter += 1
            print("self.terminated_counter: ", self.terminated_counter)
            print("The path was switched.")

        self.agent_yaw : float = 0.0
        self.desired_yaw_angle: float = 0.0
        self.tick : int = 0
        self.linear_velocity : float = self.min_linear_velocity
        self.yaw_angle_error : float = 0.0
        self.angular_velocity : float = 0.0
        self.current_goal_index : int = 1
        self.current_goal_distance : float = 0.0
        self.current_goal_position : float = self.path[0]
        self.current_position : np.ndarray = np.array([-0.3, 0.0])
        self.current_goals_window_position = self.path[1:self.goal_step * self.num_goals_window: self.goal_step]
        self.is_minor_goal_reached: bool = False
        self.goal_reached_counter: int = 0
        self.minor_goal_reacher_counter: int = 0
        self.is_goal_reached: bool = False
        self.is_terminated: bool = False

        observation = self._get_obs()
        info = self._get_info()

        return observation
        # return observation, info

    def step(self, action):
        '''
        Step the environment with the given action.
        '''
        terminated = False

        self.linear_velocity = self.min_linear_velocity + (self.max_linear_velocity - self.min_linear_velocity) * ((action[0] + 1) / 2)
        self.angular_velocity = self.min_angular_velocity + (self.max_angular_velocity - self.min_angular_velocity) * ((action[1] + 1) / 2)

        # Current position
        self._update_agent_position()
        self._get_angle_error()
        self._is_goal_reached()
        self._is_terminated()

        terminated = self.is_terminated

        if terminated:
            self.terminated_counter += 1

        reward = self._rewards()
        observation = self._get_obs()
        info = self._get_info()
        self.tick += 1
        truncated = self._is_truncated()

        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human') -> None:
        '''
        Render the environment.
        '''

        # Verify if fig is already created and create it if not
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()

            # Create markers
            self.path_marker, = self.ax.plot([], [], linestyle='-', color='blue', label='Path')
            self.agent_marker, = self.ax.plot([], [], marker='o', color='black', label='Agent')
            self.current_goal_marker, = self.ax.plot([], [], marker='x', color='green', label='Current Goal')
            self.current_goals_window_marker, = self.ax.plot([], [], marker='x', color='magenta', label='Goals Window')
            self.agent_front, = self.ax.plot([], [], linestyle='-', color='red', label='Agent Nose')
            self.desired_yaw, = self.ax.plot([], [], linestyle='--', color='orange', label='Desired Yaw')

            self.distance_title = self.ax.set_title('Distance to Goal: 0.00m | Current tick: 0')

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
        agent_x, agent_y = self.current_position
        self.agent_marker.set_data([agent_x], [agent_y])

        front_x = agent_x + 0.5 * np.cos(self.agent_yaw)
        front_y = agent_y + 0.5 * np.sin(self.agent_yaw)
        self.agent_front.set_data([agent_x, front_x], [agent_y, front_y])

        desired_yaw_x = agent_x + 0.5 * np.cos(self.desired_yaw_angle + self.agent_yaw)
        desired_yaw_y = agent_y + 0.5 * np.sin(self.desired_yaw_angle + self.agent_yaw)
        self.desired_yaw.set_data([agent_x, desired_yaw_x], [agent_y, desired_yaw_y])

        if self.current_goals_window_position.size > 0:
            multi_goals_x, multi_goals_y = zip(*self.current_goals_window_position)
            self.current_goals_window_marker.set_data(multi_goals_x, multi_goals_y)

        self.distance_title.set_text(f'Goal distance: {self.current_goal_distance:.2f}m | Current tick: {self.tick}')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self) -> None:
        '''
        Close the environment.
        '''
        pass

    def rand_action(self) -> np.ndarray:
        '''
        Generate a random action.
        '''
        return np.random.uniform(self.action_space.low, self.action_space.high)
    
    def _get_info(self) -> None:
        '''
        Get the information about the current state.
        '''
        pass
    
    def _update_agent_position(self, dt: float = 0.01) -> None:
        '''
        Update the agent position based on the current state.
        '''
        # dx = self.linear_velocity * np.cos(self.angular_velocity)
        # dy = self.linear_velocity * np.sin(self.angular_velocity)
        # self.current_position += np.array([dx, dy])
        # self.agent_yaw = self.angular_velocity

        self.agent_yaw += self.angular_velocity * dt

        dx = self.linear_velocity * np.cos(self.agent_yaw)
        dy = self.linear_velocity * np.sin(self.agent_yaw)
        self.current_position += np.array([dx, dy])

    def _get_obs(self) -> np.ndarray:
        '''
        Get the current state of the environment.
        '''
        return np.array([
            self.current_goal_distance,
            self.linear_velocity,
            self.angular_velocity,
            self.desired_yaw_angle,
        ])
    
    def _rewards(self) -> float:
        '''
        Calculate the reward based on the current state.
        '''
        reward_goal_reached = 0.0
        reward_minor_goal_reached = 0.0
        sucess_reward = 0.0
        reward_distance = 0.0

        reward_distance = -self.current_goal_distance
        
        if self.is_goal_reached:
            reward_goal_reached = 10.0
        if self.is_minor_goal_reached:
            reward_minor_goal_reached = 5.0
        if self.is_terminated:
            sucess_reward = 100.0
        rewards = reward_goal_reached + sucess_reward + reward_minor_goal_reached + reward_distance

        return rewards
    
    def _goal_distance(self) -> float:
        '''
        Calculate the distance to the goal.
        '''
        return np.linalg.norm(self.current_position - self.current_goal_position)
    
    def _is_goal_reached(self) -> None:
        '''
        Check if the goal is reached and update if in this case.
        '''
        self.current_goal_distance = self._goal_distance()

        if self.current_goal_distance < self.goal_threshold:
            self.current_goal_index += self.goal_step
            self.goal_reached_counter += 1
            if self.current_goal_index < len(self.path):
                self.current_goal_position = self.path[self.current_goal_index]
            else:
                self.current_goal_position = self.path[-1]
            self._generate_goals_window()
            self.is_goal_reached = True
            return
        self._update_minor_goal()
        self.is_goal_reached = False
    
    def _update_minor_goal(self):
        '''
        Update the minor goal for the agent.
        '''
        distances = np.array([self._get_distance(self.current_position, goal) 
                         for goal in self.current_goals_window_position])
        
        if np.any(distances < self.goal_threshold):
            self.minor_goal_reacher_counter += 1
            closest_goal_index = np.argmin(distances)
            self.current_goal_index = closest_goal_index + self.current_goal_index + 1
            self.current_goal_position = self.path[self.current_goal_index]
            self._generate_goals_window()
            self.is_minor_goal_reached = True
            return
        self.is_minor_goal_reached = False
    
    def _is_terminated(self):
        '''
        Check if the agent is in the goal.
        '''
        if self._get_distance(self.current_position, self.path[-1]) < self.goal_threshold:
            self.is_terminated = True
            return
        self.is_terminated = False
    
    def _get_distance(self, p1 : np.ndarray, p2: np.ndarray):
        '''
        Calculate the distance between two points.
        '''
        return np.linalg.norm(p1 - p2)
    
    def _is_truncated(self) -> bool:
        '''
        Verify if the episode is done.
        '''
        timeout = self.tick >= self.max_tick
        out_of_bound = self.current_goal_distance > self.out_of_bound_threshold

        if not self.is_terminated and self.current_position[0] > self.path[-1][0]:
            out_of_bound = True 

        return (timeout or out_of_bound)
    
    def _switch_path(self) -> str:
        print("path switched")
        path_type = np.random.choice(['straight', 'sine'])
        path = self._create_path(path_type)
        return path

    def _create_path(self, path_type: str = 'straight') -> np.ndarray:
        '''
        Create path for the agent to follow.
        '''
        if path_type == 'straight':
            return np.array([[i, 0.0] for i in range(100)])

        if path_type == 'sine':
            x = np.arange(100)
            y = np.random.uniform(-2.5, 2.5) * np.exp(-0.5 * ((x - np.random.uniform(20, 80)) / 10)**2)
            return np.column_stack((x, y))
    
    def _get_angle_error(self) -> None:
        '''
        Calculate the angle error between the agent and the goal.
        '''
        # self.yaw_angle_error = np.arctan2(
        #     self.current_goal_position[1] - self.current_position[1],
        #     self.current_goal_position[0] - self.current_position[0]
        # )

        # self.desired_yaw_angle = self.yaw_angle_error - self.agent_yaw

        angle_to_goal = np.arctan2(
            self.current_goal_position[1] - self.current_position[1],
            self.current_goal_position[0] - self.current_position[0]
        )

        self.desired_yaw_angle = angle_to_goal - self.agent_yaw
        self.desired_yaw_angle = (self.desired_yaw_angle + np.pi) % (2 * np.pi) - np.pi
        # self.desired_yaw_angle = np.arctan2(np.sin(self.yaw_angle_error), np.cos(self.yaw_angle_error))
        # print("self.desired_yaw_angle: ", self.desired_yaw_angle)
    
    def _generate_goals_window(self) -> None:
        '''
        Generate a window of goals for the agent to follow.
        '''
        self.current_goals_window_position = self.path[self.current_goal_index + 1:self.current_goal_index + self.num_goals_window: self.goal_step]