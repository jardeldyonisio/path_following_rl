#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym

from typing import Optional 

'''
This class implements a single-agent environment for path following.
'''

class SimplePathFollowingEnv(gym.Env):
    def __init__(self, buffer_size=1000):
        '''
        
        '''
        super(SimplePathFollowingEnv, self).__init__()
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        self.action_space = gym.spaces.Box(
            low=np.array([0.05, -0.5]),
            high=np.array([1.0, 0.5]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Reset the environment to the initial state.
        '''
        super().reset(seed=seed)
        self.path = self._create_path()

        self.time : int = 0
        self.linear_velocity : float = 0.0
        self.yaw_angle_error : float = 0.0
        self.angular_velocity : float = 0.0
        self.current_goal_index : int = 1
        self.current_goal_distance : float = 0.0
        self.current_goal_position : float = self.path[0]

        self.arrived : bool = False
        self.timeout : bool = False
        self.out_of_bound : bool = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action, dt : float=0.01):
        '''
        Step the environment with the given action.
        '''
        self.linear_velocity, self.angular_velocity = action
        self.angular_velocity = self.angular_velocity * dt

        terminated = self._is_goal_reached()
        truncated = self._is_truncated()

        reward = self._rewards()
        observation = self._get_obs()
        info = self._get_info()
        self.time =+ 1

        return observation, reward, terminated, truncated, info
    
    def _get_info(self):
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
            self.yaw_angle_error,
        ])
    
    def _rewards(self):
        '''
        Calculate the reward based on the current state.
        '''
        if self._is_goal_reached():
            reward_goal_reached = 10.0
        rewards = reward_goal_reached

        return rewards
    
    def _goal_distance(self):
        '''
        Calculate the distance to the goal.
        '''
        return np.linalg.norm(self.current_position - self.current_goal_position)
    
    def _is_goal_reached(self, step : int = 1):
        '''
        Check if the goal is reached and update if in this case.
        '''
        if self._goal_distance() < 0.2:
            self.current_goal_position = self.path[self.current_goal_index]
            self.current_goal_index =+ step
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
        self.timeout = self.time > 1000
        self.out_of_bound = self._goal_distance() > 10.0

        return (self.timeout or self.out_of_bound)
    
    def _create_path(self):
        '''
        Create path for the agent to follow.
        '''
        return np.array([[i, 0.0] for i in range(100)])