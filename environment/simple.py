#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

        self.agent_radius: float = 0.105 # Based on turtlebot3
        self.agent_footprint_radius: float = self.agent_radius * 1.2

        self.min_goal_distance: float = 0.0
        self.max_goal_distance: float = self.out_of_bound_threshold

        self.min_yaw_error: float = -np.pi * 2
        self.max_yaw_error: float = np.pi * 2

        self.terminated_counter: int = 1
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # State: [distance_to_goal, linear_vel, angular_vel, yaw_error]
        self.observation_space = gym.spaces.Box(
            low=np.array([self.min_goal_distance, 
                          -1.0, 
                          -1.0, 
                          self.min_yaw_error
                          ] + [self.min_goal_distance] * self.num_goals_window),
            high=np.array([self.max_goal_distance, 
                           1.0, 
                           1.0, 
                           self.max_yaw_error
                           ] + [self.max_goal_distance] * self.num_goals_window),
            dtype=np.float32
        )

        self.previous_action = np.zeros(self.action_space.shape[0])
        self.current_action = np.zeros(self.action_space.shape[0])

        self.path = self._create_path()
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Reset the environment to the initial state.
        '''
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.path = self._switch_path()
        self._generate_obstacle()

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
        self.is_subgoal_reached: bool = False
        self.goal_reached_counter: int = 0
        self.subgoal_reacher_counter: int = 0
        self.is_goal_reached: bool = False
        self.is_terminated: bool = False
        self.is_truncated: bool = False
        self.distances = np.zeros(self.num_goals_window)
        self.obstacle_collision: bool = False

        observation = self._get_obs()
        info = self._get_info()

        return observation
        # return observation, info

    def step(self, action):
        '''
        Step the environment with the given action.
        '''
        terminated = False

        self.previous_action = self.current_action.copy() if self.current_action is not None else np.array([0.0, 0.0])
        self.current_action = np.array(action)

        self.linear_velocity = self.min_linear_velocity + (self.max_linear_velocity - self.min_linear_velocity) * ((self.current_action[0] + 1) / 2)
        self.angular_velocity = self.min_angular_velocity + (self.max_angular_velocity - self.min_angular_velocity) * ((self.current_action[1] + 1) / 2)

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
        self._is_truncated()

        return observation, reward, terminated, self.is_truncated, info
    
    def render(self, mode='human') -> None:
        '''
        Render the environment.
        '''

        # Verify if fig is already created and create it if not
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()

            # Create markers
            self.path_marker, = self.ax.plot([], 
                                             [], 
                                             linestyle='-', 
                                             color='blue', 
                                             label='Path')
            self.agent_marker = Circle((0, 0), 
                                       self.agent_radius, 
                                       fill=True,
                                       color='black',
                                       label='Agent')
            self.ax.add_patch(self.agent_marker)

            self.current_goal_marker, = self.ax.plot([], [], marker='x', color='green', label='Current Goal')
            self.current_goals_window_marker, = self.ax.plot([], [], marker='x', color='magenta', label='Goals Window')
            self.agent_front, = self.ax.plot([], [], linestyle='-', color='red', label='Agent Nose')
            self.desired_yaw, = self.ax.plot([], [], linestyle='--', color='orange', label='Desired Yaw')
            
            self.obstacle_marker = Circle((0, 0), 
                                    0,  # Initial radius 0, will be updated
                                    fill=True,
                                    color='red',
                                    alpha=0.7,
                                    label='Obstacle')
            self.ax.add_patch(self.obstacle_marker)
            
            self.agent_footprint_marker = Circle((0, 0), 
                                           self.agent_footprint_radius, 
                                           fill=False,
                                           color='gray',
                                           linestyle='--',
                                           linewidth=2,
                                           label='Agent Footprint')
            self.ax.add_patch(self.agent_footprint_marker)

            self.distance_title = self.ax.set_title('Distance to Goal: 0.00m | Current tick: 0')

            self.ax.set_xlim(-5, 100)
            self.ax.set_ylim(-8, 8)
            self.ax.set_aspect('equal')
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
        self.agent_marker.set_center((agent_x, agent_y))

        front_x = agent_x + 0.5 * np.cos(self.agent_yaw)
        front_y = agent_y + 0.5 * np.sin(self.agent_yaw)
        self.agent_front.set_data([agent_x, front_x], [agent_y, front_y])

        desired_yaw_x = agent_x + 0.5 * np.cos(self.desired_yaw_angle + self.agent_yaw)
        desired_yaw_y = agent_y + 0.5 * np.sin(self.desired_yaw_angle + self.agent_yaw)
        self.desired_yaw.set_data([agent_x, desired_yaw_x], [agent_y, desired_yaw_y])

        if self.current_goals_window_position.size > 0:
            multi_goals_x, multi_goals_y = zip(*self.current_goals_window_position)
            self.current_goals_window_marker.set_data(multi_goals_x, multi_goals_y)
 
        # Update obstacle in world space
        if hasattr(self, 'obstacle') and self.obstacle is not None:
            obstacle_x, obstacle_y = self.obstacle['position']
            obstacle_size = self.obstacle['size']
            # Update obstacle circle position and size in world space
            self.obstacle_marker.set_center((obstacle_x, obstacle_y))
            self.obstacle_marker.set_radius(obstacle_size / 2)  # Assuming size is diameter
            self.obstacle_marker.set_visible(True)
        else:
            self.obstacle_marker.set_visible(False)

        self.agent_footprint_marker.set_center((agent_x, agent_y))

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
        return np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0)
        ], dtype=np.float32)

    def _get_info(self) -> None:
        '''
        Get the information about the current state.
        '''
        pass
    
    def _update_agent_position(self, dt: float = 0.1) -> None:
        '''
        Update the agent position based on the current state.
        '''
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
            self.previous_action[0],
            self.previous_action[1],
            self.desired_yaw_angle,
            *self.distances,
            self.obstacle['position'][0],
            self.obstacle['position'][1],
            self.obstacle['size']
        ])

    def _rewards(self) -> float:
        '''
        Calculate the reward based on the current state.
        '''
        reward_goal_reached = 0.0
        reward_subgoal_reached = 0.0
        sucess_reward = 0.0
        reward_distance = 0.0
        truncated_reward = 0.0
        reward_obstacle_collision = 0.0

        reward_distance = -self.current_goal_distance * 0.1
        
        if self.obstacle_collision:
            reward_obstacle_collision -= 100.0
        if self.is_goal_reached:
            reward_goal_reached = 10.0
        if self.is_subgoal_reached:
            reward_subgoal_reached = 5.0
        if self.is_terminated:
            sucess_reward = 100.0
        if self.is_truncated:
            truncated_reward = -100.0
        
        rewards = reward_goal_reached + \
                  sucess_reward + \
                  reward_subgoal_reached + \
                  reward_distance + \
                  truncated_reward + \
                  reward_obstacle_collision

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
        self._update_subgoal()
        self.is_goal_reached = False
    
    def _update_subgoal(self):
        '''
        Update the minor goal for the agent.
        '''

        # Calculate distances to available goals using correct variable names
        available_distances = np.array([self._get_distance(self.current_position, goal) 
                            for goal in self.current_goals_window_position])
        
        # Initialize distances array
        self.distances = np.full(self.num_goals_window, 0.0)
        
        if len(available_distances) > 0:
            # Fill real distances
            self.distances[:len(available_distances)] = available_distances
            
            if len(available_distances) < self.num_goals_window:
                last_distance = available_distances[-1]
                self.distances[len(available_distances):] = last_distance
        else:
            self.distances.fill(self.current_goal_distance)

        # Check if any minor goal is reached
        if len(available_distances) > 0 and np.any(available_distances < self.goal_threshold):
            closest_goal_index = np.argmin(available_distances)
            # self.current_goal_index = closest_goal_index + self.current_goal_index + 1
            # reached_subgoal_index = (self.current_goal_index + 1) + closest_goal_index * self.goal_step
            # self.current_goal_index = reached_subgoal_index + self.goal_step
            self.current_goal_index = (self.current_goal_index + 1) + (closest_goal_index + 1) * self.goal_step
            # self.current_goal_index = (self.current_goal_index + 1) + closest_goal_index * self.goal_step # Dranaju fixed version
            if self.current_goal_index < len(self.path):
                self.current_goal_position = self.path[self.current_goal_index]
            else:
                self.current_goal_position = self.path[-1]
            self._generate_goals_window()
            self.is_subgoal_reached = True
            return
        self.is_subgoal_reached = False
    
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

        if timeout or out_of_bound:
            self.is_truncated = True

    def _switch_path(self) -> str:
        path_type = np.random.choice(['straight', 'sine', 'zigzag'])
        path = self._create_path(path_type)
        return path

    def _create_path(self, path_type: str = 'straight') -> np.ndarray:
        '''
        Create path for the agent to follow.
        '''
        random_path_size = np.random.randint(50, 100)

        if path_type == 'straight':
            y_point = np.random.uniform(-2.5, 2.5)
            x_start = np.random.randint(0, 20)
            return np.array([[x_start + i, y_point] for i in range(random_path_size)])

        if path_type == 'sine':
            x = np.arange(random_path_size)
            y = np.random.uniform(-2.5, 2.5) * np.exp(-0.5 * ((x - np.random.uniform(20, 80)) / 10)**2)
            return np.column_stack((x, y))
        
        if path_type == 'zigzag':
            x = np.arange(random_path_size)
            y = np.sin(x / 5) * 2.5
            return np.column_stack((x, y))
    
    def _get_angle_error(self) -> None:
        '''
        Calculate the angle error between the agent and the goal.
        '''
        angle_to_goal = np.arctan2(
            self.current_goal_position[1] - self.current_position[1],
            self.current_goal_position[0] - self.current_position[0]
        )

        self.desired_yaw_angle = angle_to_goal - self.agent_yaw
        self.desired_yaw_angle = (self.desired_yaw_angle + np.pi) % (2 * np.pi) - np.pi
    
    def _generate_goals_window(self) -> None:
        '''
        Generate a window of goals for the agent to follow.
        '''
        self.current_goals_window_position = self.path[self.current_goal_index + 1:self.current_goal_index + self.num_goals_window: self.goal_step]

    def _generate_obstacle(self) -> None:
        '''
        Generate an obstacle in the environment.

        TODO: Make it works for different shapes
        '''
        # Get a random position on the path for the obstacle
        obstacle_index = np.random.randint(0, len(self.path) - 1)
        obstacle_position = self.path[obstacle_index]
        obstacle_size = np.random.uniform(0.1, 1.0)

        # Create an obstacle at the position
        self.obstacle = {
            'position': obstacle_position,
            'size': obstacle_size
        }

    def _get_agent_boundary(self) -> np.ndarray:
        '''
        Get the boundary of the agent as a circle.
        '''
        num_points = 16  # Number of points to approximate the circle
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Generate circle points around the agent's center position
        circle_points = []
        for angle in angles:
            x = self.current_position[0] + self.agent_radius * np.cos(angle)
            y = self.current_position[1] + self.agent_radius * np.sin(angle)
            circle_points.append([x, y])
        
        return np.array(circle_points)
    
    def _get_obstacle_boundary(self) -> np.ndarray:
        '''
        Get the boundary of the obstacle.
        '''
        if self.obstacle is None:
            return None
        
        obstacle_x, obstacle_y = self.obstacle['position']
        obstacle_size = self.obstacle['size']
        obstacle_radius = obstacle_size

        return np.array([
            [obstacle_x - obstacle_radius, obstacle_y - obstacle_radius],
            [obstacle_x + obstacle_radius, obstacle_y - obstacle_radius],
            [obstacle_x + obstacle_radius, obstacle_y + obstacle_radius],
            [obstacle_x - obstacle_radius, obstacle_y + obstacle_radius]
        ])
    
    def _check_obstacle_collision(self) -> None:
        '''
        Check if the agent collides with the obstacle.
        '''
        agent_boundary = self._get_agent_boundary()
        obstacle_boundary = self._get_obstacle_boundary()

        if obstacle_boundary is None:
            return
        
        # Check if any point of the agent's boundary is inside the obstacle's boundary
        for point in agent_boundary:
            if (obstacle_boundary[0][0] <= point[0] <= obstacle_boundary[1][0] and
                obstacle_boundary[0][1] <= point[1] <= obstacle_boundary[2][1]):
                self.obstacle_collision = True
                return
        
        self.obstacle_collision = False

        
