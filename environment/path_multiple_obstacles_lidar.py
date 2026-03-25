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

class PathMultiObstaclesLidarEnv(gym.Env):
    def __init__(self):
        '''
        @brief Initialize the environment.
        '''
        super().__init__()

        # Simulation parameters
        self.max_tick: int = 800
        self.terminated_counter: int = 1

        # Goals parameters
        self.goal_step: int = 1
        self.num_goals_window: int = 15
        self.out_of_bound_threshold: int = (self.goal_step * self.num_goals_window) + self.goal_step
        self.goal_threshold: float = 0.2
        self.min_goal_distance: float = 0.0
        self.max_goal_distance: float = self.out_of_bound_threshold
        self.min_yaw_error: float = -np.pi * 2
        self.max_yaw_error: float = np.pi * 2

        # Agents parameters
        self.agent_radius: float = 0.105 # Based on turtlebot3
        self.agent_footprint_radius: float = self.agent_radius * 1.2
        self.min_linear_velocity: float = 0.01
        self.max_linear_velocity: float = 0.25
        self.min_angular_velocity: float = -0.5
        self.max_angular_velocity: float = 0.5

        # Obstacle parameters
        self.obstacle_inflation_radius: float = 0.2

        # LiDAR parameters
        self.num_rays: int = 24  # Number of rays to cast
        self.lidar_range: float = 2.5  # Maximum detection range
        self.lidar_fov: float = np.pi  # Field of view (±90° ahead)

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # State: [distance_to_goal, linear_vel, angular_vel, yaw_error, ray_distances..., goal_distances...]
        self.observation_space = gym.spaces.Box(
            low=np.array([self.min_goal_distance, 
                          -1.0, 
                          -1.0, 
                          self.min_yaw_error,
                          ] + [0.0] * self.num_rays + [self.min_goal_distance] * self.num_goals_window),
            high=np.array([self.max_goal_distance, 
                           1.0, 
                           1.0, 
                           self.max_yaw_error
                           ] + [1.0] * self.num_rays + [self.max_goal_distance] * self.num_goals_window),
            dtype=np.float32
        )

        self.previous_action = np.zeros(self.action_space.shape[0])
        self.current_action = np.zeros(self.action_space.shape[0])

        self.path = self._create_path()
        
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[dict] = None):
        '''
        @brief Reset the environment to the initial state.

        @param seed: Optional seed for the environment.
        @param options: Optional dictionary of options.
        @return: The initial observation of the environment.
        '''
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Create a new path
        self.path = self._switch_path()

        # Generate random obstacles
        self._generate_obstacles()

        # Simulation parameters
        self.tick : int = 0
        self.is_terminated: bool = False
        self.is_truncated: bool = False

        # Agent parameters
        self.agent_yaw : float = 0.0
        self.desired_yaw_angle: float = 0.0
        self.linear_velocity : float = self.min_linear_velocity
        self.angular_velocity : float = 0.0

        # Goals parameters
        self.yaw_angle_error : float = 0.0
        self.current_goal_index : int = 1
        self.current_goal_distance : float = 0.0
        self.current_goal_position : float = self.path[0]
        self.current_position : np.ndarray = np.array([-0.3, 0.0])
        self.current_goals_window_position = self.path[1:self.goal_step * self.num_goals_window: self.goal_step]
        self.is_subgoal_reached: bool = False
        self.goal_reached_counter: int = 0
        self.subgoal_reacher_counter: int = 0
        self.is_goal_reached: bool = False
        self.distances = np.zeros(self.num_goals_window)

        self.yaw_error_to_path = 0.0

        # Initialize LiDAR ray distances
        self.ray_distances = self._cast_rays()

        observation = self._get_obs()
        info = self._get_info()

        return observation
        # return observation, info

    def step(self, action):
        '''
        @brief Step the environment with the given action.
        '''
        terminated = False

        self.previous_action = self.current_action.copy() if self.current_action is not None else np.array([0.0, 0.0])
        self.current_action = np.array(action)

        self.linear_velocity = self.min_linear_velocity + (self.max_linear_velocity - self.min_linear_velocity) * ((self.current_action[0] + 1) / 2)
        self.angular_velocity = self.min_angular_velocity + (self.max_angular_velocity - self.min_angular_velocity) * ((self.current_action[1] + 1) / 2)

        self._update_agent_position()
        
        # Update LiDAR ray distances BEFORE collision checks
        # so the agent can see obstacles in the next observation
        self.ray_distances = self._cast_rays()

        # Just testing
        self._get_closest_point_direction()
        
        self._check_footprint_costmap_overlap()
        self._check_footprint_obstacle_overlap()
        self._check_obstacle_collision()
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
    
    def render(self, 
               mode='human') -> None:
        '''
        Render the environment.

        @param mode: The mode to render the environment.
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

            # Initialize obstacle markers lists (will be populated dynamically)
            self.obstacle_markers = []
            self.obstacle_costmaps = []
            
            # Agent footprint
            self.agent_footprint_marker = Circle((0, 0),
                                           self.agent_footprint_radius, 
                                           fill=False,
                                           color='gray',
                                           linestyle='--',
                                           linewidth=2,
                                           label='Agent Footprint')
            self.ax.add_patch(self.agent_footprint_marker)

            self.current_goal_marker, = self.ax.plot([], [], marker='x', color='green', label='Current Goal')
            self.current_goals_window_marker, = self.ax.plot([], [], marker='x', color='magenta', label='Goals Window')
            self.agent_front, = self.ax.plot([], [], linestyle='-', color='red', label='Agent Nose')
            self.desired_yaw, = self.ax.plot([], [], linestyle='--', color='orange', label='Desired Yaw')
            
            # LiDAR ray markers
            self.lidar_rays = []
            for i in range(self.num_rays):
                ray_line, = self.ax.plot([], [], linestyle='-', color='cyan', alpha=0.6, linewidth=1)
                self.lidar_rays.append(ray_line)
            
            self.distance_title = self.ax.set_title('Distance to Goal: 0.00m | Current tick: 0')

            self.ax.legend()
            plt.ion()
            plt.show(block = False)

        self.ax.set_xlim(self.current_position[0] - 10, self.current_position[0] + 10)
        self.ax.set_ylim(self.current_position[1] - 5, self.current_position[1] + 5)
        
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

        # Update obstacles and costmaps in world space
        if hasattr(self, 'obstacles') and len(self.obstacles) > 0:
            # Ensure we have enough markers for all obstacles
            while len(self.obstacle_markers) < len(self.obstacles):
                # Create new obstacle marker
                obstacle_marker = Circle((0, 0), 0, fill=True, color='red', alpha=0.7)
                self.ax.add_patch(obstacle_marker)
                self.obstacle_markers.append(obstacle_marker)
                
                # Create new costmap marker  
                costmap_marker = Circle((0, 0), 0, fill=False, color='red', 
                                      linestyle=':', linewidth=2, alpha=0.5)
                self.ax.add_patch(costmap_marker)
                self.obstacle_costmaps.append(costmap_marker)
            
            # Update visible obstacles
            for i, obstacle in enumerate(self.obstacles):
                obstacle_x, obstacle_y = obstacle['position']
                obstacle_radius = obstacle['radius']
                
                # Update core obstacle
                self.obstacle_markers[i].set_center((obstacle_x, obstacle_y))
                self.obstacle_markers[i].set_radius(obstacle_radius)
                self.obstacle_markers[i].set_visible(True)
                
                # Update costmap (core + inflation radius)
                costmap_radius = obstacle_radius + self.obstacle_inflation_radius
                self.obstacle_costmaps[i].set_center((obstacle_x, obstacle_y))
                self.obstacle_costmaps[i].set_radius(costmap_radius)
                self.obstacle_costmaps[i].set_visible(True)
            
            # Hide unused markers
            for i in range(len(self.obstacles), len(self.obstacle_markers)):
                self.obstacle_markers[i].set_visible(False)
                self.obstacle_costmaps[i].set_visible(False)
        else:
            # Hide all obstacle markers when no obstacles
            for marker in self.obstacle_markers:
                marker.set_visible(False)
            for costmap in self.obstacle_costmaps:
                costmap.set_visible(False)

        self.agent_footprint_marker.set_center((agent_x, agent_y))

        # Update LiDAR rays visualization
        if hasattr(self, 'ray_distances'):
            if self.num_rays == 1:
                ray_angles = [0.0]
            else:
                ray_angles = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, self.num_rays)
            
            for i, (ray_line, relative_angle) in enumerate(zip(self.lidar_rays, ray_angles)):
                ray_angle = self.agent_yaw + relative_angle
                
                # Convert normalized distance back to actual distance for visualization
                actual_distance = (1.0 - self.ray_distances[i]) * self.lidar_range
                
                end_x = agent_x + actual_distance * np.cos(ray_angle)
                end_y = agent_y + actual_distance * np.sin(ray_angle)
                
                ray_line.set_data([agent_x, end_x], [agent_y, end_y])

        self.distance_title.set_text(f'Goal distance: {self.current_goal_distance:.2f}m | Current tick: {self.tick}')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self) -> None:
        '''
        @brief Close the environment.
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
            *self.ray_distances,
            *self.distances # Distances to subgoals
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
        
        # Obstacle penalty rewards
        direction_penalty = 0.0
        footprint_obstacle_penalty = 0.0
        footprint_costmap_penalty = 0.0

        reward_distance = -self.current_goal_distance * 0.1
        
        if self.is_goal_reached:
            reward_goal_reached = 10.0
        if self.is_subgoal_reached:
            reward_subgoal_reached = 5.0
        if self.is_terminated:
            sucess_reward = 100.0
        if self.is_truncated:
            truncated_reward = -100.0
        if abs(self.yaw_error_to_path) > 0.15:
            direction_penalty = -1.0 

        # Add a reward that penalties when the agent orientation is wrong

        # Add a reward that penalties when the agent is going to a different direction of the path

        # if self.is_terminated:
        #     reward_success = 100.0 * (1 - self.tick / self.max_tick)
            
        # Add obstacle penalties (no collision penalty - episode ends instead)
        if hasattr(self, 'footprint_obstacle_overlap') and self.footprint_obstacle_overlap:
            footprint_obstacle_penalty = -50.0  # High penalty for footprint touching obstacle
            
        if hasattr(self, 'footprint_costmap_overlap') and self.footprint_costmap_overlap:
            footprint_costmap_penalty = -25.0  # Moderate penalty for entering inflation zone
            
        rewards = (reward_goal_reached + sucess_reward + reward_subgoal_reached + 
                  reward_distance + truncated_reward + 
                  footprint_obstacle_penalty + footprint_costmap_penalty + direction_penalty)

        return rewards
    
    def _goal_distance(self) -> float:
        '''
        Calculate the distance to the goal.
        '''
        return np.linalg.norm(self.current_position - self.current_goal_position)

    # def _is_goal_reached(self) -> None:
    #     '''
    #     Check if the goal is reached and update if in this case.
    #     '''
    #     self.current_goal_distance = self._goal_distance()

    #     if self.current_goal_distance < self.goal_threshold:
    #         self.current_goal_index += self.goal_step
    #         self.goal_reached_counter += 1
    #         if self.current_goal_index < len(self.path):
    #             self.current_goal_position = self.path[self.current_goal_index]
    #         else:
    #             self.current_goal_position = self.path[-1]
    #         self._generate_goals_window()
    #         self.is_goal_reached = True
    #         return
    #     self._update_subgoal()
    #     self.is_goal_reached = False

    def _is_goal_reached(self):
        """
        Check if the goal is reached (within threshold) or if the robot naturally
        passed the goal. In both cases, update the goal index,
        but only mark `is_goal_reached=True` when within the threshold.
        """

        # Distance to current goal
        self.current_goal_distance = self._goal_distance()

        # 1. Check normal "goal reached" condition
        goal_reached_by_threshold = self.current_goal_distance < self.goal_threshold

        # 2. Check "passed goal" condition
        passed_goal = False
        if self.current_goal_index < len(self.path) - 1:
            next_goal = self.path[self.current_goal_index + 1]
            dist_to_next = self._get_distance(self.current_position, next_goal)

            # If robot is closer to the next goal than the current one, it passed it
            if dist_to_next < self.current_goal_distance:
                passed_goal = True

        # 3. If goal was "reached" (threshold OR passed), update goal
        if goal_reached_by_threshold or passed_goal:

            # Advance goal index
            self.current_goal_index = min(
                self.current_goal_index + self.goal_step,
                len(self.path) - 1
            )

            self.current_goal_position = self.path[self.current_goal_index]
            self._generate_goals_window()

            # Only count as "is_goal_reached = True" when threshold reached
            if goal_reached_by_threshold:
                self.goal_reached_counter += 1
                self.is_goal_reached = True       # <-- HERE
            else:
                self.is_goal_reached = False      # passed the goal, not reached

            return

        # 4. Otherwise keep following subgoal
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

            # TODO: Verify if this logic is correct
            self.current_goal_index = (self.current_goal_index + 1) + (closest_goal_index + 1) * self.goal_step

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
        collision_occurred = hasattr(self, 'obstacle_collision') and self.obstacle_collision

        if not self.is_terminated and self.current_position[0] > self.path[-1][0]:
            out_of_bound = True 

        if timeout or out_of_bound or collision_occurred:
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

    def _generate_obstacles(self) -> None:
        '''
        Generate random obstacles in the environment.
        Randomly places between 0 and 5 obstacles along the path, ensuring some are within LiDAR range.
        '''
        # If path is too short or empty, do not place obstacles
        if self.path is None or len(self.path) < 2:
            self.obstacles = []
            return

        # Random number of obstacles (0 to 5)
        num_obstacles = np.random.randint(0, 6)  # 0 to 5 inclusive
        
        self.obstacles = []
        
        if num_obstacles == 0:
            return

        # Agent starts at (-0.3, 0.0), ensure at least some obstacles are within detectable range
        agent_start_pos = np.array([-0.3, 0.0])
        
        # Create multiple obstacles at different positions
        used_indices = set()
        min_distance_between_obstacles = 5  # Reduced minimum distance
        
        for obstacle_idx in range(num_obstacles):
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts:
                # For first 1-2 obstacles, place them within LiDAR detection range
                if obstacle_idx < 2 and np.random.random() < 0.8:
                    # Place very near beginning of path (within first few path points that are within LiDAR range)
                    # Based on debug: path indices 0-2 are within 2.5m range
                    obstacle_index = np.random.randint(0, min(5, len(self.path) // 4))
                else:
                    # Place anywhere along the path for variety
                    obstacle_index = np.random.randint(len(self.path) // 10, len(self.path) - 1)
                
                # Check if this position is too close to existing obstacles
                too_close = False
                for used_index in used_indices:
                    if abs(obstacle_index - used_index) < min_distance_between_obstacles:
                        too_close = True
                        break
                
                if not too_close:
                    used_indices.add(obstacle_index)
                    break
                    
                attempts += 1
            
            if attempts < max_attempts:  # Successfully found a valid position
                obstacle_position = self.path[obstacle_index].copy()
                
                # Add random offset under 1 meter to make obstacles more interesting while keeping them path-relevant
                # Calculate path direction for proper perpendicular offset
                if obstacle_index > 0 and obstacle_index < len(self.path) - 1:
                    # Get path direction vector
                    path_direction = self.path[obstacle_index + 1] - self.path[obstacle_index - 1]
                    path_direction = path_direction / np.linalg.norm(path_direction)
                    
                    # Calculate perpendicular direction (rotate 90 degrees)
                    perpendicular_direction = np.array([-path_direction[1], path_direction[0]])
                    
                    # Random offset under 1 meter along perpendicular direction
                    offset_distance = np.random.uniform(-1.0, 1.0)  # Random value under 1m
                    obstacle_position = obstacle_position + offset_distance * perpendicular_direction
                else:
                    # Fallback for edge cases - simple random offset under 1 meter
                    offset_distance = np.random.uniform(-1.0, 1.0)  # Random value under 1m
                    obstacle_position = obstacle_position + np.array([0, offset_distance])
                
                # Random size for the obstacle
                obstacle_radius = np.random.uniform(0.1, 1.0)
                
                # Create obstacle
                obstacle = {
                    'position': obstacle_position,
                    'radius': obstacle_radius
                }
                self.obstacles.append(obstacle)

    def _check_obstacle_collision(self) -> None:
        '''
        Check if the agent collides with any obstacle using simple distance calculation.
        '''
        self.obstacle_collision = False
        
        if not hasattr(self, 'obstacles') or len(self.obstacles) == 0:
            return
        
        # Check collision with all obstacles
        agent_pos = self.current_position
        agent_radius = self.agent_radius
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle['position'])
            obstacle_radius = obstacle['radius']
            
            distance = self._get_distance(agent_pos, obstacle_pos)
            
            # Overlap occurs when distance < sum of radii
            if distance < (agent_radius + obstacle_radius):
                self.obstacle_collision = True
                print("Collision detected with obstacle!")
                break  # No need to check other obstacles once collision is found

    def _check_footprint_obstacle_overlap(self) -> None:
        '''
        @brief Check if the agent's footprint overlaps with any obstacle.
        '''
        self.footprint_obstacle_overlap = False
        
        if not hasattr(self, 'obstacles') or len(self.obstacles) == 0:
            return
        
        # Check overlap with all obstacles
        agent_pos = self.current_position
        agent_footprint_radius = self.agent_footprint_radius
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle['position'])
            obstacle_radius = obstacle['radius']
            
            distance = self._get_distance(agent_pos, obstacle_pos)
            
            # Overlap occurs when distance < sum of radii
            if distance < (agent_footprint_radius + obstacle_radius):
                self.footprint_obstacle_overlap = True
                print("Footprint overlaps with obstacle!")
                break 

    def _get_closest_point_direction(self):
        if len(self.path) == 0:
            self.path_direction = 0.0
            self.yaw_error_to_path = 0.0
            return

        closest_point = None
        min_distance = float('inf')
        closest_index = -1

        # Find the closest point on the path
        for i, point in enumerate(self.path):
            distance = self._get_distance(self.current_position, point)
            if distance < min_distance:
                min_distance = distance
                closest_point = point
                closest_index = i

        # Compute local direction if valid
        if closest_index != -1:
            next_index = min(closest_index + 1, len(self.path) - 1)
            next_point = self.path[next_index]

            # Local path direction (yaw desired)
            self.path_direction = np.arctan2(
                next_point[1] - closest_point[1],
                next_point[0] - closest_point[0]
            )

            # Yaw error relative to the path
            self.yaw_error_to_path = np.arctan2(
                np.sin(self.path_direction - self.agent_yaw),
                np.cos(self.path_direction - self.agent_yaw)
            )

    def _check_footprint_costmap_overlap(self) -> None:
        '''
        Check if the agent's footprint overlaps with any obstacle's costmap.
        '''
        self.footprint_costmap_overlap = False
        
        if not hasattr(self, 'obstacles') or len(self.obstacles) == 0:
            return
        
        # Check overlap with all obstacles costmaps
        agent_pos = self.current_position
        agent_footprint_radius = self.agent_footprint_radius
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle['position'])
            obstacle_inflation = obstacle['radius'] + self.obstacle_inflation_radius
            
            distance = self._get_distance(agent_pos, obstacle_pos)
            
            # Overlap occurs when distance < sum of inflated radii
            if distance < (agent_footprint_radius + obstacle_inflation):
                self.footprint_costmap_overlap = True
                print("Footprint overlaps with obstacle costmap!")
                break  # No need to check other obstacles once overlap is found
    
    def _cast_rays(self) -> np.ndarray:
        '''
        Cast rays in multiple directions to simulate LiDAR and measure distances to obstacles.
        
        @return: Array of distances for each ray (normalized to [0,1])
        '''
        ray_distances = np.full(self.num_rays, self.lidar_range)  # Initialize with max range
        
        if not hasattr(self, 'obstacles') or len(self.obstacles) == 0:
            # When no obstacles, all rays reach maximum range
            # Return 0.0 (meaning far/no detection) for all rays
            return np.zeros(self.num_rays)
        
        # Generate ray angles relative to agent's current heading
        # Rays span from -lidar_fov/2 to +lidar_fov/2 relative to agent's yaw
        if self.num_rays == 1:
            ray_angles = [0.0]
        else:
            ray_angles = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, self.num_rays)
        
        for i, relative_angle in enumerate(ray_angles):
            # Calculate absolute ray direction in world coordinates
            ray_angle = self.agent_yaw + relative_angle
            ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            closest_distance = self.lidar_range
            
            # Check intersection with all obstacles
            for obstacle in self.obstacles:
                obstacle_pos = np.array(obstacle['position'])
                obstacle_radius = obstacle['radius']  # Use only core obstacle radius (no inflation)
                
                # Ray-circle intersection calculation
                # Ray: P = agent_pos + t * ray_direction (t >= 0)
                # Circle: |P - obstacle_pos|² = obstacle_radius²
                
                # Vector from agent to obstacle center
                d = self.current_position - obstacle_pos
                
                # Quadratic equation coefficients for intersection
                # |d + t * ray_direction|² = obstacle_radius²
                a = np.dot(ray_direction, ray_direction)  # Should be 1 for unit vector
                b = 2 * np.dot(d, ray_direction)
                c = np.dot(d, d) - obstacle_radius**2
                
                discriminant = b**2 - 4*a*c
                
                if discriminant >= 0:
                    # Ray intersects the circle
                    sqrt_discriminant = np.sqrt(discriminant)
                    t1 = (-b - sqrt_discriminant) / (2*a)
                    t2 = (-b + sqrt_discriminant) / (2*a)
                    
                    # We want the closest positive intersection
                    valid_intersections = [t for t in [t1, t2] if t >= 0]
                    
                    if valid_intersections:
                        obstacle_distance = min(valid_intersections)
                        
                        # Keep the closest obstacle intersection
                        if obstacle_distance <= self.lidar_range and obstacle_distance < closest_distance:
                            closest_distance = obstacle_distance
            
            ray_distances[i] = closest_distance
        
        # Normalize distances to [0,1] where 0 = max_range (far), 1 = 0 (very close)
        normalized_distances = 1.0 - (ray_distances / self.lidar_range)
        return np.clip(normalized_distances, 0.0, 1.0)