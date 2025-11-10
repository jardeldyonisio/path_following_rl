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

class PathObstacleLidarEnv(gym.Env):
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

        # Generate random obstacle
        self._generate_obstacle()

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
        self._check_footprint_costmap_overlap()
        self._check_footprint_obstacle_overlap()
        self._check_obstacle_collision()
        self._get_angle_error()
        self._is_goal_reached()
        self._is_terminated()
        
        # Update LiDAR ray distances
        self.ray_distances = self._cast_rays()

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

            # Obstacle marker
            self.obstacle_marker = Circle((0, 0), 
                                    0,  # Initial radius 0, will be updated
                                    fill=True,
                                    color='red',
                                    alpha=0.7,
                                    label='Obstacle')
            self.ax.add_patch(self.obstacle_marker)

            # Obstacle inflation
            self.obstacle_costmap = Circle((0, 0),
                                     0,  # Initial radius 0, will be updated
                                     fill=False,
                                     color='red',
                                     linestyle=':',
                                     linewidth=2,
                                     alpha=0.5,
                                     label='Obstacle Costmap')
            self.ax.add_patch(self.obstacle_costmap)
            
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

        # Update obstacle and costmap in world space
        if hasattr(self, 'obstacle') and self.obstacle is not None:
            obstacle_x, obstacle_y = self.obstacle['position']
            obstacle_radius = self.obstacle['radius']

            # Update core obstacle
            self.obstacle_marker.set_center((obstacle_x, obstacle_y))
            self.obstacle_marker.set_radius(obstacle_radius)
            self.obstacle_marker.set_visible(True)
            
            # Update costmap (core + inflation radius)
            costmap_radius = obstacle_radius + self.obstacle_inflation_radius
            self.obstacle_costmap.set_center((obstacle_x, obstacle_y))
            self.obstacle_costmap.set_radius(costmap_radius)
            self.obstacle_costmap.set_visible(True)
        else:
            self.obstacle_marker.set_visible(False)
            self.obstacle_costmap.set_visible(False)

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
            *self.distances
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
        collision_penalty = 0.0
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
            
        # Add obstacle penalties (no collision penalty - episode ends instead)
        if hasattr(self, 'footprint_obstacle_overlap') and self.footprint_obstacle_overlap:
            footprint_obstacle_penalty = -50.0  # High penalty for footprint touching obstacle
            
        if hasattr(self, 'footprint_costmap_overlap') and self.footprint_costmap_overlap:
            footprint_costmap_penalty = -25.0  # Moderate penalty for entering inflation zone
            
        rewards = (reward_goal_reached + sucess_reward + reward_subgoal_reached + 
                  reward_distance + truncated_reward + 
                  footprint_obstacle_penalty + footprint_costmap_penalty)

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

    def _generate_obstacle(self) -> None:
        '''
        Generate an obstacle in the environment.

        TODO: Make it works for different shapes
        '''
        # If path is too short or empty, do not place an obstacle
        if self.path is None or len(self.path) < 2:
            self.obstacle = None
            return

        # Get a random position on the path for the obstacle
        obstacle_index = np.random.randint(0, len(self.path) - 1)
        obstacle_position = self.path[obstacle_index]

        # Random size for the obstacle
        obstacle_radius = np.random.uniform(0.1, 1.0)

        # Create an obstacle at the position
        self.obstacle = {
            'position': obstacle_position,
            'radius': obstacle_radius
        }

    def _check_obstacle_collision(self) -> None:
        '''
        Check if the agent collides with the obstacle using simple distance calculation.
        '''
        if self.obstacle is None:
            self.obstacle_collision = False
            return
        
        # Current implementation: Circle-Circle overlap
        agent_pos = self.current_position
        obstacle_pos = np.array(self.obstacle['position'])
        
        self.obstacle_distance = self._get_distance(agent_pos, obstacle_pos)
        
        # True geometric overlap (no safety margins)
        agent_radius = self.agent_radius
        obstacle_radius = self.obstacle['radius']
        
        # Overlap occurs when distance < sum of radii
        self.obstacle_collision = self.obstacle_distance < (agent_radius + obstacle_radius)
        if self.obstacle_collision:
            print("Collision detected with obstacle!")

    def _check_footprint_obstacle_overlap(self) -> None:
        '''
        Check if the agent's footprint overlaps with the obstacle.
        '''
        if self.obstacle is None:
            self.footprint_obstacle_overlap = False
            return
        
        # Current implementation: Circle-Circle overlap
        agent_pos = self.current_position
        obstacle_pos = np.array(self.obstacle['position'])
        
        distance = self._get_distance(agent_pos, obstacle_pos)
        
        # True geometric overlap with footprint
        agent_footprint_radius = self.agent_footprint_radius
        obstacle_radius = self.obstacle['radius']
        
        # Overlap occurs when distance < sum of radii
        self.footprint_obstacle_overlap = distance < (agent_footprint_radius + obstacle_radius)

        if self.footprint_obstacle_overlap:
            print("Footprint overlaps with obstacle!")
        
    def _check_footprint_costmap_overlap(self) -> None:
        '''
        Check if the agent's footprint overlaps with the obstacle's costmap.
        '''
        if self.obstacle is None:
            self.footprint_costmap_overlap = False
            return
        
        # Current implementation: Circle-Circle overlap with inflation
        agent_pos = self.current_position
        obstacle_pos = np.array(self.obstacle['position'])
        
        distance = self._get_distance(agent_pos, obstacle_pos)
        
        # Inflated geometric overlap
        agent_footprint_radius = self.agent_footprint_radius
        obstacle_inflation = self.obstacle['radius'] + self.obstacle_inflation_radius
        
        # Overlap occurs when distance < sum of inflated radii
        self.footprint_costmap_overlap = distance < (agent_footprint_radius + obstacle_inflation)

        if self.footprint_costmap_overlap:
            print("Footprint overlaps with obstacle costmap!")
    
    def _cast_rays(self) -> np.ndarray:
        '''
        Cast rays in multiple directions to simulate LiDAR and measure distances to obstacles.
        
        @return: Array of distances for each ray (normalized to [0,1])
        '''
        ray_distances = np.full(self.num_rays, self.lidar_range)  # Initialize with max range
        
        if self.obstacle is None:
            return ray_distances / self.lidar_range  # Normalize to [0,1]
        
        # Generate ray angles relative to agent's current heading
        # Rays span from -lidar_fov/2 to +lidar_fov/2 relative to agent's yaw
        if self.num_rays == 1:
            ray_angles = [0.0]
        else:
            ray_angles = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, self.num_rays)
        
        obstacle_pos = np.array(self.obstacle['position'])
        obstacle_radius = self.obstacle['radius']  # Use only core obstacle radius (no inflation)
        
        for i, relative_angle in enumerate(ray_angles):
            # Calculate absolute ray direction in world coordinates
            ray_angle = self.agent_yaw + relative_angle
            ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            # Ray-circle intersection calculation
            # Ray: P = agent_pos + t * ray_direction (t >= 0)
            # Circle: |P - obstacle_pos|² = obstacle_radius²
            
            # Vector from agent to obstacle center (corrected)
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
                    closest_t = min(valid_intersections)
                    distance = closest_t
                    
                    # Clamp distance to lidar range
                    if distance <= self.lidar_range:
                        ray_distances[i] = distance
        
        # Normalize distances to [0,1] where 0 = max_range (far), 1 = 0 (very close)
        normalized_distances = 1.0 - (ray_distances / self.lidar_range)
        return np.clip(normalized_distances, 0.0, 1.0)