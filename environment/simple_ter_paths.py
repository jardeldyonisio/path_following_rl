#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from typing import Optional 

'''
This class implements a single-agent environment for path following.
'''

class SimpleTerPathFollowingEnv(gym.Env):

    def __init__(self, path_mode: str = "ficticio", paths_folder: str = "../paths", selected_path_name: str = None):
        '''
        path_mode: "real", "ficticio" ou "ambos"
        paths_folder: pasta onde estão os arquivos de caminho real
        selected_path_name: nome do arquivo de path desejado (ex: "path_circular.txt")
        '''
        super().__init__()
        self.max_tick = 800
        self.goal_step = 1
        self.num_goals_window = 15
        self.out_of_bound_threshold = (self.goal_step * self.num_goals_window) + self.goal_step
        self.goal_threshold = 0.2

        self.min_linear_velocity = 0.01
        self.max_linear_velocity = 0.25

        self.min_angular_velocity = -0.5
        self.max_angular_velocity = 0.5

        self.min_goal_distance = 0.0
        self.max_goal_distance = self.out_of_bound_threshold

        self.min_yaw_error = -np.pi * 2
        self.max_yaw_error = np.pi * 2

        self.terminated_counter = 1

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

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

        self.min_goal_separation = 0.2  # parâmetro ajustável

        self.path_mode = path_mode
        self.paths_folder = paths_folder
        self.selected_path_name = selected_path_name
        self.real_paths, self.real_paths_names = self._load_real_paths_with_names(self.paths_folder)
        self.path = self._switch_path()

    def _load_real_paths_with_names(self, folder_path):
        '''Lê todos os arquivos .txt da pasta e retorna lista de np.arrays e lista de nomes'''
        import os
        paths = []
        names = []
        print(f"Lendo arquivos de path em: {folder_path}")
        for fname in os.listdir(folder_path):
            if fname.endswith('.txt'):
                fpath = os.path.join(folder_path, fname)
                print(f"Arquivo encontrado: {fpath}")
                with open(fpath, 'r') as f:
                    pts = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Aceita separador vírgula ou espaço
                        if ',' in line:
                            vals = line.split(',')
                        else:
                            vals = line.split()
                        if len(vals) == 2:
                            try:
                                pts.append([float(vals[0]), float(vals[1])])
                            except ValueError:
                                print(f"[WARN] Linha ignorada no arquivo {fname}: {line}")
                    if pts:
                        paths.append(np.array(pts))
                        names.append(fname)
        return paths, names
    def reset(self, seed: Optional[int] = None):
        # Print explícito do path selecionado
        if self.selected_path_name is not None and self.real_paths:
            for i, name in enumerate(self.real_paths_names):
                if (self.selected_path_name == name or self.selected_path_name + ".txt" == name) and np.array_equal(self.path, self.real_paths[i]):
                    print(f"[INFO] Path executado: {name}")
                    break
        elif hasattr(self, 'path') and isinstance(self.path, np.ndarray):
            print(f"[INFO] Path executado: Fictício ({'straight' if np.all(self.path[:,1]==self.path[0,1]) else 'custom'})")
        '''
        Reset the environment to the initial state.
        '''
        if seed is not None:
            np.random.seed(seed)

        self.path = self._switch_path()

        # Filtragem dos goals para paths reais
        if self.path_mode == "real" or (self.path_mode == "ambos" and self.path in self.real_paths):
            self.filtered_goals = self._filter_goals_by_distance(self.path, self.min_goal_separation)
        else:
            self.filtered_goals = self.path

        self.agent_yaw = 0.0
        self.desired_yaw_angle = 0.0
        self.tick = 0
        self.linear_velocity = self.min_linear_velocity
        self.yaw_angle_error = 0.0
        self.angular_velocity = 0.0
        self.current_goal_index = 0
        self.current_goal_distance = 0.0
        self.current_position = np.array([-0.3, 0.0])
        self.is_subgoal_reached = False
        self.goal_reached_counter = 0
        self.subgoal_reacher_counter = 0
        self.is_goal_reached = False
        self.is_terminated = False
        self.is_truncated = False
        self.distances = np.zeros(self.num_goals_window)

        self.current_goal_position = self.filtered_goals[0]
        self.current_goals_window_position = self.filtered_goals[1:1+self.num_goals_window]

        print(f"RESET: Path selecionado tem {len(self.path)} pontos.")
        print(f"RESET: Path filtrado tem {len(self.filtered_goals)} goals.")
        print(f"RESET: Primeiro goal: {self.current_goal_position}, Último goal: {self.filtered_goals[-1]}")

        observation = self._get_obs()
        info = self._get_info()
        return observation

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
        
        # ...existing code...
        path_x, path_y = zip(*self.path)
        self.path_marker.set_data(path_x, path_y)

        # Show current goal only if not terminated and not goal reached
        if not self.is_terminated and not self.is_goal_reached:
            goal_x, goal_y = self.current_goal_position
            self.current_goal_marker.set_data([goal_x], [goal_y])
        else:
            self.current_goal_marker.set_data([], [])

        # Show current agent position
        agent_x, agent_y = self.current_position
        self.agent_marker.set_data([agent_x], [agent_y])

        front_x = agent_x + 0.5 * np.cos(self.agent_yaw)
        front_y = agent_y + 0.5 * np.sin(self.agent_yaw)
        self.agent_front.set_data([agent_x, front_x], [agent_y, front_y])

        desired_yaw_x = agent_x + 0.5 * np.cos(self.desired_yaw_angle + self.agent_yaw)
        desired_yaw_y = agent_y + 0.5 * np.sin(self.desired_yaw_angle + self.agent_yaw)
        self.desired_yaw.set_data([agent_x, desired_yaw_x], [agent_y, desired_yaw_y])

        # Show goals window only if not terminated
        if not self.is_terminated and self.current_goals_window_position.size > 0:
            multi_goals_x, multi_goals_y = zip(*self.current_goals_window_position)
            self.current_goals_window_marker.set_data(multi_goals_x, multi_goals_y)
        else:
            self.current_goals_window_marker.set_data([], [])

        self.distance_title.set_text(f'Goal distance: {self.current_goal_distance:.2f}m | Current tick: {self.tick}')

        # Ajusta a janela do gráfico para seguir o agente
        margin_x = 5
        margin_y = 2.5
        self.ax.set_xlim(agent_x - margin_x, agent_x + margin_x)
        self.ax.set_ylim(agent_y - margin_y, agent_y + margin_y)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Reinicia automaticamente o episódio se terminado
        if self.is_terminated:
            self.reset()
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

        reward_distance = -self.current_goal_distance * 0.1
        
        if self.is_goal_reached:
            reward_goal_reached = 10.0
        if self.is_subgoal_reached:
            reward_subgoal_reached = 5.0
        if self.is_terminated:
            sucess_reward = 100.0
        if self.is_truncated:
            truncated_reward = -100.0
        rewards = reward_goal_reached + sucess_reward + reward_subgoal_reached + reward_distance + truncated_reward

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

        goals = self.filtered_goals if hasattr(self, 'filtered_goals') else self.path

        if self.current_goal_distance < self.goal_threshold:
            if self.current_goal_index >= len(goals) - 1:
                # Último objetivo atingido, termina episódio
                self.current_goal_position = goals[-1]
                self.is_goal_reached = True
                self.is_terminated = True
                self.current_goals_window_position = np.array([])
                self.distances = np.zeros(self.num_goals_window)
                return
            else:
                self.current_goal_index += 1
                self.goal_reached_counter += 1
                self.current_goal_position = goals[self.current_goal_index]
                self.current_goals_window_position = goals[self.current_goal_index+1:self.current_goal_index+1+self.num_goals_window]
                self.is_goal_reached = True
                return
        self._update_subgoal()
        self.is_goal_reached = False
    
    def _update_subgoal(self):
        '''
        Update the minor goal for the agent.
        '''
        # if np.any(self.distances < self.goal_threshold):
        #     self.subgoal_reacher_counter += 1
        #     closest_goal_index = np.argmin(self.distances)
        #     self.current_goal_index = closest_goal_index + self.current_goal_index + 1
        #     self.current_goal_position = self.path[self.current_goal_index]
        #     self._generate_goals_window()
        #     self.is_subgoal_reached = True
        #     return
        # self.is_subgoal_reached = False

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
        # Só permite terminar se o agente percorreu pelo menos 80% dos goals
        goals = self.filtered_goals if hasattr(self, 'filtered_goals') else self.path
        min_goals_required = int(len(goals) * 0.8)
        
        if self.current_goal_index < min_goals_required:
            # Agente ainda não percorreu o suficiente do path
            self.is_terminated = False
            return
            
        # Verifica se chegou ao último goal filtrado
        distance_to_final = self._get_distance(self.current_position, goals[-1])
        
        if distance_to_final < self.goal_threshold:
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

        if timeout or out_of_bound:
            self.is_truncated = True

    def _switch_path(self) -> np.ndarray:
        '''Escolhe aleatoriamente entre real, fictício ou ambos, ou seleciona pelo nome'''
        options = []
        if self.path_mode == "real":
            options = ["real"]
        elif self.path_mode == "ficticio":
            options = ["ficticio"]
        elif self.path_mode == "ambos":
            options = ["real", "ficticio"]
        else:
            options = ["ficticio"]

        # Se nome de path foi especificado, seleciona esse EXATAMENTE independentemente do modo
        if self.selected_path_name is not None and self.real_paths:
            for i, name in enumerate(self.real_paths_names):
                # Garante correspondência exata, ignorando extensão se necessário
                if self.selected_path_name == name or self.selected_path_name + ".txt" == name:
                    print(f"Selecionando path específico: {name}")
                    return self.real_paths[i]
            print(f"Path '{self.selected_path_name}' não encontrado, usando aleatório.")

        chosen = np.random.choice(options)
        if chosen == "real" and self.real_paths:
            # Escolhe um dos caminhos reais
            return self.real_paths[np.random.randint(len(self.real_paths))]
        else:
            # Escolhe um dos fictícios
            path_type = np.random.choice(['straight', 'sine', 'zigzag'])
            return self._create_path(path_type)

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
        goals = self.filtered_goals if hasattr(self, 'filtered_goals') else self.path
        self.current_goals_window_position = goals[self.current_goal_index + 1:self.current_goal_index + 1 + self.num_goals_window]

    def _filter_goals_by_distance(self, path, min_dist):
        '''Retorna lista de goals filtrados por distância mínima entre eles'''
        filtered = [path[0]]
        last = path[0]
        for pt in path[1:]:
            if np.linalg.norm(np.array(pt) - np.array(last)) >= min_dist:
                filtered.append(pt)
                last = pt
        # Só adiciona o último ponto se for realmente distinto do inicial
        if not np.allclose(filtered[-1], path[-1]) and not np.allclose(path[0], path[-1]):
            filtered.append(path[-1])
        return np.array(filtered)