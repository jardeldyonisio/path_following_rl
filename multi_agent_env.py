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

    def reset(self):
        self.positions = np.zeros((self.num_agents, 2))
        self.orientations = np.zeros(self.num_agents)
        self.path = self.generate_path()
        self.time = 0
        self.active_agents = np.ones(self.num_agents, dtype=bool)
        return self.get_state(), {}

    def get_state(self):
        distances, angles = self.calculate_path_metrics()
        return np.column_stack((self.positions, distances, angles))

    def step(self, actions):
        self.update_positions(actions)
        distances, angles = self.calculate_path_metrics()
        
        # Extraindo velocidades das ações para passar para a função de recompensa
        speeds = actions[:, 1]  
        rewards = self.calculate_rewards(distances, angles, speeds)
        
        self.active_agents = self.active_agents & (distances <= 5.0)

        next_state = self.get_state()
        for i in range(self.num_agents):
            self.replay_buffer.append((self.get_state(), actions[i], rewards[i], next_state[i]))
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.time += 1
        done = self.time > 500 or np.sum(self.active_agents) == 0
        return next_state, rewards, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            # Create the plot
            self.fig, self.ax = plt.subplots()

            # Create agent markers
            self.path_plot, = self.ax.plot([], [], label='Path')

            # Create front indicators (small lines)
            self.agent_plots = [self.ax.plot([], [], 'o', label=f'Agent {i}')[0] for i in range(self.num_agents)]
            self.agent_fronts = [self.ax.plot([], [], '-', color='red')[0] for _ in range(self.num_agents)]
            
            # Set plot limits
            self.ax.set_xlim(-10, 110)
            self.ax.set_ylim(-10, 10)
            self.ax.legend()
            plt.ion()
            plt.show(block=False)

        # Update path
        path_x, path_y = zip(*self.path)
        self.path_plot.set_data(path_x, path_y)

        # Update agent positions and front indicators
        for i, agent_plot in enumerate(self.agent_plots):
            if self.active_agents[i]:
                x, y = self.positions[i]
                agent_plot.set_data(x, y)  # Update agent position
                
                # Compute front indicator position (short line in the direction of orientation)
                front_x = x + 0.5 * np.cos(self.orientations[i])
                front_y = y + 0.5 * np.sin(self.orientations[i])
                self.agent_fronts[i].set_data([x, front_x], [y, front_y])  # Draw front line

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_positions(self, actions):
        actions = actions.reshape((self.num_agents, 2))  # Certifica-se de que as ações têm a forma correta
        steerings, speeds = actions[:, 0], actions[:, 1]
        self.orientations += steerings * 0.1
        dx = speeds * np.cos(self.orientations)
        dy = speeds * np.sin(self.orientations)
        self.positions += np.column_stack((dx, dy))

    def calculate_path_metrics(self):
        distances = np.linalg.norm(self.path[:, None, :] - self.positions[None, :, :], axis=2)
        closest_indices = np.argmin(distances, axis=0)
        closest_points = self.path[closest_indices]
        
        distance_to_path = distances[closest_indices, np.arange(self.num_agents)]
        path_directions = np.arctan2(closest_points[:, 1] - self.positions[:, 1],
                                     closest_points[:, 0] - self.positions[:, 0])
        angle_to_path = path_directions - self.orientations
        
        return distance_to_path, angle_to_path

    def calculate_rewards(self, distances, angles, speeds):
        """
        Calcula a recompensa com base na distância do caminho, ângulo de orientação,
        velocidade do robô e chegada ao final do caminho.
        """
        reward = -distances - 0.01 * (angles ** 2)

        # The variable self.active_agents is a boolean array that keeps track 
        # of which agents are still active, if they are not, it's beacause they
        # are either too far from the path or they have reached the end of the path
        reward[~self.active_agents] -= 100.0  

        # Bônus por estar muito próximo do caminho
        reward[distances < 0.1] += 1.0  

        # Penalizar se a velocidade for muito baixa (evita que o robô fique parado)
        reward[speeds < 0.05] -= 5.0  

        # 🎯 Alta recompensa para agentes que chegaram ao fim do caminho
        final_position = self.path[-1]  # Última posição do caminho
        reached_goal = np.linalg.norm(self.positions - final_position, axis=1) < 0.2  # Verifica se o agente chegou perto do fim
        
        reward[reached_goal] += 100.0  # Alta recompensa para incentivar alcançar o objetivo

        # reward = np.maximum(reward, 0.0)

        return reward

    def generate_path(self):
        return np.array([[i, 0.0] for i in range(100)])

    def sample_from_replay_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) >= batch_size else []
