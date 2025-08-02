import torch
import numpy as np
from environment.multi_agent_env import MultiAgentPathFollowingEnv
from agent.ddpg import DDPGAgent

# Definir o ambiente
env = MultiAgentPathFollowingEnv(num_agents=1)

# Criar o agente DDPG com os mesmos parâmetros do treinamento
agent = DDPGAgent(state_dim=4, action_dim=2, max_action=1.0)

# Carregar os pesos do modelo treinado
checkpoint = torch.load("models/ddpg_model.pth", map_location=torch.device('cpu'))
agent.actor.load_state_dict(checkpoint['actor'])
agent.critic.load_state_dict(checkpoint['critic'])
agent.actor.eval()  # Coloca a rede em modo de avaliação

# Rodar a simulação
num_episodes = 100  # Número de episódios de teste

for episode in range(num_episodes):
    observation = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.get_action(observation)  # Pegar ação do agente treinado
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        observation = next_observation

        env.render()  # Se houver renderização no ambiente

    print(f"Episode {episode + 1}, Reward: {episode_reward}")

env.close()
