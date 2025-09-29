
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from environment.simple_ter_paths import SimpleTerPathFollowingEnv
from agent.ddpg import DDPGAgent

def main():
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Teste do agente DDPG no ambiente de path following')
    parser.add_argument('--episodes', '-e', type=int, default=10, 
                       help='Número de episódios para executar (padrão: 10)')
    parser.add_argument('--path', '-p', type=str, default='path_circular',
                       help='Nome do path a ser usado (padrão: path_circular)')
    args = parser.parse_args()

    # Definir o ambiente
    # Escolha o modo: "real", "ficticio" ou "ambos"

    # Use o mesmo SEED do treinamento
    SEED = 42
    paths_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paths'))
    env = SimpleTerPathFollowingEnv(path_mode="real", paths_folder=paths_folder, selected_path_name=args.path)
    obs_dim = 4 + env.num_goals_window
    agent = DDPGAgent(observation_dim=obs_dim, action_dim=2, max_action=1.0, seed=SEED)

    # Carregar os pesos do modelo treinado
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models/2025-09-29_03-27-42/best_avg_model.pth'))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.actor.eval()  # Coloca a rede em modo de avaliação

    print(f"Iniciando teste com {args.episodes} episódios usando path '{args.path}'...")

    # Rodar a simulação
    for episode in range(args.episodes):
        observation = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(observation)  # Pegar ação do agente treinado
            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

            env.render()  # Se houver renderização no ambiente

        print(f"Episode {episode + 1}/{args.episodes}, Reward: {episode_reward:.2f}")

    print(f"\nTeste concluído! {args.episodes} episódios executados.")
    env.close()

if __name__ == "__main__":
    main()
