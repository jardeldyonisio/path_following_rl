import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from network import Actor, Critic

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=3e-4, buffer_size=100000):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usar GPU se disponível

        # Inicializar as redes e movê-las para o dispositivo
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.device = device  # Armazenar o dispositivo (GPU ou CPU)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Mover o estado para o dispositivo
        action = self.actor(state).cpu().data.numpy().flatten()  # Mover a ação para a CPU para retorno
        action += noise * np.random.randn(*action.shape)
        return np.clip(action, -self.max_action, self.max_action)
    
    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Mover os dados para o dispositivo
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_next, q2_next)
        
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states))[0].mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Atualização das redes alvo com o dispositivo
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
