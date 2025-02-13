import torch
import numpy as np
import torch.autograd
import torch.nn as nn
import torch.optim as optim

from models import *
from utils import Memory
from ddpg import Actor, Critic

'''
This class implements a DDPG agent.
'''

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, buffer_size=100000, 
                 actor_learning_rate=1e-4, critic_learning_rate=1e-3):
        
        # Parameters
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(buffer_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_learning_rate)
    
    # def get_action(self, state):
    #     '''
    #     This function is responsible for getting the action from the actor network.
    #     '''
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     print('state: ', state)
    #     action = self.actor.forward(state)
    #     # action = self.actor(state).cpu().detach().numpy().flatten()
    #     action = action.detach().numpy()[0,0]
    #     print("action: ", action)
    #     return action

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor.forward(state).detach().numpy()
        
        if action.shape[0] == 1:  # Caso seja um vetor 1D
            action = action.reshape(-1, 2)  # Converte para (num_agents, 2)

        return action
    
    def update(self, batch_size):

        '''
        This function is responsible for updating the networks.
        '''

        # Get a batch of experiences
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)

        # Convert the experiences to PyTorch tensors
        # for training with the networks
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)
        
        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_model(self, filename="ddpg_model.pth"):
        '''
        Saves the model's weights to a file.
        '''
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="ddpg_model.pth"):
        '''
        Loads the model's weights from a file.
        '''
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor.eval()
        print(f"Model loaded from {filename}")