import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from agent.models import Actor, Critic
from utils.utils import ReplayBuffer
from agent.ddpg import Actor, Critic

'''
This class implements a DDPG agent.
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, observation_dim, action_dim, max_action, gamma=0.99, tau=0.005, buffer_size=100000, 
                 actor_learning_rate=1e-4, critic_learning_rate=1e-3):
        
        # Parameters
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(observation_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(observation_dim, action_dim, max_action).to(device)
        
        self.critic = Critic(observation_dim, action_dim).to(device)
        self.critic_target = Critic(observation_dim, action_dim).to(device)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = ReplayBuffer(buffer_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_learning_rate)

    def get_action(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)
        # action = self.actor.forward(observation).detach().numpy()[0]
        action = self.actor.forward(observation).detach().cpu().numpy()[0]

        return action
    
    def update(self, batch_size):

        '''
        This function is responsible for updating the networks.

        @param batch_size: The number of samples to be used in each update.
        '''

        # Get a batch of experiences
        observations, actions, rewards, next_observations, terminateds, truncateds, infos = self.memory.sample(batch_size)

        # Convert the experiences to PyTorch tensors
        # for training with the networks
        observations = torch.FloatTensor(observations).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_observations = torch.FloatTensor(next_observations).to(device)
        # terminateds = torch.FloatTensor(terminateds).to(device)
        # truncateds = torch.FloatTensor(truncateds).to(device)
        # infos = torch.FloatTensor(infos).to(device)
        dones = torch.FloatTensor(np.logical_or(terminateds, truncateds)).unsqueeze(1).to(device)

        # Critic loss
        Qvals = self.critic.forward(observations, actions)
        next_actions = self.actor_target.forward(next_observations)
        next_Q = self.critic_target.forward(next_observations, next_actions.detach())
        # Qprime = rewards + self.gamma * next_Q
        Qprime = rewards + (1 - dones) * self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)
        
        # Actor loss
        policy_loss = -self.critic.forward(observations, self.actor.forward(observations)).mean()
        
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
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="ddpg_model.pth"):
        checkpoint = torch.load(filename, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Model loaded from {filename}")