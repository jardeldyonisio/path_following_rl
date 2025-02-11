import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This code implements the Actor and Critic networks for the DDPG algorithm.
'''

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size = 256):
        super(Actor, self).__init__()

        '''
        If is necessary, you can change the number of neurons in the layers.
        But, if they are too small, the network may not learn well and if 
        they are too large, it may be slow and have overfitting.
        '''

        # Input layer
        self.fc1 = nn.Linear(state_dim, hidden_size)

        # Hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Output layer
        self.fc3 = nn.Linear(hidden_size, action_dim)

        # Maximum action value, in this case, the maximum velocity
        self.max_action = max_action

    def forward(self, state):
        '''
        This funcition is responsible for the forward pass of the network.
        Using PyTorch every network must have this function implemented.
        This function describes how the input data will pass through the network.
        '''
        # Passing the state through the first layer applying the ReLU 
        # activation function
        x = F.relu(self.fc1(state))

        # Passing the output of the first layer through the second layer
        x = F.relu(self.fc2(x))

        # Passing the output of the second layer through the output layer
        action = torch.tanh(self.fc3(x)) * self.max_action

        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size = 256):
        super(Critic, self).__init__()
        
        '''
        Critic network, this network is responsible for estimating the Q-value
        of the state-action pair.
        ''' 

        # First layer linear transformation
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        '''
        This funcition is responsible for the forward pass of the network.
        Using PyTorch every network must have this function implemented.
        This function describes how the input data will pass through the network.
        '''

        # print("state: ", state)
        # print("action: ", action)
        # Concatenating state and action
        sa = torch.cat([state, action], dim=1)
        
        # Passing the state through the first layer applying the ReLU 
        # activation function
        x = F.relu(self.fc1(sa))

        # Passing the output of the first layer through the second layer
        x = F.relu(self.fc2(x))

        # Passing the output of the second layer through the output layer
        x = self.fc3(x)
        
        return x
