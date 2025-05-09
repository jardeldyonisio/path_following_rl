import random
import numpy as np

from typing import Dict, Any
from utils.base import RawExplorationStrategy

'''
This class implements a replay buffer for the DDPG algorithm.
'''

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(RawExplorationStrategy):
    '''
    @brief: This class implements the Ornstein-Ulhenbeck process for exploration noise.
    '''
    def __init__(self, 
                 action_space, 
                 mu=0.0, 
                 theta=0.15, 
                 max_sigma=0.05, 
                 min_sigma=None, 
                 decay_period=100000):
        '''
        @param action_space: The action space of the environment.
        @param mu: The mean of the noise process.
        @param theta: The speed of mean reversion.
        @param max_sigma: The maximum standard deviation of the noise process.
        @param min_sigma: The minimum standard deviation of the noise process.
        @param decay_period: The period over which the noise decays.
        '''
        if min_sigma is None:
            min_sigma = max_sigma
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self._max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self.dim = np.prod(action_space.low.shape)
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.ones(self.dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action_from_raw_action(self, action, t=0, **kwargs):
        ou_state = self.evolve_state()
        self.sigma = (
            self._max_sigma
            - (self._max_sigma - self._min_sigma)
            * min(1.0, t * 1.0 / self._decay_period)
        )
        return np.clip(action + ou_state, self.low, self.high)
class GaussianStrategy(RawExplorationStrategy):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.

    Based on the rllab implementation.
    """
    def __init__(self, action_space, max_sigma=1.0, min_sigma=None,
                 decay_period=1000000):
        assert len(action_space.shape) == 1
        self._max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        sigma = (
            self._max_sigma - (self._max_sigma - self._min_sigma) *
            min(1.0, t * 1.0 / self._decay_period)
        )
        return np.clip(
            action + np.random.normal(size=len(action)) * sigma,
            self._action_space.low,
            self._action_space.high,
        )

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, observation: np.ndarray, action: np.ndarray, reward: float, 
            next_observation: np.ndarray, terminated: bool, truncated: bool, info: Dict[str, Any]):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((observation, action, reward, next_observation, terminated, truncated, info))
    
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError("The buffer does not have enough experiences for the requested batch size.")
        
        batch = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations, terminateds, truncateds, infos = zip(*batch)
        
        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            np.array(next_observations),
            np.array(terminateds),
            np.array(truncateds),
            infos  # List of dictionaries (not converted to array)
        )
    
    def __len__(self):
        return len(self.buffer)