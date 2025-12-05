import numpy as np
import torch
from typing import Tuple

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.position = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.priorities[self.position] = max_prio
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.LongTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
            indices,
            torch.FloatTensor(weights).to(self.device)
        )
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = (abs(err) + 1e-6)
    
    def __len__(self):
        return self.size