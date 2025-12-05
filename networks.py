import torch
import torch.nn as nn
import numpy as np
from typing import List

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = [128, 128]):
        super().__init__()
        
        # Feature Extraction
        layers = []
        prev_dim = state_dim
        for hidden in hidden_layers[:-1]:
            layers.extend([nn.Linear(prev_dim, hidden), nn.ReLU()])
            prev_dim = hidden
        self.features = nn.Sequential(*layers)
        
        # Dueling Streams
        feature_dim = prev_dim
        stream_dim = hidden_layers[-1]
        
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, stream_dim), nn.ReLU(),
            nn.Linear(stream_dim, 1)
        )
        
        # Advantage Stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, stream_dim), nn.ReLU(),
            nn.Linear(stream_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

def create_network(state_dim: int, action_dim: int, hidden_layers: List[int]) -> nn.Module:
    return DuelingQNetwork(state_dim, action_dim, hidden_layers)