import torch
import torch.optim as optim
import numpy as np
import copy
from config import DDQNConfig
from networks import create_network
from replay_buffer import PrioritizedReplayBuffer

class DDQNAgent:
    def __init__(self, config: DDQNConfig, device: str):
        self.config = config
        self.device = device
        
        self.q_network = create_network(config.state_dim, config.action_dim, config.hidden_layers).to(device)
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=config.learning_rate)
        self.buffer = PrioritizedReplayBuffer(config.buffer_size, config.state_dim, device=device)
        
        self.epsilon = config.epsilon_start
        self.train_step = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.config.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_t).argmax(dim=1).item()

    def train_step_fn(self):
        if len(self.buffer) < self.config.batch_size: return None
        self.train_step += 1
        
        # Amostragem (PER)
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.config.batch_size)
        
        # Double DQN Logic
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * self.config.gamma * next_q
            
        curr_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = target - curr_q
        
        # Loss com pesos do PER
        loss = (weights * torch.nn.functional.smooth_l1_loss(curr_q, target, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # Atualiza Prioridades
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Soft Update
        if self.config.use_soft_update:
            for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.copy_(self.config.tau * p.data + (1 - self.config.tau) * tp.data)
                
        # Epsilon Decay
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        return loss.item()

    def get_policy_analysis(self, age_range: np.ndarray) -> dict:
        self.q_network.eval()
        q_ops, q_maint = [], []
        for age in age_range:
            # Estado sintético aproximado
            state = np.array([age/2000, 0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
            with torch.no_grad():
                qs = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                qs = qs.cpu().numpy()[0]
            q_ops.append(qs[0])
            q_maint.append(qs[1])
        self.q_network.train()
        return {'ages': age_range, 'q_operate': np.array(q_ops), 'q_maintain': np.array(q_maint)}
    
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)