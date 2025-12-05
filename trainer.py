import numpy as np
from typing import Dict, List
from config import Config
from environment import MaintenanceEnvironment
from agent import DDQNAgent

class DDQNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.env = MaintenanceEnvironment(config)
        self.eval_env = MaintenanceEnvironment(config, stochastic_params=False)
        self.agent = DDQNAgent(config.ddqn, device=config.device)
        
    def train(self) -> Dict:
        history = {'rewards': [], 'failures': [], 'preventive': [], 'costs': []}
        print(f"Iniciando treinamento em {self.config.device}...")
        
        for ep in range(self.config.training.n_episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                self.agent.buffer.push(state, action, reward, next_state, done)
                self.agent.train_step_fn()
                state = next_state
            
            # Log
            stats = self.env.get_episode_stats()
            history['rewards'].append(stats['total_reward'])
            history['failures'].append(stats['n_failures'])
            history['preventive'].append(stats['n_preventive'])
            history['costs'].append(stats['total_cost'])
            
            if ep % self.config.training.log_frequency == 0:
                print(f"Ep {ep} | R: {stats['total_reward']:.2f} | Fail: {stats['n_failures']} | Prev: {stats['n_preventive']} | Eps: {self.agent.epsilon:.3f}")
                
        return history

    def test_policy(self, n_episodes=50) -> Dict:
        costs = []
        for _ in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, _, term, trunc, _ = self.eval_env.step(action)
                done = term or trunc
            costs.append(self.eval_env.get_episode_stats()['total_cost'])
        return {'costs': costs}

class BaselineComparison:
    def __init__(self, config: Config):
        self.env = MaintenanceEnvironment(config, stochastic_params=False)
        
    def run_fixed_interval(self, interval: int, n_episodes=50):
        costs = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            age = 0
            done = False
            while not done:
                action = 1 if age >= interval else 0
                if action == 1: age = 0
                else: age += 1
                state, _, term, trunc, _ = self.env.step(action)
                done = term or trunc
            costs.append(self.env.get_episode_stats()['total_cost'])
        return {'costs': costs}