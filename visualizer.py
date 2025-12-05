import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from agent import DDQNAgent

class MaintenanceVisualizer:
    def __init__(self):
        plt.style.use('ggplot')

    def plot_training_curves(self, history: Dict, save_path: Optional[str] = None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Rewards
        r = history['rewards']
        window = 50
        avg_r = np.convolve(r, np.ones(window)/window, mode='valid')
        ax1.plot(r, alpha=0.3, label='Raw')
        ax1.plot(avg_r, label='Mov Avg', linewidth=2)
        ax1.set_title('Training Rewards')
        ax1.legend()
        
        # Costs
        c = history['costs']
        avg_c = np.convolve(c, np.ones(window)/window, mode='valid')
        ax2.plot(c, alpha=0.3, color='red')
        ax2.plot(avg_c, color='darkred', linewidth=2)
        ax2.set_title('Total Cost per Episode')
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_policy_structure(self, agent: DDQNAgent, max_age=2000, save_path=None):
        ages = np.linspace(0, max_age, 200)
        res = agent.get_policy_analysis(ages)
        
        plt.figure(figsize=(10, 6))
        plt.plot(res['ages'], res['q_operate'], label='Q(Operate)', color='green')
        plt.plot(res['ages'], res['q_maintain'], label='Q(Maintain)', color='red')
        
        # Find crossing point
        diff = res['q_maintain'] - res['q_operate']
        cross = np.where(diff > 0)[0]
        thresh = res['ages'][cross[0]] if len(cross) > 0 else None
        
        if thresh:
            plt.axvline(thresh, color='black', linestyle='--')
            plt.text(thresh, plt.ylim()[0], f' Threshold: {thresh:.0f}h', rotation=90)
            
        plt.title('Learned Policy Decision Boundary')
        plt.xlabel('Machine Age (hours)')
        plt.ylabel('Q-Value')
        plt.legend()
        
        if save_path: plt.savefig(save_path)
        plt.show()
        return thresh