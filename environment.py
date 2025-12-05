import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import IntEnum
from config import Config
from weibull_model import AdaptiveWeibullModel

class Action(IntEnum):
    OPERATE = 0
    MAINTAIN = 1

@dataclass
class MachineState:
    age: float = 0.0
    n_preventive: int = 0
    n_failures: int = 0
    operational_hours: float = 0.0
    is_failed: bool = False

class MaintenanceEnvironment:
    def __init__(self, config: Config, stochastic_params: bool = True):
        self.config = config
        self.stochastic_params = stochastic_params
        self.machine = MachineState()
        self.step_count = 0
        self.episode_reward = 0.0
        self._init_weibull_model()

    def _init_weibull_model(self):
        if self.stochastic_params:
            beta, eta = self.config.weibull.sample_parameters()
        else:
            beta, eta = self.config.weibull.beta, self.config.weibull.eta
            
        self.weibull = AdaptiveWeibullModel(
            beta=beta, eta=eta,
            beta_std=self.config.weibull.beta * 0.1,
            eta_std=self.config.weibull.eta * 0.05
        )

    def _get_state(self) -> np.ndarray:
        age = self.machine.age
        # Normalização e Features
        age_norm = min(age / self.config.env.max_age, 1.0)
        hazard = float(self.weibull.hazard_rate(np.array([max(age, 1.0)]))[0])
        hazard_norm = np.tanh(hazard / 0.01)
        cum_hazard = float(self.weibull.cumulative_hazard(np.array([max(age, 1.0)]))[0])
        cum_hazard_norm = np.tanh(cum_hazard)
        reliability = float(self.weibull.reliability(np.array([age]))[0])
        cost_ratio_norm = np.tanh(self.config.cost.cost_ratio / 10)
        
        metrics = self.weibull.compute_metrics()
        near_eol = 1.0 if age > metrics.b10_life else 0.0
        
        return np.array([age_norm, hazard_norm, cum_hazard_norm, reliability, cost_ratio_norm, near_eol], dtype=np.float32)

    def _calculate_failure_prob(self, dt: float) -> float:
        current_age = self.machine.age
        R_curr = self.weibull.reliability(np.array([current_age]))[0]
        R_next = self.weibull.reliability(np.array([current_age + dt]))[0]
        if R_curr == 0: return 1.0
        return float(np.clip(1 - (R_next / R_curr), 0, 1))

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed: np.random.seed(seed)
        self._init_weibull_model()
        self.machine = MachineState()
        self.step_count = 0
        self.episode_reward = 0.0
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        dt = self.config.env.time_step
        reward = 0.0
        terminated = False
        info = {}
        
        if action == Action.MAINTAIN:
            reward = -self.config.cost.total_preventive_cost / 1000.0
            self.machine.age = 0.0
            self.machine.n_preventive += 1
            info['event'] = 'preventive'
        else:
            fail_prob = self._calculate_failure_prob(dt)
            failed = np.random.random() < fail_prob
            
            if failed:
                reward = -self.config.cost.total_corrective_cost / 1000.0
                self.machine.is_failed = True
                self.machine.n_failures += 1
                self.machine.age = 0.0
                info['event'] = 'failure'
            else:
                self.machine.age += dt
                self.machine.operational_hours += dt
                reward = (self.config.cost.operation_value * dt) / 10000.0
                info['event'] = 'operate'
        
        self.episode_reward += reward
        truncated = self.step_count >= self.config.env.episode_max_steps
        
        return self._get_state(), reward, terminated, truncated, info

    def get_episode_stats(self) -> Dict:
        return {
            'total_reward': self.episode_reward,
            'n_failures': self.machine.n_failures,
            'n_preventive': self.machine.n_preventive,
            'total_cost': (self.machine.n_preventive * self.config.cost.total_preventive_cost) + 
                          (self.machine.n_failures * self.config.cost.total_corrective_cost),
            'availability': self.machine.operational_hours / max(self.step_count, 1)
        }