import numpy as np
from scipy.special import gamma as gamma_func
from dataclasses import dataclass
from typing import Tuple

@dataclass
class WeibullMetrics:
    mttf: float              # Mean Time To Failure
    median_life: float       # Vida mediana
    b10_life: float          # Vida B10 (10% de falhas)
    characteristic_life: float  # Eta

class WeibullReliabilityModel:
    """Modelo de confiabilidade Weibull."""
    
    def __init__(self, beta: float, eta: float, t0: float = 0.0):
        self.beta = beta
        self.eta = eta
        self.t0 = t0
        
    def pdf(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t).astype(float)
        t_adj = np.maximum(t - self.t0, 1e-10)
        return (self.beta / self.eta) * (t_adj / self.eta) ** (self.beta - 1) * \
               np.exp(-(t_adj / self.eta) ** self.beta)
    
    def cdf(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t).astype(float)
        t_adj = np.maximum(t - self.t0, 0)
        return 1 - np.exp(-(t_adj / self.eta) ** self.beta)
    
    def reliability(self, t: np.ndarray) -> np.ndarray:
        return 1 - self.cdf(t)
    
    def hazard_rate(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t).astype(float)
        t_adj = np.maximum(t - self.t0, 1e-10)
        return (self.beta / self.eta) * (t_adj / self.eta) ** (self.beta - 1)
    
    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t).astype(float)
        t_adj = np.maximum(t - self.t0, 0)
        return (t_adj / self.eta) ** self.beta

    def sample_failure_time(self, n_samples: int = 1) -> np.ndarray:
        u = np.random.uniform(0, 1, n_samples)
        return self.t0 + self.eta * (-np.log(1 - u)) ** (1 / self.beta)
    
    def compute_metrics(self) -> WeibullMetrics:
        mttf = self.eta * gamma_func(1 + 1/self.beta)
        median_life = self.eta * (np.log(2)) ** (1/self.beta)
        b10_life = self.eta * (np.log(1/0.9)) ** (1/self.beta)
        return WeibullMetrics(mttf, median_life, b10_life, self.eta)

class AdaptiveWeibullModel(WeibullReliabilityModel):
    """Modelo com variabilidade para simulação estocástica."""
    def __init__(self, beta: float, eta: float, beta_std: float = 0.0, eta_std: float = 0.0):
        super().__init__(beta, eta)
        self.beta_std = beta_std
        self.eta_std = eta_std
        
    def update_from_failure(self, failure_time: float):
        """Simplificação de atualização baseada em histórico."""
        # Em uma implementação real completa, usaríamos inferência Bayesiana aqui.
        pass