"""Configurações centralizadas do sistema de manutenção preditiva."""
from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class WeibullConfig:
    """Parâmetros da distribuição Weibull."""
    beta: float = 2.5          # Parâmetro de forma (β > 1 = desgaste)
    eta: float = 1000.0        # Parâmetro de escala (vida característica em horas)
    beta_uncertainty: float = 0.1   # Incerteza no β (para simulação estocástica)
    eta_uncertainty: float = 0.05   # Incerteza no η
    
    def sample_parameters(self) -> tuple[float, float]:
        """Amostra parâmetros com incerteza para robustez."""
        beta_sampled = np.random.normal(self.beta, self.beta * self.beta_uncertainty)
        eta_sampled = np.random.normal(self.eta, self.eta * self.eta_uncertainty)
        return max(0.5, beta_sampled), max(100, eta_sampled)

@dataclass
class CostConfig:
    """Custos de manutenção e operação."""
    preventive: float = 1000.0      # Custo de manutenção preventiva (R$)
    corrective: float = 10000.0     # Custo de manutenção corretiva/falha (R$)
    downtime_per_hour: float = 500.0  # Custo por hora de parada (R$)
    operation_value: float = 100.0   # Valor gerado por hora de operação (R$)
    
    # Tempos associados (horas)
    preventive_downtime: float = 4.0
    corrective_downtime: float = 24.0
    
    @property
    def total_preventive_cost(self) -> float:
        return self.preventive + (self.preventive_downtime * self.downtime_per_hour)
    
    @property
    def total_corrective_cost(self) -> float:
        return self.corrective + (self.corrective_downtime * self.downtime_per_hour)
    
    @property
    def cost_ratio(self) -> float:
        """Razão entre custos (Corretiva / Preventiva)."""
        return self.total_corrective_cost / self.total_preventive_cost

@dataclass
class EnvironmentConfig:
    """Configurações do ambiente de simulação."""
    max_age: float = 2000.0         # Idade máxima em horas
    time_step: float = 10.0          # Passo de tempo em horas
    episode_max_steps: int = 3000   # Máximo de passos por episódio

@dataclass
class DDQNConfig:
    """Hiperparâmetros do DDQN."""
    # Arquitetura
    state_dim: int = 6
    action_dim: int = 2             # [0=Operar, 1=Manter]
    hidden_layers: list = field(default_factory=lambda: [128, 128, 64])
    
    # Treinamento
    learning_rate: float = 1e-4
    gamma: float = 0.99             # Fator de desconto
    tau: float = 0.005              # Taxa de atualização soft do target
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Exploração
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    
    # Regularização
    grad_clip: float = 1.0
    weight_decay: float = 1e-5
    
    # Configs Extras
    target_update_freq: int = 100
    use_soft_update: bool = True

@dataclass
class TrainingConfig:
    """Configurações de treinamento."""
    n_episodes: int = 1000
    warmup_episodes: int = 50
    eval_frequency: int = 50
    n_eval_episodes: int = 20
    save_frequency: int = 100
    early_stopping_patience: int = 100
    log_frequency: int = 10

@dataclass
class Config:
    """Configuração Principal."""
    weibull: WeibullConfig = field(default_factory=WeibullConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ddqn: DDQNConfig = field(default_factory=DDQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    seed: int = 42
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"