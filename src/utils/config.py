from dataclasses import dataclass, field, asdict
from typing import Optional, Iterable


@dataclass
class ProblemConfig:
    dim: Optional[int] = field(default=1)
    N: Optional[int] = field(default=101)
    L: Optional[float] = field(default=20.0)
    beta: Optional[float] = field(default=10.0)
    alpha: Optional[float] = field(default=0.5)
    N_electrons: Optional[int] = field(default=2)

@dataclass
class OptimizerConfig:
    lr: Optional[float] = field(default=0.1)
    max_iter: Optional[int] = field(default=2000)
    tol: Optional[float] = field(default=1e-5)
    poles_iter: Optional[int] = field(default=20)
    eval_iter: Optional[int] = field(default=2)
    update_poles_iter: Optional[int] = field(default=50)
    N_samples: Optional[int] = field(default=10)
    N_eval: Optional[int] = field(default=200)
    N_poles: Optional[int] = field(default=100)
    cheat: Optional[bool] = field(default=False)

@dataclass
class LoggingConfig:
    wandb_logging: bool = True
    wandb_project: str = "MirrorDescent-DFT"
    wandb_path: str = "./assets/wandb"
    raw: Optional[bool] = field(default=True)
    seed: Optional[int] = field(default=224)

@dataclass 
class UtilConfig(LoggingConfig, ProblemConfig, OptimizerConfig):
    device: Optional[str] = field(default='cuda')