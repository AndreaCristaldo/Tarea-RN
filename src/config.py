from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    seed: int = 42

    input_dim: int = 784
    num_classes: int = 10
    latent_dim: int = 32

    # Etapa 1
    ae_epochs: int = 15
    ae_batch_size: int = 256
    ae_lr: float = 1e-3

    # Etapa 2
    clf_epochs: int = 10
    clf_batch_size: int = 256
    clf_lr: float = 1e-3
    dropout: float = 0.3

    root: Path = Path(".")
    artifacts: Path = root / "artifacts"
    stage1_dir: Path = artifacts / "stage1"
    stage2_dir: Path = artifacts / "stage2"
