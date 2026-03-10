from dataclasses import dataclass
from typing import Optional, List, Tuple
from hydra.core.config_store import ConfigStore


# ===========================
# Data Configuration
# ===========================
@dataclass
class DataConfig:
    """Data loading configuration"""
    batch_size: int = 1
    num_sample_index: int = 0
    num_samples: int = 100
    output: str = "..."
    cle_data_path: str = "..."
    text_path: str = "..."


# ===========================
# Optimization Configuration
# ===========================
@dataclass
class OptimConfig:
    """Optimization parameters"""
    alpha: float = 0.75
    epsilon: int = 16
    steps: int = 200
    # use_sign=True: sign-PGD（高频噪声扰动，视觉上颗粒感而非模糊）
    # use_sign=False: Adam-PGD（默认）
    use_sign: bool = True


# ===========================
# Model Configuration
# ===========================
@dataclass
class ModelConfig:
    """Model-specific parameters"""
    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = False
    crop_scale: Tuple[float, float] = (0.7, 0.95)
    ensemble: bool = False
    device: str = "cuda:2"  # Using GPU 2
    backbone: List[str] = ("L336",)
    full_resolution: bool = False  # 保持原始图片分辨率：DataLoader不resize，特征提取器内部自行缩放，输出与输入尺寸一致


# ===========================
# Attack Configuration
# ===========================
@dataclass
class AttackConfig:
    """Attack configuration"""
    method: str = "pgd"
    vattack: bool = True
    vision_attack: bool = False
    source_text: bool = True
    target_text: bool = True


# ===========================
# Main Configuration
# ===========================
@dataclass
class MainConfig:
    """Main configuration combining all sub-configs"""
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    model: ModelConfig = ModelConfig()
    attack: AttackConfig = AttackConfig()

# ===========================
# Register with Hydra
# ===========================
cs = ConfigStore.instance()
cs.store(name="single_config", node=MainConfig)
