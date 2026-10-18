"""
SSTF-XMF Configuration Module
==============================
Centralised configuration for the Self-Supervised Temporal Transformer
with Explainable Multi-Modal Fusion framework.

Author: MSc Dissertation Research
Licence: Academic Research Use
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Architecture configuration for the SSTF-XMF model."""

    # Core transformer dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    intra_visit_layers: int = 2
    inter_visit_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Sequence length constraints
    max_visits: int = 512
    max_events_per_visit: int = 64
    max_medications_per_visit: int = 20
    max_text_tokens: int = 512

    # Modality vocabularies
    num_lab_features: int = 128
    num_vital_features: int = 32
    num_diagnosis_codes: int = 1800  # CCSR categories
    num_procedure_codes: int = 500
    num_medication_codes: int = 600  # ATC level-4
    imaging_feature_dim: int = 24

    # Text encoder
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_embed_dim: int = 768
    text_trainable_layers: int = 2

    # Time encoding
    time_num_frequencies: int = 64
    time_max_hours: float = 87600.0  # 10 years in hours

    # Concept bottleneck
    num_concepts: int = 15
    concept_names: List[str] = field(default_factory=lambda: [
        "sbp_trajectory", "dbp_trajectory", "ldl_control",
        "hdl_level", "hba1c_control", "bmi_trajectory",
        "smoking_exposure", "physical_activity", "med_adherence",
        "lv_function", "renal_trajectory", "inflammatory_status",
        "age_risk", "family_history_risk", "psychosocial_risk"
    ])

    # Multi-modal fusion
    num_modalities: int = 4
    modality_names: List[str] = field(default_factory=lambda: [
        "structured", "text", "medication", "imaging"
    ])

    # Prediction
    prediction_horizon_years: float = 5.0
    num_time_bins: int = 20  # quarterly bins over 5 years


@dataclass
class PreTrainingConfig:
    """Configuration for self-supervised pre-training."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    gradient_clip: float = 1.0

    # Masked Clinical Event Modelling
    mcem_enabled: bool = True
    mask_ratio: float = 0.15
    mask_random_ratio: float = 0.1
    mask_keep_ratio: float = 0.1

    # Temporal Contrastive Learning
    tcl_enabled: bool = True
    tcl_temperature: float = 0.07
    tcl_weight: float = 0.5
    tcl_window_hours: float = 720.0  # 30 days
    tcl_hard_negative_ratio: float = 0.3

    # Checkpointing
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 10


@dataclass
class FineTuningConfig:
    """Configuration for supervised fine-tuning."""

    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Loss weighting
    survival_loss_weight: float = 1.0
    calibration_loss_weight: float = 0.1
    concept_consistency_weight: float = 0.05
    fairness_lambda: float = 0.1

    # Fairness groups
    fairness_attributes: List[str] = field(default_factory=lambda: [
        "age_group", "sex", "ethnicity", "deprivation_quintile"
    ])

    # Early stopping
    early_stopping_metric: str = "val_auroc"
    early_stopping_patience: int = 8


@dataclass
class DataConfig:
    """Configuration for data loading and pre-processing."""

    mimic_iv_path: str = "./data/mimic-iv"
    eicu_path: str = "./data/eicu"
    ukbiobank_path: str = "./data/ukbiobank"

    # Splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Cohort criteria
    min_age: int = 40
    min_encounters: int = 2
    min_followup_days: int = 365
    min_events_per_patient: int = 10

    # Pre-processing
    outlier_clip_quantile: float = 0.001
    missing_indicator: bool = True
    feature_scaler: str = "robust"  # options: robust, standard, minmax

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class DeploymentConfig:
    """Configuration for clinical deployment."""

    # FHIR
    fhir_base_url: str = "https://fhir.example.nhs.uk/R4"
    fhir_auth_token: Optional[str] = None

    # Inference
    onnx_export_path: str = "./artefacts/sstf_xmf.onnx"
    inference_quantisation: str = "int8"  # options: fp32, fp16, int8
    inference_batch_size: int = 32
    max_latency_ms: int = 2000

    # Monitoring
    psi_threshold: float = 0.2
    ks_threshold: float = 0.1
    fairness_auroc_threshold: float = 0.05
    monitoring_interval_hours: int = 24

    # Regulatory
    audit_log_path: str = "./logs/audit.log"
    model_version: str = "1.0.0"
    uk_mhra_classification: str = "Class IIa SaMD"
    eu_ai_act_classification: str = "High-risk AI system"


@dataclass
class SSTFConfig:
    """Master configuration combining all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PreTrainingConfig = field(default_factory=PreTrainingConfig)
    finetune: FineTuningConfig = field(default_factory=FineTuningConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    # Global settings
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    log_level: str = "INFO"
    experiment_name: str = "sstf_xmf_v1"
    output_dir: str = "./outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "SSTFConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            pretrain=PreTrainingConfig(**data.get("pretrain", {})),
            finetune=FineTuningConfig(**data.get("finetune", {})),
            data=DataConfig(**data.get("data", {})),
            deployment=DeploymentConfig(**data.get("deployment", {})),
            **{k: v for k, v in data.items()
               if k not in ["model", "pretrain", "finetune", "data", "deployment"]}
        )

    def to_yaml(self, path: str) -> None:
        """Serialise configuration to a YAML file."""
        from dataclasses import asdict
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False)


def get_default_config() -> SSTFConfig:
    """Return a default configuration instance."""
    return SSTFConfig()
