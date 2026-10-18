"""SSTF-XMF: Self-Supervised Temporal Transformer with Explainable Multi-Modal Fusion."""

__version__ = "1.0.0"
__author__ = "MSc Dissertation Research"

from .config import SSTFConfig, get_default_config
from .models.sstf_xmf import SSTFXMF
from .training.trainer import SSTFXMFTrainer
from .evaluation.evaluator import ClinicalEvaluator
from .explainability.engine import ExplainabilityEngine
from .deployment.service import InferenceService, ModelMonitoringService

__all__ = [
    "SSTFConfig",
    "get_default_config",
    "SSTFXMF",
    "SSTFXMFTrainer",
    "ClinicalEvaluator",
    "ExplainabilityEngine",
    "InferenceService",
    "ModelMonitoringService",
]
