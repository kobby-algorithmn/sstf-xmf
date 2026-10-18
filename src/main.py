"""
SSTF-XMF Main Entry Point
==========================
Command-line interface for training, evaluation, and deployment.

Usage:
    python -m src.main pretrain --config configs/default.yaml
    python -m src.main finetune --config configs/default.yaml --pretrained <path>
    python -m src.main evaluate --config configs/default.yaml --checkpoint <path>
    python -m src.main predict --config configs/default.yaml --patient-id <id>
"""

from __future__ import annotations
import argparse
import logging
import sys
import random
import numpy as np
import torch

from .config import SSTFConfig, get_default_config
from .models.sstf_xmf import SSTFXMF
from .training.trainer import SSTFXMFTrainer
from .evaluation.evaluator import ClinicalEvaluator


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> SSTFConfig:
    if path:
        return SSTFConfig.from_yaml(path)
    return get_default_config()


def cmd_pretrain(args):
    """Run self-supervised pre-training."""
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    set_seed(config.seed)

    logger.info("=" * 60)
    logger.info("SSTF-XMF Self-Supervised Pre-Training")
    logger.info("=" * 60)

    # Build model
    model = SSTFXMF(config)
    param_counts = model.get_num_parameters()
    logger.info(f"Model parameters: {param_counts}")

    # Build dataloaders (requires real data setup)
    logger.info("Data pipeline must be configured for your specific datasets")
    logger.info("See scripts/prepare_data.py for dataset preparation")

    trainer = SSTFXMFTrainer(model, config)
    # In production: trainer.pretrain(train_loader, val_loader)
    logger.info("Pre-training infrastructure ready")


def cmd_finetune(args):
    """Run supervised fine-tuning."""
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    set_seed(config.seed)

    logger.info("=" * 60)
    logger.info("SSTF-XMF Fairness-Aware Fine-Tuning")
    logger.info("=" * 60)

    model = SSTFXMF(config)

    if args.pretrained:
        logger.info(f"Loading pre-trained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    trainer = SSTFXMFTrainer(model, config)
    logger.info("Fine-tuning infrastructure ready")


def cmd_evaluate(args):
    """Evaluate a trained model."""
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    set_seed(config.seed)

    logger.info("=" * 60)
    logger.info("SSTF-XMF Comprehensive Evaluation")
    logger.info("=" * 60)

    model = SSTFXMF(config)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = ClinicalEvaluator(model, config)
    logger.info("Evaluator ready for clinical validation")


def cmd_predict(args):
    """Generate prediction for a single patient via FHIR."""
    from .deployment.service import InferenceService, FHIRClient

    logger = logging.getLogger(__name__)
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("SSTF-XMF Clinical Prediction")
    logger.info("=" * 60)

    model = SSTFXMF(config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    fhir_client = FHIRClient(config)
    inference_service = InferenceService(model, config)

    logger.info(f"Fetching FHIR data for patient {args.patient_id}")
    fhir_data = fhir_client.fetch_patient_data(args.patient_id)
    logger.info("Transform FHIR data and run inference in production deployment")


def main():
    parser = argparse.ArgumentParser(
        description="SSTF-XMF: Cardiovascular Risk Prediction Framework"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Pre-train command
    pt = subparsers.add_parser("pretrain", help="Self-supervised pre-training")
    pt.add_argument("--config", type=str, help="Path to YAML config")
    pt.set_defaults(func=cmd_pretrain)

    # Fine-tune command
    ft = subparsers.add_parser("finetune", help="Supervised fine-tuning")
    ft.add_argument("--config", type=str, help="Path to YAML config")
    ft.add_argument("--pretrained", type=str, help="Path to pre-trained checkpoint")
    ft.set_defaults(func=cmd_finetune)

    # Evaluate command
    ev = subparsers.add_parser("evaluate", help="Comprehensive evaluation")
    ev.add_argument("--config", type=str, help="Path to YAML config")
    ev.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    ev.set_defaults(func=cmd_evaluate)

    # Predict command
    pr = subparsers.add_parser("predict", help="Single-patient prediction")
    pr.add_argument("--config", type=str, help="Path to YAML config")
    pr.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    pr.add_argument("--patient-id", type=str, required=True, help="FHIR patient ID")
    pr.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    setup_logging(getattr(args, "log_level", "INFO"))
    args.func(args)


if __name__ == "__main__":
    main()
