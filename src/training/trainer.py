"""
SSTF-XMF Trainer
=================
Training loops for self-supervised pre-training and fairness-aware
fine-tuning of the SSTF-XMF model.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import SSTFConfig
from ..models.sstf_xmf import SSTFXMF
from .losses import (
    MaskedClinicalEventLoss,
    TemporalContrastiveLoss,
    FairnessAwareSurvivalLoss,
)

logger = logging.getLogger(__name__)


class SSTFXMFTrainer:
    """
    Orchestrator for pre-training and fine-tuning the SSTF-XMF model.
    """

    def __init__(
        self,
        model: SSTFXMF,
        config: SSTFConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device or config.device)
        self.model.to(self.device)

        # Mixed precision
        self.scaler = GradScaler(enabled=config.mixed_precision)

        # Output directories
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Self-supervised pre-training with MCEM + TCL objectives.

        Returns a dictionary of final training metrics.
        """
        logger.info("Starting self-supervised pre-training")
        c = self.config.pretrain

        optimiser = AdamW(
            self.model.parameters(),
            lr=c.learning_rate,
            weight_decay=c.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimiser, T_max=c.epochs)

        mcem_loss_fn = MaskedClinicalEventLoss(self.config).to(self.device)
        tcl_loss_fn = TemporalContrastiveLoss(self.config).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(c.epochs):
            train_metrics = self._pretrain_epoch(
                epoch, train_loader, optimiser, mcem_loss_fn, tcl_loss_fn
            )
            scheduler.step()

            logger.info(
                f"Pre-train epoch {epoch + 1}/{c.epochs}: "
                f"loss={train_metrics['total_loss']:.4f}, "
                f"MCEM={train_metrics['mcem_loss']:.4f}, "
                f"TCL={train_metrics['tcl_loss']:.4f}"
            )

            # Validation
            if val_loader is not None:
                val_loss = self._pretrain_validate(
                    val_loader, mcem_loss_fn, tcl_loss_fn
                )
                logger.info(f"  Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint("pretrain_best.pt", epoch, val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= c.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break

            # Periodic checkpointing
            if (epoch + 1) % c.save_every_n_epochs == 0:
                self._save_checkpoint(
                    f"pretrain_epoch_{epoch + 1}.pt", epoch, train_metrics["total_loss"]
                )

        logger.info(f"Pre-training complete. Best val loss: {best_val_loss:.4f}")
        return {"best_val_loss": best_val_loss}

    def _pretrain_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        mcem_loss_fn: nn.Module,
        tcl_loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Run one epoch of self-supervised pre-training."""
        self.model.train()
        total_loss = 0.0
        mcem_total = 0.0
        tcl_total = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.config.mixed_precision):
                # MCEM: apply random masking
                masked_batch, mask_positions, targets = self._apply_masking(batch)
                outputs = self.model(masked_batch)

                mcem_predictions = {
                    "num_pred": self.model.mask_value_head(
                        outputs["fused_representation"]
                    ),
                }
                mcem_losses = mcem_loss_fn(
                    mcem_predictions, targets, mask_positions
                )

                # TCL: generate temporal pairs
                z_anchor, z_positive = self._generate_temporal_pairs(batch)
                tcl_loss = tcl_loss_fn(z_anchor, z_positive)

                loss = (
                    mcem_losses["mcem_total_loss"]
                    + self.config.pretrain.tcl_weight * tcl_loss
                )

            optimiser.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.pretrain.gradient_clip
            )
            self.scaler.step(optimiser)
            self.scaler.update()

            total_loss += loss.item()
            mcem_total += mcem_losses["mcem_total_loss"].item()
            tcl_total += tcl_loss.item()
            num_batches += 1

        return {
            "total_loss": total_loss / max(num_batches, 1),
            "mcem_loss": mcem_total / max(num_batches, 1),
            "tcl_loss": tcl_total / max(num_batches, 1),
        }

    def _pretrain_validate(
        self,
        loader: DataLoader,
        mcem_loss_fn: nn.Module,
        tcl_loss_fn: nn.Module,
    ) -> float:
        """Validation pass during pre-training."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                masked_batch, mask_positions, targets = self._apply_masking(batch)
                outputs = self.model(masked_batch)
                mcem_preds = {
                    "num_pred": self.model.mask_value_head(
                        outputs["fused_representation"]
                    )
                }
                mcem_losses = mcem_loss_fn(mcem_preds, targets, mask_positions)
                z_a, z_p = self._generate_temporal_pairs(batch)
                tcl_loss = tcl_loss_fn(z_a, z_p)
                loss = mcem_losses["mcem_total_loss"] + self.config.pretrain.tcl_weight * tcl_loss
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Fairness-aware supervised fine-tuning for cardiovascular risk prediction.
        """
        logger.info("Starting fairness-aware fine-tuning")
        c = self.config.finetune

        optimiser = AdamW(
            self.model.parameters(),
            lr=c.learning_rate,
            weight_decay=c.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimiser, T_max=c.epochs)
        loss_fn = FairnessAwareSurvivalLoss(self.config).to(self.device)

        best_auroc = 0.0
        patience_counter = 0

        for epoch in range(c.epochs):
            train_metrics = self._finetune_epoch(
                epoch, train_loader, optimiser, loss_fn
            )
            val_metrics = self.evaluate(val_loader, loss_fn)
            scheduler.step()

            logger.info(
                f"Fine-tune epoch {epoch + 1}/{c.epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_auroc={val_metrics['auroc']:.4f}"
            )

            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                patience_counter = 0
                self._save_checkpoint(
                    "finetune_best.pt", epoch, val_metrics["auroc"]
                )
            else:
                patience_counter += 1
                if patience_counter >= c.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info(f"Fine-tuning complete. Best val AUROC: {best_auroc:.4f}")
        return {"best_val_auroc": best_auroc}

    def _finetune_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Run one epoch of supervised fine-tuning."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(batch)
                losses = loss_fn(outputs, batch)
                loss = losses["total_loss"]

            optimiser.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.finetune.gradient_clip
            )
            self.scaler.step(optimiser)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Evaluate model on the given DataLoader."""
        from sklearn.metrics import roc_auc_score, average_precision_score

        self.model.eval()
        all_predictions = []
        all_events = []
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch)
            all_predictions.append(outputs["five_year_risk"].cpu())
            all_events.append(batch["event"].cpu())

            if loss_fn is not None:
                losses = loss_fn(outputs, batch)
                total_loss += losses["total_loss"].item()
                num_batches += 1

        predictions = torch.cat(all_predictions).numpy()
        events = torch.cat(all_events).numpy()

        metrics = {
            "auroc": float(roc_auc_score(events, predictions)) if len(set(events)) > 1 else 0.0,
            "auprc": float(average_precision_score(events, predictions)) if len(set(events)) > 1 else 0.0,
            "loss": total_loss / max(num_batches, 1) if loss_fn else 0.0,
            "num_patients": len(predictions),
            "event_rate": float(events.mean()),
        }
        return metrics

    def _apply_masking(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple:
        """Apply random masking for MCEM pre-training."""
        masked = {k: v.clone() for k, v in batch.items()}
        mask_ratio = self.config.pretrain.mask_ratio

        # Mask value positions (simplified)
        value_mask = batch["value_mask"].clone()
        random_mask = torch.rand_like(value_mask.float()) < mask_ratio
        mask_positions = value_mask & random_mask

        targets = {
            "num_targets": batch["values"].clone(),
            "code_targets": batch["feature_ids"].clone(),
        }

        # Set masked values to 0 for model input
        masked["values"] = torch.where(
            mask_positions, torch.zeros_like(batch["values"]), batch["values"]
        )
        return masked, mask_positions.any(dim=-1), targets

    def _generate_temporal_pairs(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Generate temporally adjacent positive pairs for contrastive learning.
        Simplified approach: apply stochastic augmentation to create views.
        """
        # View 1: standard forward pass
        z_anchor = self.model.encode_for_contrastive(batch)

        # View 2: augment by randomly dropping events
        augmented = {k: v.clone() for k, v in batch.items()}
        drop_prob = 0.2
        drop_mask = torch.rand_like(batch["value_mask"].float()) > drop_prob
        augmented["value_mask"] = batch["value_mask"] & drop_mask.bool()
        z_positive = self.model.encode_for_contrastive(augmented)

        return z_anchor, z_positive

    def _save_checkpoint(
        self, filename: str, epoch: int, metric: float
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "metric": metric,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
