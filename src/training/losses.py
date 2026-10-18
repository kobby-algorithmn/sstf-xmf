"""
SSTF-XMF Loss Functions
=========================
Implementation of self-supervised pre-training objectives and
fairness-aware supervised fine-tuning losses.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SSTFConfig


class MaskedClinicalEventLoss(nn.Module):
    """
    Masked Clinical Event Modelling (MCEM) loss.

    Extends the BERT masked language modelling paradigm to the multi-modal
    EHR domain, combining regression for numerical features with
    classification for medical codes.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict with 'num_pred' (B, N) and 'code_pred' (B, C)
            targets: dict with 'num_targets' and 'code_targets'
            mask_positions: boolean tensor indicating masked locations

        Returns:
            Dictionary of individual and combined loss components
        """
        # Numerical feature regression loss
        num_loss = torch.tensor(0.0, device=mask_positions.device)
        if "num_pred" in predictions and mask_positions.any():
            num_errors = self.mse(
                predictions["num_pred"],
                targets["num_targets"]
            )
            num_loss = num_errors[mask_positions].mean()

        # Code classification loss
        code_loss = torch.tensor(0.0, device=mask_positions.device)
        if "code_pred" in predictions and "code_targets" in targets:
            code_targets = targets["code_targets"].clone()
            code_targets[~mask_positions] = -100
            code_errors = self.ce(
                predictions["code_pred"].transpose(1, 2),
                code_targets,
            )
            if mask_positions.any():
                code_loss = code_errors[mask_positions].mean()

        total = num_loss + code_loss
        return {
            "mcem_numerical_loss": num_loss,
            "mcem_code_loss": code_loss,
            "mcem_total_loss": total,
        }


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal Contrastive Learning (TCL) using InfoNCE objective.

    Encourages representations of temporally adjacent patient record
    segments to be similar whilst distinguishing between patients.
    Hard negative mining selects demographically similar but
    trajectory-different patients as negatives.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        self.temperature = config.pretrain.tcl_temperature

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_anchor: (B, D) anchor representations
            z_positive: (B, D) positive pair representations
            z_negatives: optional (B, K, D) hard negative representations

        Returns:
            InfoNCE contrastive loss scalar
        """
        # L2 normalise
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)

        # In-batch negatives + optional hard negatives
        if z_negatives is not None:
            z_negatives = F.normalize(z_negatives, dim=-1)
            # (B, 1 + K + B-1)
            all_keys = torch.cat([
                z_positive.unsqueeze(1),
                z_negatives,
                z_positive.unsqueeze(0).expand(z_anchor.size(0), -1, -1),
            ], dim=1)
            logits = torch.bmm(
                z_anchor.unsqueeze(1), all_keys.transpose(-2, -1)
            ).squeeze(1) / self.temperature
            labels = torch.zeros(
                z_anchor.size(0), dtype=torch.long, device=z_anchor.device
            )
            return F.cross_entropy(logits, labels)

        # Standard in-batch contrastive
        logits = torch.matmul(z_anchor, z_positive.T) / self.temperature
        labels = torch.arange(z_anchor.size(0), device=z_anchor.device)
        return F.cross_entropy(logits, labels)


class DiscreteTimeSurvivalLoss(nn.Module):
    """
    Discrete-time survival analysis loss (Kvamme and Borgan, 2021).

    Models the probability of an event occurring in each of K discrete
    time intervals, accommodating right-censored observations.
    """

    def __init__(self, num_time_bins: int):
        super().__init__()
        self.num_time_bins = num_time_bins

    def forward(
        self,
        hazard_logits: torch.Tensor,
        events: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hazard_logits: (B, T) discrete hazard logits
            events: (B,) event indicator (1 = event, 0 = censored)
            time_bins: (B,) time-bin index of event or censoring

        Returns:
            Negative log-likelihood loss
        """
        B, T = hazard_logits.shape
        hazards = torch.sigmoid(hazard_logits)
        log_hazards = torch.log(hazards.clamp(min=1e-8))
        log_survival = torch.log((1.0 - hazards).clamp(min=1e-8))

        # Loss per observation
        losses = torch.zeros(B, device=hazard_logits.device)
        for i in range(B):
            t = time_bins[i].item()
            e = events[i].item()

            # Sum log-survival over intervals before event/censoring
            if t > 0:
                losses[i] = losses[i] - log_survival[i, :t].sum()

            # If event occurred, add event log-hazard; else add censored survival
            if e > 0 and t < T:
                losses[i] = losses[i] - log_hazards[i, t]
            elif e == 0 and t < T:
                losses[i] = losses[i] - log_survival[i, t]

        return losses.mean()


class CalibrationLoss(nn.Module):
    """
    Calibration regularisation penalising predicted probability deviations
    from observed event rates across deciles.
    """

    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins

    def forward(
        self, predicted_risk: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """Args: (B,) predicted 5-year risk, (B,) event indicator."""
        if len(predicted_risk) < self.num_bins:
            return torch.tensor(0.0, device=predicted_risk.device)

        # Sort by predicted risk
        sorted_idx = torch.argsort(predicted_risk)
        sorted_risk = predicted_risk[sorted_idx]
        sorted_events = events[sorted_idx]

        bin_size = len(sorted_risk) // self.num_bins
        calibration_error = torch.tensor(0.0, device=predicted_risk.device)

        for b in range(self.num_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < self.num_bins - 1 else len(sorted_risk)
            if end > start:
                mean_pred = sorted_risk[start:end].mean()
                mean_obs = sorted_events[start:end].float().mean()
                calibration_error = calibration_error + (mean_pred - mean_obs).abs()

        return calibration_error / self.num_bins


class FairnessRegularisation(nn.Module):
    """
    Fairness regularisation penalising systematic performance disparities
    across demographic subgroups (following Obermeyer et al., 2021).
    """

    def forward(
        self,
        predictions: torch.Tensor,
        events: torch.Tensor,
        demographic_groups: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) predicted risks
            events: (B,) event indicator
            demographic_groups: (B,) group indicator

        Returns:
            Pairwise group risk disparity penalty
        """
        unique_groups = torch.unique(demographic_groups)
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=predictions.device)

        group_mean_risks = []
        for g in unique_groups:
            mask = demographic_groups == g
            if mask.sum() > 0:
                group_mean_risks.append(predictions[mask].mean())

        penalty = torch.tensor(0.0, device=predictions.device)
        num_pairs = 0
        for i in range(len(group_mean_risks)):
            for j in range(i + 1, len(group_mean_risks)):
                penalty = penalty + (group_mean_risks[i] - group_mean_risks[j]).abs()
                num_pairs += 1

        return penalty / max(num_pairs, 1)


class ConceptConsistencyLoss(nn.Module):
    """
    Enforces clinical plausibility constraints on concept bottleneck outputs.

    Penalises implausible combinations such as high predicted risk with
    excellent concept values across all major risk factors.
    """

    def forward(
        self,
        concept_values: torch.Tensor,
        predicted_risk: torch.Tensor,
    ) -> torch.Tensor:
        """Args: (B, C) concept values in [0,1], (B,) predicted risks."""
        # Average concept score (higher => worse control)
        concept_mean = concept_values.mean(dim=-1)
        # Inconsistency when risk prediction contradicts concept signal
        inconsistency = F.relu(predicted_risk - concept_mean - 0.3).mean()
        return inconsistency


class FairnessAwareSurvivalLoss(nn.Module):
    """
    Composite loss combining discrete-time survival NLL, calibration,
    concept consistency, and fairness regularisation.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        self.config = config
        self.survival_loss = DiscreteTimeSurvivalLoss(config.model.num_time_bins)
        self.calibration_loss = CalibrationLoss()
        self.concept_loss = ConceptConsistencyLoss()
        self.fairness_loss = FairnessRegularisation()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute composite loss with per-component logging."""
        w = self.config.finetune

        nll = self.survival_loss(
            outputs["hazard_logits"],
            batch["event"],
            batch["time_bin"],
        )
        calib = self.calibration_loss(
            outputs["five_year_risk"],
            batch["event"],
        )
        concept = self.concept_loss(
            outputs["concept_values"],
            outputs["five_year_risk"],
        )
        fairness = self.fairness_loss(
            outputs["five_year_risk"],
            batch["event"],
            batch["demographic_group"],
        )

        total = (
            w.survival_loss_weight * nll
            + w.calibration_loss_weight * calib
            + w.concept_consistency_weight * concept
            + w.fairness_lambda * fairness
        )

        return {
            "total_loss": total,
            "survival_nll": nll,
            "calibration_loss": calib,
            "concept_consistency": concept,
            "fairness_penalty": fairness,
        }
