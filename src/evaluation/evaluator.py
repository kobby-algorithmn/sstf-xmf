"""
SSTF-XMF Evaluation Module
===========================
Comprehensive evaluation with discrimination, calibration, clinical utility,
and fairness metrics, accompanied by rigorous statistical validation.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy import stats

from ..config import SSTFConfig
from ..models.sstf_xmf import SSTFXMF

logger = logging.getLogger(__name__)


class ClinicalEvaluator:
    """
    Comprehensive evaluation aligned with clinical AI validation standards
    (Wiens et al., 2022; Obermeyer et al., 2021).
    """

    def __init__(
        self, model: SSTFXMF, config: SSTFConfig, device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device or config.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """Generate predictions on the provided DataLoader."""
        all_predictions = []
        all_events = []
        all_time_bins = []
        all_demographics = []
        all_concepts = []
        all_gates = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch)
            all_predictions.append(outputs["five_year_risk"].cpu().numpy())
            all_events.append(batch["event"].cpu().numpy())
            all_time_bins.append(batch["time_bin"].cpu().numpy())
            all_demographics.append(batch["demographic_group"].cpu().numpy())
            all_concepts.append(outputs["concept_values"].cpu().numpy())
            all_gates.append(outputs["gate_values"].cpu().numpy())

        return {
            "predictions": np.concatenate(all_predictions),
            "events": np.concatenate(all_events),
            "time_bins": np.concatenate(all_time_bins),
            "demographics": np.concatenate(all_demographics),
            "concepts": np.concatenate(all_concepts),
            "gates": np.concatenate(all_gates),
        }

    def compute_discrimination_metrics(
        self, predictions: np.ndarray, events: np.ndarray
    ) -> Dict[str, float]:
        """Discrimination metrics: AUROC, AUPRC, Brier score."""
        if len(np.unique(events)) < 2:
            return {"auroc": 0.5, "auprc": events.mean(), "brier": 0.0}

        return {
            "auroc": float(roc_auc_score(events, predictions)),
            "auprc": float(average_precision_score(events, predictions)),
            "brier": float(brier_score_loss(events, predictions)),
        }

    def compute_calibration_metrics(
        self, predictions: np.ndarray, events: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """Calibration metrics: ICI, calibration slope, calibration-in-the-large."""
        # Integrated Calibration Index (Austin and Steyerberg, 2023)
        sorted_idx = np.argsort(predictions)
        sorted_preds = predictions[sorted_idx]
        sorted_events = events[sorted_idx]

        bin_size = max(len(sorted_preds) // n_bins, 1)
        bin_mean_pred = []
        bin_mean_obs = []
        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else len(sorted_preds)
            if end > start:
                bin_mean_pred.append(sorted_preds[start:end].mean())
                bin_mean_obs.append(sorted_events[start:end].mean())

        bin_mean_pred = np.array(bin_mean_pred)
        bin_mean_obs = np.array(bin_mean_obs)
        ici = float(np.abs(bin_mean_pred - bin_mean_obs).mean())

        # Calibration slope via logistic regression of events on logit(predictions)
        eps = 1e-7
        logits = np.log(np.clip(predictions, eps, 1 - eps) /
                        np.clip(1 - predictions, eps, 1 - eps))
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(logits.reshape(-1, 1), events)
            calib_slope = float(lr.coef_[0, 0])
            calib_large = float(lr.intercept_[0])
        except Exception:
            calib_slope = 1.0
            calib_large = 0.0

        return {
            "integrated_calibration_index": ici,
            "calibration_slope": calib_slope,
            "calibration_in_the_large": calib_large,
        }

    def compute_clinical_utility(
        self,
        predictions: np.ndarray,
        events: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> Dict[str, List[float]]:
        """Decision curve analysis: net benefit across clinical thresholds."""
        if thresholds is None:
            thresholds = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]

        net_benefits = []
        for t in thresholds:
            tp = ((predictions >= t) & (events == 1)).sum()
            fp = ((predictions >= t) & (events == 0)).sum()
            n = len(predictions)
            if n == 0:
                nb = 0.0
            else:
                odds = t / (1 - t) if t < 1 else 1e6
                nb = float(tp / n - (fp / n) * odds)
            net_benefits.append(nb)

        return {
            "thresholds": thresholds,
            "net_benefit": net_benefits,
        }

    def compute_fairness_metrics(
        self,
        predictions: np.ndarray,
        events: np.ndarray,
        demographics: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Demographic-stratified performance (Obermeyer et al., 2021).

        Reports AUROC, AUPRC, event rate, and mean predicted risk per group,
        plus the maximum subgroup disparity.
        """
        unique_groups = np.unique(demographics)
        group_metrics: Dict[str, Dict] = {}

        for g in unique_groups:
            mask = demographics == g
            if mask.sum() < 10:
                continue
            if len(np.unique(events[mask])) < 2:
                continue
            group_metrics[f"group_{int(g)}"] = {
                "n": int(mask.sum()),
                "event_rate": float(events[mask].mean()),
                "mean_predicted_risk": float(predictions[mask].mean()),
                "auroc": float(roc_auc_score(events[mask], predictions[mask])),
                "auprc": float(average_precision_score(events[mask], predictions[mask])),
            }

        aurocs = [m["auroc"] for m in group_metrics.values()]
        max_disparity = max(aurocs) - min(aurocs) if aurocs else 0.0

        return {
            "per_group_metrics": group_metrics,
            "max_auroc_disparity": float(max_disparity),
            "num_groups_evaluated": len(group_metrics),
        }

    def bootstrap_confidence_intervals(
        self,
        predictions: np.ndarray,
        events: np.ndarray,
        metric_fn,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap percentile confidence intervals with stratified resampling.
        """
        n = len(predictions)
        pos_idx = np.where(events == 1)[0]
        neg_idx = np.where(events == 0)[0]

        bootstrap_metrics = []
        rng = np.random.RandomState(self.config.seed)

        for _ in range(n_bootstrap):
            # Stratified resampling preserves class balance
            pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            sample_idx = np.concatenate([pos_sample, neg_sample])
            try:
                bootstrap_metrics.append(
                    metric_fn(events[sample_idx], predictions[sample_idx])
                )
            except Exception:
                pass

        point_estimate = metric_fn(events, predictions)
        lower = float(np.percentile(bootstrap_metrics, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2)))
        return point_estimate, lower, upper

    def delong_test(
        self,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        events: np.ndarray,
    ) -> Dict[str, float]:
        """
        DeLong's test for comparing two correlated AUROCs.
        Simplified implementation; production use should employ the
        pROC package or equivalent validated implementation.
        """
        auroc_a = roc_auc_score(events, predictions_a)
        auroc_b = roc_auc_score(events, predictions_b)

        # Approximate variance via bootstrap
        n_boot = 200
        diffs = []
        rng = np.random.RandomState(self.config.seed)
        for _ in range(n_boot):
            idx = rng.choice(len(events), len(events), replace=True)
            try:
                da = roc_auc_score(events[idx], predictions_a[idx])
                db = roc_auc_score(events[idx], predictions_b[idx])
                diffs.append(da - db)
            except Exception:
                pass
        diffs = np.array(diffs)
        z_stat = (auroc_a - auroc_b) / (diffs.std() + 1e-8)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "auroc_a": float(auroc_a),
            "auroc_b": float(auroc_b),
            "difference": float(auroc_a - auroc_b),
            "z_statistic": float(z_stat),
            "p_value": float(p_value),
            "significant_at_0.05": bool(p_value < 0.05),
        }

    def comprehensive_evaluation(
        self, loader: DataLoader, compute_ci: bool = True
    ) -> Dict:
        """
        Run the complete evaluation pipeline and return a structured report.
        """
        logger.info("Running comprehensive evaluation")
        preds = self.predict(loader)

        report = {
            "discrimination": self.compute_discrimination_metrics(
                preds["predictions"], preds["events"]
            ),
            "calibration": self.compute_calibration_metrics(
                preds["predictions"], preds["events"]
            ),
            "clinical_utility": self.compute_clinical_utility(
                preds["predictions"], preds["events"]
            ),
            "fairness": self.compute_fairness_metrics(
                preds["predictions"], preds["events"], preds["demographics"]
            ),
            "num_patients": int(len(preds["predictions"])),
            "event_rate": float(preds["events"].mean()),
        }

        if compute_ci:
            auroc_pt, auroc_lo, auroc_hi = self.bootstrap_confidence_intervals(
                preds["predictions"], preds["events"], roc_auc_score
            )
            report["discrimination"]["auroc_95ci"] = [auroc_lo, auroc_hi]

        logger.info(
            f"Evaluation complete: AUROC={report['discrimination']['auroc']:.3f}, "
            f"ICI={report['calibration']['integrated_calibration_index']:.3f}"
        )
        return report
