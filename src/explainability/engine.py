"""
SSTF-XMF Explainability Module
================================
Integrated explanation framework combining:
  - Modality-level gating attributions
  - Temporal attention-based visit importance
  - SHAP-based feature attributions
  - Concept bottleneck clinical interpretations

Informed by Ghassemi, Oakden-Rayner and Beam (2021) critique of
explainability in clinical AI.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import torch

from ..config import SSTFConfig
from ..models.sstf_xmf import SSTFXMF

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """Multi-layered explainability for SSTF-XMF predictions."""

    def __init__(
        self, model: SSTFXMF, config: SSTFConfig, device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device or config.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def explain_patient(
        self, batch: Dict[str, torch.Tensor], patient_idx: int = 0
    ) -> Dict:
        """Generate a comprehensive explanation for a single patient."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(batch)

        explanation = {
            "prediction": {
                "five_year_risk": float(outputs["five_year_risk"][patient_idx].item()),
                "survival_curve": outputs["survival_probs"][patient_idx].cpu().numpy().tolist(),
            },
            "modality_contributions": self._extract_modality_gates(
                outputs["gate_values"][patient_idx]
            ),
            "concept_attributions": self._extract_concept_values(
                outputs["concept_values"][patient_idx]
            ),
            "temporal_attention": self._extract_temporal_attention(
                outputs["attention_weights"], patient_idx, batch["visit_mask"][patient_idx]
            ),
        }
        return explanation

    def _extract_modality_gates(
        self, gate_values: torch.Tensor
    ) -> Dict[str, float]:
        """Extract modality-level contribution gating values."""
        gates = gate_values.cpu().numpy()
        return {
            name: float(gates[i])
            for i, name in enumerate(self.config.model.modality_names)
        }

    def _extract_concept_values(
        self, concept_values: torch.Tensor
    ) -> Dict[str, float]:
        """Extract concept bottleneck values in clinical vocabulary."""
        values = concept_values.cpu().numpy()
        return {
            name: float(values[i])
            for i, name in enumerate(self.config.model.concept_names)
        }

    def _extract_temporal_attention(
        self,
        attention_weights: List[torch.Tensor],
        patient_idx: int,
        visit_mask: torch.Tensor,
    ) -> Dict:
        """Aggregate inter-visit attention weights across layers."""
        if not attention_weights:
            return {"visit_importance": []}

        # Average across layers and heads
        stacked = torch.stack([aw[patient_idx] for aw in attention_weights])
        mean_attn = stacked.mean(dim=(0, 1))  # (V, V)

        # Column-wise attention indicates how often each visit is attended to
        visit_importance = mean_attn.mean(dim=0).cpu().numpy()
        mask = visit_mask.cpu().numpy()
        visit_importance = visit_importance * mask

        return {
            "visit_importance": visit_importance.tolist(),
            "num_valid_visits": int(mask.sum()),
        }

    def compute_shap_attributions(
        self,
        batch: Dict[str, torch.Tensor],
        background_batch: Dict[str, torch.Tensor],
        n_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP-style feature attributions using Integrated Gradients.
        Production deployment should use the shap package's DeepExplainer.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        background_batch = {
            k: v.to(self.device) for k, v in background_batch.items()
        }

        # Simplified integrated gradients over the values tensor
        values = batch["values"].clone().requires_grad_(True)
        baseline = background_batch["values"]

        alphas = torch.linspace(0, 1, n_samples, device=self.device)
        attributions = torch.zeros_like(values)

        for alpha in alphas:
            interpolated = baseline + alpha * (values - baseline)
            modified_batch = {**batch, "values": interpolated}
            modified_batch["values"].requires_grad_(True)

            with torch.enable_grad():
                outputs = self.model(modified_batch)
                risk = outputs["five_year_risk"].sum()
                grads = torch.autograd.grad(
                    risk, modified_batch["values"], retain_graph=False
                )[0]
            attributions = attributions + grads / n_samples

        attributions = (values - baseline) * attributions

        return {
            "feature_attributions": attributions.cpu().detach().numpy(),
        }

    def rank_top_risk_factors(
        self,
        concept_values: np.ndarray,
        baseline_values: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """Rank concepts contributing most to elevated risk."""
        if baseline_values is None:
            baseline_values = np.full_like(concept_values, 0.3)

        deviations = concept_values - baseline_values
        ranked_indices = np.argsort(-deviations)[:top_k]

        return [
            {
                "concept": self.config.model.concept_names[i],
                "value": float(concept_values[i]),
                "baseline": float(baseline_values[i]),
                "deviation": float(deviations[i]),
            }
            for i in ranked_indices
        ]

    def generate_clinical_narrative(self, explanation: Dict) -> str:
        """
        Generate a natural-language clinical narrative from explanation.
        Aligned with the recommendations of Saria, Buber and Gao (2023)
        for clinician-friendly AI output presentation.
        """
        risk = explanation["prediction"]["five_year_risk"]
        risk_category = (
            "high" if risk >= 0.20 else "intermediate" if risk >= 0.10 else "low"
        )

        concepts = explanation["concept_attributions"]
        top_concepts = sorted(
            concepts.items(), key=lambda x: -x[1]
        )[:3]
        top_concept_names = [c[0].replace("_", " ") for c in top_concepts]

        modalities = explanation["modality_contributions"]
        dominant_modality = max(modalities.items(), key=lambda x: x[1])[0]

        narrative = (
            f"This patient has an estimated five-year cardiovascular risk of "
            f"{risk * 100:.1f}%, classifying them in the {risk_category}-risk category. "
            f"The principal risk drivers identified by the model are "
            f"{', '.join(top_concept_names)}. "
            f"The prediction is primarily informed by the {dominant_modality} modality "
            f"(contribution weight: {modalities[dominant_modality]:.2f}). "
            f"Clinical review of these risk factors is recommended to guide "
            f"targeted preventive interventions."
        )
        return narrative
