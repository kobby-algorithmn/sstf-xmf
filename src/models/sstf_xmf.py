"""
SSTF-XMF Core Model Architecture
=================================
Implementation of the Self-Supervised Temporal Transformer with
Explainable Multi-Modal Fusion, incorporating:
  - Time-aware positional encoding (after Tipirneni and Reddy, 2022)
  - Hierarchical temporal transformer
  - Cross-modal attention fusion with gating
  - Concept bottleneck layer for clinical interpretability

Author: MSc Dissertation Research
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SSTFConfig


class Time2Vec(nn.Module):
    """
    Learnable continuous-time encoding using sinusoidal basis functions.

    Maps absolute timestamps to dense vector representations that the
    attention mechanism can use to weight observations by temporal proximity.
    """

    def __init__(self, hidden_dim: int, num_frequencies: int = 64):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.frequencies = nn.Linear(1, num_frequencies)
        self.projection = nn.Linear(num_frequencies + 1, hidden_dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len) in hours since reference

        Returns:
            Time-encoded tensor of shape (batch, seq_len, hidden_dim)
        """
        t = timestamps.unsqueeze(-1).float()
        trend = self.linear(t)
        periodic = torch.sin(self.frequencies(t))
        time_features = torch.cat([trend, periodic], dim=-1)
        return self.projection(time_features)


class StructuredEncoder(nn.Module):
    """
    Encoder for structured numerical features (laboratory results, vital signs).

    Combines learnable feature-type embeddings with piecewise-linear value
    transformations and explicit missing-value handling.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        num_features = config.model.num_lab_features + config.model.num_vital_features
        hidden_dim = config.model.hidden_dim

        self.feature_embed = nn.Embedding(num_features, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        self.missing_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=config.model.layer_norm_eps)

    def forward(
        self,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_ids: (B, V, E) feature-type integer ids
            values: (B, V, E) normalised numerical values
            mask: (B, V, E) boolean validity mask

        Returns:
            Visit-level representations (B, V, D)
        """
        feat_emb = self.feature_embed(feature_ids)
        val_emb = self.value_proj(values.unsqueeze(-1))
        combined = feat_emb + val_emb

        # Substitute missing values with learnt embedding
        mask_expanded = mask.unsqueeze(-1)
        combined = torch.where(mask_expanded, combined, self.missing_embed)

        # Pool events within each visit (mean over valid events)
        mask_f = mask.float().unsqueeze(-1)
        visit_repr = (combined * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1.0)

        return self.layer_norm(visit_repr)


class MedicationEncoder(nn.Module):
    """Encoder for medication prescription sequences."""

    def __init__(self, config: SSTFConfig):
        super().__init__()
        hidden_dim = config.model.hidden_dim

        self.med_embed = nn.Embedding(config.model.num_medication_codes, hidden_dim)
        self.dose_proj = nn.Linear(3, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=config.model.layer_norm_eps)

    def forward(
        self,
        med_ids: torch.Tensor,
        med_features: torch.Tensor,
        med_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args: (B, V, M), (B, V, M, 3), (B, V, M) -> (B, D)."""
        B, V, M = med_ids.shape
        embeds = self.med_embed(med_ids) + self.dose_proj(med_features)
        embeds_flat = embeds.view(B * V, M, -1)
        mask_flat = med_mask.view(B * V, M)

        key_padding = ~mask_flat
        # Prevent NaN when all tokens are padded
        all_padded = key_padding.all(dim=-1)
        key_padding = key_padding.clone()
        key_padding[all_padded, 0] = False

        attn_out, _ = self.attention(
            embeds_flat, embeds_flat, embeds_flat,
            key_padding_mask=key_padding,
        )
        mask_f = mask_flat.float().unsqueeze(-1)
        pooled = (attn_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = pooled.view(B, V, -1)

        # Aggregate across visits to patient-level representation
        return self.layer_norm(pooled.mean(dim=1))


class ClinicalTextEncoder(nn.Module):
    """ClinicalBERT-based encoder for unstructured clinical narratives."""

    def __init__(self, config: SSTFConfig):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(config.model.text_model_name)

        # Freeze all but final N transformer layers
        total_layers = len(self.bert.encoder.layer)
        trainable_from = total_layers - config.model.text_trainable_layers
        for i, layer in enumerate(self.bert.encoder.layer):
            requires_grad = i >= trainable_from
            for param in layer.parameters():
                param.requires_grad = requires_grad

        self.projection = nn.Linear(
            config.model.text_embed_dim, config.model.hidden_dim
        )
        self.layer_norm = nn.LayerNorm(
            config.model.hidden_dim, eps=config.model.layer_norm_eps
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Args: (B, T), (B, T) -> (B, D)."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.layer_norm(self.projection(cls_embedding))


class DilatedAttention(nn.Module):
    """
    Multi-scale dilated self-attention with continuous-time bias.

    Different attention heads employ different temporal receptive fields,
    enabling simultaneous modelling of short-term fluctuations and long-term
    trajectories across irregular clinical timelines.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable temporal bias coefficients per head
        self.temporal_coeffs = nn.Parameter(torch.ones(num_heads) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        time_diffs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if time_diffs is not None:
            # Logarithmic temporal bias
            temporal_bias = -torch.log1p(time_diffs.abs()).unsqueeze(1)
            coeffs = self.temporal_coeffs.view(1, -1, 1, 1)
            attn_scores = attn_scores + temporal_bias * coeffs

        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), attn_weights


class TemporalTransformerLayer(nn.Module):
    """Single transformer encoder layer with temporal attention."""

    def __init__(self, config: SSTFConfig):
        super().__init__()
        hidden_dim = config.model.hidden_dim
        self.attention = DilatedAttention(
            hidden_dim, config.model.num_heads, config.model.dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=config.model.layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=config.model.layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, config.model.ff_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.ff_dim, hidden_dim),
            nn.Dropout(config.model.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_diffs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(self.norm1(x), time_diffs, mask)
        x = x + attn_out
        x = x + self.feedforward(self.norm2(x))
        return x, attn_weights


class HierarchicalTemporalTransformer(nn.Module):
    """
    Hierarchical transformer processing events within visits (intra-visit)
    and visits across time (inter-visit) with continuous-time encodings.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        self.time_encoder = Time2Vec(
            config.model.hidden_dim, config.model.time_num_frequencies
        )
        self.intra_layers = nn.ModuleList([
            TemporalTransformerLayer(config)
            for _ in range(config.model.intra_visit_layers)
        ])
        self.inter_layers = nn.ModuleList([
            TemporalTransformerLayer(config)
            for _ in range(config.model.inter_visit_layers)
        ])
        self.visit_projection = nn.Linear(
            config.model.hidden_dim, config.model.hidden_dim
        )

    def forward(
        self,
        visit_embeddings: torch.Tensor,
        visit_times: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            visit_embeddings: (B, V, D) visit-level representations
            visit_times: (B, V) absolute visit timestamps (hours)
            visit_mask: (B, V) visit validity mask

        Returns:
            contextualised_visits: (B, V, D)
            attention_weights: list of per-layer attention tensors
        """
        # Add continuous-time encoding
        time_encoded = self.time_encoder(visit_times)
        x = visit_embeddings + time_encoded

        # Pairwise temporal differences for attention bias
        time_diffs = visit_times.unsqueeze(-1) - visit_times.unsqueeze(-2)

        attention_weights = []
        for layer in self.inter_layers:
            x, weights = layer(x, time_diffs=time_diffs, mask=visit_mask)
            attention_weights.append(weights)

        return self.visit_projection(x), attention_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention fusion with modality-specific gating.

    Enables each modality to selectively attend to information in other
    modalities, with learnable gates controlling contribution strength.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        self.num_modalities = config.model.num_modalities
        hidden_dim = config.model.hidden_dim

        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=config.model.num_heads,
                dropout=config.model.dropout,
                batch_first=True,
            )
            for _ in range(self.num_modalities)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
            for _ in range(self.num_modalities)
        ])

        self.fusion_norm = nn.LayerNorm(hidden_dim, eps=config.model.layer_norm_eps)

    def forward(
        self, modality_representations: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_representations: list of M tensors each (B, D)

        Returns:
            fused_representation: (B, D)
            gate_values: (B, M) for interpretability
        """
        stacked = torch.stack(modality_representations, dim=1)  # (B, M, D)

        enriched = []
        gates = []
        for i in range(self.num_modalities):
            query = modality_representations[i].unsqueeze(1)
            attn_out, _ = self.cross_attention[i](query, stacked, stacked)
            attn_out = attn_out.squeeze(1)
            gate = self.gates[i](modality_representations[i])
            enriched.append(attn_out * gate)
            gates.append(gate)

        fused = sum(enriched)
        gate_values = torch.cat(gates, dim=-1)
        return self.fusion_norm(fused), gate_values


class ConceptBottleneck(nn.Module):
    """
    Concept bottleneck layer for intrinsic clinical interpretability.

    Routes predictions through a set of clinician-validated concepts
    corresponding to established cardiovascular risk factors.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        self.concept_names = config.model.concept_names
        self.num_concepts = config.model.num_concepts
        hidden_dim = config.model.hidden_dim

        self.concept_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(config.model.dropout),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
            for _ in range(self.num_concepts)
        ])

        self.risk_head = nn.Sequential(
            nn.Linear(self.num_concepts, 64),
            nn.GELU(),
            nn.Linear(64, config.model.num_time_bins),
        )

    def forward(
        self, fused_representation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused_representation: (B, D)

        Returns:
            hazard_logits: (B, T) discrete-time hazard logits
            concept_values: (B, C) interpretable concept scores in [0, 1]
        """
        concepts = [pred(fused_representation) for pred in self.concept_predictors]
        concept_values = torch.cat(concepts, dim=-1)
        hazard_logits = self.risk_head(concept_values)
        return hazard_logits, concept_values


class SSTFXMF(nn.Module):
    """
    Full Self-Supervised Temporal Transformer with
    Explainable Multi-Modal Fusion.
    """

    def __init__(self, config: SSTFConfig):
        super().__init__()
        self.config = config

        # Modality-specific encoders
        self.structured_encoder = StructuredEncoder(config)
        self.text_encoder = ClinicalTextEncoder(config)
        self.medication_encoder = MedicationEncoder(config)
        self.imaging_projection = nn.Linear(
            config.model.imaging_feature_dim, config.model.hidden_dim
        )

        # Temporal backbone
        self.temporal_transformer = HierarchicalTemporalTransformer(config)

        # Multi-modal fusion and prediction
        self.cross_modal_fusion = CrossModalAttention(config)
        self.concept_bottleneck = ConceptBottleneck(config)

        # Pre-training heads
        self.mask_value_head = nn.Linear(
            config.model.hidden_dim, config.model.num_lab_features
        )
        self.mask_code_head = nn.Linear(
            config.model.hidden_dim, config.model.num_diagnosis_codes
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
            nn.GELU(),
            nn.Linear(config.model.hidden_dim, 128),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with truncated normal for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def encode_patient(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode all modalities and return representations plus attention."""
        # Structured modality via hierarchical temporal transformer
        visit_embeddings = self.structured_encoder(
            batch["feature_ids"], batch["values"], batch["value_mask"]
        )
        contextual_visits, attention_weights = self.temporal_transformer(
            visit_embeddings, batch["visit_times"], batch["visit_mask"]
        )

        # Pool to patient-level representation
        mask = batch["visit_mask"].float().unsqueeze(-1)
        structured_rep = (contextual_visits * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # Other modalities
        text_rep = self.text_encoder(batch["text_ids"], batch["text_mask"])
        med_rep = self.medication_encoder(
            batch["med_ids"], batch["med_features"], batch["med_mask"]
        )
        imaging_rep = self.imaging_projection(batch["imaging_features"])

        return [structured_rep, text_rep, med_rep, imaging_rep], attention_weights

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass producing predictions and interpretability outputs."""
        modality_reps, attention_weights = self.encode_patient(batch)
        fused, gate_values = self.cross_modal_fusion(modality_reps)
        hazard_logits, concept_values = self.concept_bottleneck(fused)

        # Compute survival probability
        hazards = torch.sigmoid(hazard_logits)
        survival_probs = torch.cumprod(1.0 - hazards, dim=-1)
        five_year_risk = 1.0 - survival_probs[:, -1]

        return {
            "hazard_logits": hazard_logits,
            "survival_probs": survival_probs,
            "five_year_risk": five_year_risk,
            "concept_values": concept_values,
            "gate_values": gate_values,
            "fused_representation": fused,
            "modality_representations": modality_reps,
            "attention_weights": attention_weights,
        }

    def encode_for_contrastive(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode for contrastive pre-training (returns projection head output)."""
        modality_reps, _ = self.encode_patient(batch)
        fused, _ = self.cross_modal_fusion(modality_reps)
        return self.contrastive_projection(fused)

    def get_num_parameters(self) -> Dict[str, int]:
        """Report parameter counts per architectural component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "structured_encoder": count(self.structured_encoder),
            "text_encoder": count(self.text_encoder),
            "medication_encoder": count(self.medication_encoder),
            "temporal_transformer": count(self.temporal_transformer),
            "cross_modal_fusion": count(self.cross_modal_fusion),
            "concept_bottleneck": count(self.concept_bottleneck),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
