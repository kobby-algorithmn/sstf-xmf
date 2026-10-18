"""
Unit Tests for SSTF-XMF Model
===============================
Verifies correct tensor shapes and forward-pass behaviour.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_default_config
from src.models.sstf_xmf import (
    SSTFXMF,
    Time2Vec,
    StructuredEncoder,
    HierarchicalTemporalTransformer,
    CrossModalAttention,
    ConceptBottleneck,
)


@pytest.fixture
def config():
    c = get_default_config()
    # Smaller dimensions for fast testing
    c.model.hidden_dim = 64
    c.model.num_heads = 4
    c.model.intra_visit_layers = 1
    c.model.inter_visit_layers = 2
    c.model.ff_dim = 128
    c.model.max_visits = 16
    c.model.max_events_per_visit = 8
    c.model.max_medications_per_visit = 4
    c.model.max_text_tokens = 32
    c.model.num_time_bins = 10
    c.device = "cpu"
    return c


@pytest.fixture
def dummy_batch(config):
    B = 2
    V = config.model.max_visits
    E = config.model.max_events_per_visit
    M = config.model.max_medications_per_visit
    T = config.model.max_text_tokens

    return {
        "feature_ids": torch.randint(0, 128, (B, V, E)),
        "values": torch.randn(B, V, E),
        "value_mask": torch.ones(B, V, E, dtype=torch.bool),
        "event_times": torch.randn(B, V, E).abs(),
        "visit_times": torch.linspace(0, 8000, V).unsqueeze(0).expand(B, -1),
        "visit_mask": torch.ones(B, V, dtype=torch.bool),
        "med_ids": torch.randint(0, 600, (B, V, M)),
        "med_features": torch.randn(B, V, M, 3),
        "med_mask": torch.ones(B, V, M, dtype=torch.bool),
        "text_ids": torch.randint(0, 30000, (B, T)),
        "text_mask": torch.ones(B, T, dtype=torch.bool),
        "imaging_features": torch.randn(B, config.model.imaging_feature_dim),
        "event": torch.randint(0, 2, (B,)).float(),
        "time_bin": torch.randint(0, config.model.num_time_bins, (B,)),
        "demographic_group": torch.randint(0, 4, (B,)),
    }


class TestTime2Vec:
    def test_output_shape(self, config):
        t2v = Time2Vec(config.model.hidden_dim, 16)
        timestamps = torch.randn(2, 10)
        out = t2v(timestamps)
        assert out.shape == (2, 10, config.model.hidden_dim)


class TestStructuredEncoder:
    def test_forward(self, config, dummy_batch):
        encoder = StructuredEncoder(config)
        out = encoder(
            dummy_batch["feature_ids"],
            dummy_batch["values"],
            dummy_batch["value_mask"],
        )
        B, V = dummy_batch["feature_ids"].shape[:2]
        assert out.shape == (B, V, config.model.hidden_dim)


class TestHierarchicalTransformer:
    def test_forward(self, config, dummy_batch):
        transformer = HierarchicalTemporalTransformer(config)
        B, V = dummy_batch["visit_mask"].shape
        visit_embeds = torch.randn(B, V, config.model.hidden_dim)
        out, attn = transformer(
            visit_embeds, dummy_batch["visit_times"], dummy_batch["visit_mask"]
        )
        assert out.shape == (B, V, config.model.hidden_dim)
        assert len(attn) == config.model.inter_visit_layers


class TestCrossModalAttention:
    def test_forward(self, config):
        fusion = CrossModalAttention(config)
        B = 2
        modality_reps = [
            torch.randn(B, config.model.hidden_dim)
            for _ in range(config.model.num_modalities)
        ]
        fused, gates = fusion(modality_reps)
        assert fused.shape == (B, config.model.hidden_dim)
        assert gates.shape == (B, config.model.num_modalities)
        # Gates should be in [0, 1]
        assert torch.all(gates >= 0) and torch.all(gates <= 1)


class TestConceptBottleneck:
    def test_forward(self, config):
        bottleneck = ConceptBottleneck(config)
        B = 2
        fused = torch.randn(B, config.model.hidden_dim)
        hazards, concepts = bottleneck(fused)
        assert hazards.shape == (B, config.model.num_time_bins)
        assert concepts.shape == (B, config.model.num_concepts)
        # Concept values should be in [0, 1]
        assert torch.all(concepts >= 0) and torch.all(concepts <= 1)


class TestSSTFXMF:
    @pytest.mark.skipif(
        not torch.cuda.is_available() and False,  # allow CPU test
        reason="Requires full dependencies"
    )
    def test_forward_pass(self, config, dummy_batch):
        """Test that the full model can process a batch end-to-end."""
        # Note: ClinicalBERT download may be skipped in test environments
        try:
            model = SSTFXMF(config)
        except Exception as e:
            pytest.skip(f"Model init requires transformers: {e}")

        outputs = model(dummy_batch)

        B = dummy_batch["feature_ids"].shape[0]
        assert outputs["hazard_logits"].shape == (B, config.model.num_time_bins)
        assert outputs["five_year_risk"].shape == (B,)
        assert outputs["concept_values"].shape == (B, config.model.num_concepts)
        assert outputs["gate_values"].shape == (B, config.model.num_modalities)

    def test_parameter_count(self, config):
        """Verify parameter counting utility."""
        try:
            model = SSTFXMF(config)
        except Exception as e:
            pytest.skip(f"Model init requires transformers: {e}")

        counts = model.get_num_parameters()
        assert "total_trainable" in counts
        assert counts["total_trainable"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
