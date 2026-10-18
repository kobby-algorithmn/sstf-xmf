"""
SSTF-XMF Dataset Module
========================
PyTorch Dataset and DataLoader implementations for multi-modal EHR data.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from ..config import SSTFConfig

logger = logging.getLogger(__name__)


class CardiovascularEHRDataset(Dataset):
    """
    PyTorch Dataset for multi-modal cardiovascular EHR records.

    Each item returns a dictionary containing tensors for all modalities:
    structured features, tokenised clinical text, medication sequences,
    imaging biomarkers, and outcome labels.
    """

    def __init__(
        self,
        timelines: Dict[int, Dict],
        outcomes: Dict[int, Dict],
        config: SSTFConfig,
        split: str = "train",
    ):
        self.timelines = timelines
        self.outcomes = outcomes
        self.config = config
        self.split = split
        self.subject_ids = list(timelines.keys())

        # Lazy-load tokeniser for text modality
        self._tokeniser = None

        # Build feature index mapping
        self._build_feature_index()

        logger.info(
            f"Dataset initialised: {len(self.subject_ids)} patients "
            f"in {split} split"
        )

    @property
    def tokeniser(self):
        """Lazy-load the ClinicalBERT tokeniser."""
        if self._tokeniser is None:
            self._tokeniser = AutoTokenizer.from_pretrained(
                self.config.model.text_model_name
            )
        return self._tokeniser

    def _build_feature_index(self) -> None:
        """Construct feature-name to index mapping from all timelines."""
        feature_set = set()
        for timeline in self.timelines.values():
            for visit in timeline["visits"]:
                for lab in visit.get("labs", []):
                    feature_set.add(lab["feature_name"])
        self.feature_to_idx = {f: i for i, f in enumerate(sorted(feature_set))}
        logger.info(f"Feature vocabulary: {len(self.feature_to_idx)} unique features")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.subject_ids[idx]
        timeline = self.timelines[sid]
        outcome = self.outcomes.get(sid, {"event": 0, "time_bin": 0})

        item = self._encode_timeline(timeline)
        item["event"] = torch.tensor(outcome["event"], dtype=torch.float32)
        item["time_bin"] = torch.tensor(outcome.get("time_bin", 0), dtype=torch.long)
        item["subject_id"] = torch.tensor(sid, dtype=torch.long)

        # Demographic attributes for fairness audits
        item["demographic_group"] = torch.tensor(
            outcome.get("demographic_group", 0), dtype=torch.long
        )
        return item

    def _encode_timeline(self, timeline: Dict) -> Dict[str, torch.Tensor]:
        """Encode a patient timeline into tensor representations."""
        visits = timeline["visits"][: self.config.model.max_visits]
        max_v = self.config.model.max_visits
        max_e = self.config.model.max_events_per_visit
        max_m = self.config.model.max_medications_per_visit

        # Structured features
        feature_ids = torch.zeros((max_v, max_e), dtype=torch.long)
        values = torch.zeros((max_v, max_e), dtype=torch.float32)
        value_mask = torch.zeros((max_v, max_e), dtype=torch.bool)
        event_times = torch.zeros((max_v, max_e), dtype=torch.float32)

        # Visit-level
        visit_times = torch.zeros(max_v, dtype=torch.float32)
        visit_mask = torch.zeros(max_v, dtype=torch.bool)

        # Medications
        med_ids = torch.zeros((max_v, max_m), dtype=torch.long)
        med_features = torch.zeros((max_v, max_m, 3), dtype=torch.float32)
        med_mask = torch.zeros((max_v, max_m), dtype=torch.bool)

        # Reference time for temporal encoding
        if visits:
            t0 = visits[0]["date"]
        else:
            t0 = None

        for v_idx, visit in enumerate(visits):
            visit_mask[v_idx] = True
            if t0 is not None:
                hours_elapsed = (visit["date"] - t0).total_seconds() / 3600.0
                visit_times[v_idx] = float(hours_elapsed)

            # Encode laboratory events
            labs = visit.get("labs", [])[:max_e]
            for e_idx, lab in enumerate(labs):
                fname = lab["feature_name"]
                if fname in self.feature_to_idx:
                    feature_ids[v_idx, e_idx] = self.feature_to_idx[fname]
                    values[v_idx, e_idx] = float(lab["value_scaled"])
                    value_mask[v_idx, e_idx] = True

            # Encode medications
            meds = visit.get("medications", [])[:max_m]
            for m_idx, med in enumerate(meds):
                # In production: map drug name to ATC level-4 integer id
                med_ids[v_idx, m_idx] = 0  # placeholder
                med_mask[v_idx, m_idx] = True

        # Tokenise most recent clinical note
        text_ids, text_mask = self._tokenise_notes(visits)

        # Imaging features (placeholder - extract from imaging reports in practice)
        imaging_features = torch.zeros(
            self.config.model.imaging_feature_dim, dtype=torch.float32
        )

        return {
            "feature_ids": feature_ids,
            "values": values,
            "value_mask": value_mask,
            "event_times": event_times,
            "visit_times": visit_times,
            "visit_mask": visit_mask,
            "med_ids": med_ids,
            "med_features": med_features,
            "med_mask": med_mask,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "imaging_features": imaging_features,
        }

    def _tokenise_notes(
        self, visits: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenise clinical notes from the most recent visit with notes."""
        max_len = self.config.model.max_text_tokens

        # Find most recent visit with notes
        text = ""
        for visit in reversed(visits):
            notes = visit.get("notes", [])
            if notes:
                text = " ".join(notes[-3:])  # concat last three notes
                break

        if not text:
            return (
                torch.zeros(max_len, dtype=torch.long),
                torch.zeros(max_len, dtype=torch.bool),
            )

        encoded = self.tokeniser(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0).bool(),
        )


def create_dataloaders(
    train_dataset: CardiovascularEHRDataset,
    val_dataset: CardiovascularEHRDataset,
    test_dataset: CardiovascularEHRDataset,
    config: SSTFConfig,
    phase: str = "finetune",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders for train, validation, and test splits."""
    if phase == "pretrain":
        batch_size = config.pretrain.batch_size
    else:
        batch_size = config.finetune.batch_size

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.persistent_workers and config.data.num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader


def save_dataset_artefacts(
    dataset: CardiovascularEHRDataset, path: str
) -> None:
    """Persist dataset artefacts for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "feature_to_idx": dataset.feature_to_idx,
            "subject_ids": dataset.subject_ids,
            "split": dataset.split,
        }, f)
    logger.info(f"Dataset artefacts saved to {path}")
