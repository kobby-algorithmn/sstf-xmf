"""
MIMIC-IV Data Pre-processing Pipeline
======================================
Extracts cardiovascular cohorts and builds longitudinal patient timelines
from MIMIC-IV database files for the SSTF-XMF framework.

Author: MSc Dissertation Research
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from ..config import SSTFConfig

logger = logging.getLogger(__name__)


# Cardiovascular ICD-10 code prefixes for MACE composite outcome
MACE_ICD10_PREFIXES = {
    "I21": "acute_myocardial_infarction",
    "I22": "subsequent_myocardial_infarction",
    "I50": "heart_failure",
    "I60": "subarachnoid_haemorrhage",
    "I61": "intracerebral_haemorrhage",
    "I63": "cerebral_infarction",
    "I46": "cardiac_arrest",
    "I71": "aortic_aneurysm_dissection",
}

# Key cardiovascular laboratory tests (MIMIC-IV itemid -> feature name)
CARDIOVASCULAR_LAB_ITEMIDS = {
    50912: "creatinine",
    50902: "chloride",
    50882: "bicarbonate",
    50862: "albumin",
    51006: "blood_urea_nitrogen",
    51265: "platelet_count",
    51301: "white_blood_cell_count",
    50893: "calcium_total",
    50931: "glucose",
    50983: "sodium",
    50971: "potassium",
    51222: "haemoglobin",
    51279: "red_blood_cell_count",
    50852: "activated_partial_thromboplastin_time",
    51003: "troponin_t",
    50963: "b_type_natriuretic_peptide",
    50907: "cholesterol_total",
    50904: "cholesterol_hdl",
    50906: "cholesterol_ldl",
    51144: "haemoglobin_a1c",
    50889: "c_reactive_protein",
    50960: "magnesium",
    51237: "international_normalised_ratio",
    50813: "lactate",
    50811: "haemoglobin_arterial",
}

# Physiologically plausible ranges for quality assurance
LAB_VALID_RANGES = {
    "creatinine": (0.1, 15.0),
    "glucose": (20.0, 800.0),
    "sodium": (110.0, 180.0),
    "potassium": (1.5, 9.0),
    "haemoglobin": (3.0, 22.0),
    "troponin_t": (0.0, 50.0),
    "cholesterol_total": (50.0, 500.0),
    "cholesterol_hdl": (10.0, 200.0),
    "cholesterol_ldl": (20.0, 400.0),
    "haemoglobin_a1c": (3.0, 20.0),
    "c_reactive_protein": (0.0, 500.0),
}


class MIMICIVPreprocessor:
    """
    Pre-processor for MIMIC-IV cardiovascular cohort extraction.

    Implements the cohort selection criteria, temporal alignment, and
    feature normalisation pipeline described in the dissertation methodology.
    """

    def __init__(self, config: SSTFConfig):
        self.config = config
        self.data_path = Path(config.data.mimic_iv_path)
        self.scalers: Dict[str, RobustScaler] = {}
        self.feature_stats: Dict[str, Dict] = {}

    def extract_cardiovascular_cohort(self) -> pd.DataFrame:
        """
        Extract eligible patients meeting the cardiovascular cohort criteria:
            - Age >= 40 years at index encounter
            - At least 2 encounters
            - At least 365 days of follow-up
            - At least 10 recorded clinical events
        """
        logger.info("Extracting cardiovascular cohort from MIMIC-IV")

        patients = pd.read_csv(
            self.data_path / "hosp" / "patients.csv.gz",
            compression="gzip"
        )
        admissions = pd.read_csv(
            self.data_path / "hosp" / "admissions.csv.gz",
            compression="gzip",
            parse_dates=["admittime", "dischtime"]
        )

        # Age filter
        patients = patients[patients["anchor_age"] >= self.config.data.min_age]

        # Encounter count filter
        encounter_counts = admissions.groupby("subject_id")["hadm_id"].nunique()
        eligible_ids = encounter_counts[
            encounter_counts >= self.config.data.min_encounters
        ].index

        # Follow-up duration filter
        followup = admissions.groupby("subject_id").agg(
            first_admit=("admittime", "min"),
            last_admit=("admittime", "max")
        )
        followup["followup_days"] = (
            followup["last_admit"] - followup["first_admit"]
        ).dt.days
        long_followup_ids = followup[
            followup["followup_days"] >= self.config.data.min_followup_days
        ].index

        final_ids = set(eligible_ids) & set(long_followup_ids)
        cohort = patients[patients["subject_id"].isin(final_ids)].copy()

        logger.info(
            f"Cardiovascular cohort extracted: {len(cohort)} patients "
            f"({len(cohort) / len(patients) * 100:.1f}% of source cohort)"
        )
        return cohort

    def extract_outcomes(
        self, cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract MACE composite outcome events with timestamps.
        Returns per-patient event indicator and time-to-event.
        """
        diagnoses = pd.read_csv(
            self.data_path / "hosp" / "diagnoses_icd.csv.gz",
            compression="gzip"
        )
        admissions = pd.read_csv(
            self.data_path / "hosp" / "admissions.csv.gz",
            compression="gzip",
            parse_dates=["admittime"]
        )

        subject_ids = set(cohort["subject_id"])
        diagnoses = diagnoses[diagnoses["subject_id"].isin(subject_ids)]

        # Mark MACE diagnoses
        diagnoses["icd_prefix"] = diagnoses["icd_code"].str[:3]
        mace_dx = diagnoses[
            diagnoses["icd_prefix"].isin(MACE_ICD10_PREFIXES.keys())
        ]
        mace_dx = mace_dx.merge(
            admissions[["hadm_id", "admittime"]],
            on="hadm_id",
            how="left"
        )

        # First MACE event per patient
        first_mace = mace_dx.groupby("subject_id").agg(
            mace_time=("admittime", "min"),
            mace_type=("icd_prefix", "first")
        ).reset_index()

        # Merge with cohort, set event indicator
        outcomes = cohort[["subject_id"]].merge(
            first_mace, on="subject_id", how="left"
        )
        outcomes["event"] = outcomes["mace_time"].notna().astype(int)

        logger.info(
            f"MACE outcomes: {outcomes['event'].sum()} events "
            f"({outcomes['event'].mean() * 100:.1f}% event rate)"
        )
        return outcomes

    def extract_longitudinal_labs(
        self, subject_ids: List[int]
    ) -> pd.DataFrame:
        """Extract and normalise longitudinal laboratory results."""
        logger.info("Extracting longitudinal laboratory data")

        labs = pd.read_csv(
            self.data_path / "hosp" / "labevents.csv.gz",
            compression="gzip",
            usecols=["subject_id", "hadm_id", "charttime",
                     "itemid", "valuenum"],
            parse_dates=["charttime"]
        )
        labs = labs[labs["subject_id"].isin(subject_ids)]
        labs = labs[labs["itemid"].isin(CARDIOVASCULAR_LAB_ITEMIDS.keys())]
        labs["feature_name"] = labs["itemid"].map(CARDIOVASCULAR_LAB_ITEMIDS)

        # Quality assurance: remove implausible values
        labs = self._apply_clinical_ranges(labs)

        # Robust scaling per feature
        labs["value_scaled"] = self._robust_scale_features(labs)

        return labs

    def _apply_clinical_ranges(self, labs: pd.DataFrame) -> pd.DataFrame:
        """Remove physiologically implausible laboratory values."""
        before = len(labs)
        for feature, (lo, hi) in LAB_VALID_RANGES.items():
            mask = labs["feature_name"] == feature
            invalid = mask & ((labs["valuenum"] < lo) | (labs["valuenum"] > hi))
            labs.loc[invalid, "valuenum"] = np.nan
        labs = labs.dropna(subset=["valuenum"])
        logger.info(
            f"Quality filter: removed {before - len(labs)} implausible values "
            f"({(before - len(labs)) / before * 100:.2f}%)"
        )
        return labs

    def _robust_scale_features(self, labs: pd.DataFrame) -> np.ndarray:
        """Apply feature-specific robust scaling."""
        scaled = np.zeros(len(labs))
        for feature in labs["feature_name"].unique():
            mask = labs["feature_name"] == feature
            vals = labs.loc[mask, "valuenum"].values.reshape(-1, 1)
            scaler = RobustScaler(quantile_range=(5.0, 95.0))
            scaled[mask.values] = scaler.fit_transform(vals).flatten()
            self.scalers[feature] = scaler
            self.feature_stats[feature] = {
                "n": int(mask.sum()),
                "median": float(np.median(vals)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25))
            }
        return scaled

    def extract_medications(
        self, subject_ids: List[int]
    ) -> pd.DataFrame:
        """Extract medication prescription sequences with ATC mapping."""
        logger.info("Extracting medication sequences")
        prescriptions = pd.read_csv(
            self.data_path / "hosp" / "prescriptions.csv.gz",
            compression="gzip",
            parse_dates=["starttime", "stoptime"]
        )
        prescriptions = prescriptions[
            prescriptions["subject_id"].isin(subject_ids)
        ]
        # In production, map drug names to ATC codes via external reference
        return prescriptions[[
            "subject_id", "hadm_id", "starttime", "stoptime",
            "drug", "drug_type", "dose_val_rx", "dose_unit_rx"
        ]]

    def extract_clinical_notes(
        self, subject_ids: List[int], max_notes_per_patient: int = 50
    ) -> pd.DataFrame:
        """Extract discharge summaries and clinical narratives."""
        logger.info("Extracting clinical notes")
        notes_path = self.data_path / "note" / "discharge.csv.gz"
        if not notes_path.exists():
            logger.warning("Note data not found - continuing without text modality")
            return pd.DataFrame(columns=["subject_id", "charttime", "text"])

        notes = pd.read_csv(notes_path, compression="gzip",
                            parse_dates=["charttime"])
        notes = notes[notes["subject_id"].isin(subject_ids)]
        # Retain only the most recent notes per patient
        notes = (notes.sort_values("charttime", ascending=False)
                      .groupby("subject_id")
                      .head(max_notes_per_patient))
        return notes[["subject_id", "charttime", "text"]]

    def build_patient_timelines(
        self,
        cohort: pd.DataFrame,
        labs: pd.DataFrame,
        medications: pd.DataFrame,
        notes: pd.DataFrame,
    ) -> Dict[int, Dict]:
        """
        Organise all clinical events into per-patient longitudinal timelines.
        Returns a dictionary mapping subject_id to the patient's timeline.
        """
        logger.info("Building patient timelines")
        timelines: Dict[int, Dict] = {}

        for sid in cohort["subject_id"].unique():
            p_labs = labs[labs["subject_id"] == sid].sort_values("charttime")
            p_meds = medications[medications["subject_id"] == sid].sort_values("starttime")
            p_notes = notes[notes["subject_id"] == sid].sort_values("charttime")

            # Group lab events into daily visits
            visits: List[Dict] = []
            if len(p_labs) > 0:
                for date, group in p_labs.groupby(p_labs["charttime"].dt.date):
                    visits.append({
                        "date": pd.Timestamp(date),
                        "labs": group[
                            ["feature_name", "value_scaled", "charttime"]
                        ].to_dict("records"),
                    })

            # Attach medications to nearest visit
            for visit in visits:
                visit_date = visit["date"]
                window_start = visit_date - timedelta(days=1)
                window_end = visit_date + timedelta(days=1)
                active_meds = p_meds[
                    (p_meds["starttime"] >= window_start)
                    & (p_meds["starttime"] <= window_end)
                ]
                visit["medications"] = active_meds.to_dict("records")

            # Attach notes to nearest visit
            for visit in visits:
                visit_date = visit["date"]
                nearby_notes = p_notes[
                    (p_notes["charttime"] >= visit_date - timedelta(days=1))
                    & (p_notes["charttime"] <= visit_date + timedelta(days=1))
                ]
                visit["notes"] = nearby_notes["text"].tolist()

            # Filter by minimum events criterion
            total_events = sum(
                len(v["labs"]) + len(v["medications"]) for v in visits
            )
            if total_events >= self.config.data.min_events_per_patient:
                timelines[sid] = {
                    "subject_id": sid,
                    "visits": visits[:self.config.model.max_visits],
                    "num_visits": len(visits),
                    "total_events": total_events,
                }

        logger.info(f"Built {len(timelines)} patient timelines")
        return timelines

    def run_full_pipeline(self) -> Tuple[Dict[int, Dict], pd.DataFrame]:
        """Execute the complete pre-processing pipeline end-to-end."""
        cohort = self.extract_cardiovascular_cohort()
        outcomes = self.extract_outcomes(cohort)
        subject_ids = cohort["subject_id"].tolist()
        labs = self.extract_longitudinal_labs(subject_ids)
        meds = self.extract_medications(subject_ids)
        notes = self.extract_clinical_notes(subject_ids)
        timelines = self.build_patient_timelines(cohort, labs, meds, notes)
        return timelines, outcomes
