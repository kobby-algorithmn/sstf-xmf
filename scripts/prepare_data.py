"""
Data Preparation Script
=========================
End-to-end data preparation for MIMIC-IV cardiovascular cohort.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
"""

from __future__ import annotations
import argparse
import logging
import pickle
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SSTFConfig, get_default_config
from src.data.mimic_preprocessor import MIMICIVPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Prepare MIMIC-IV cardiovascular dataset")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="./data/processed",
                        help="Output directory for processed data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    config = SSTFConfig.from_yaml(args.config) if args.config else get_default_config()

    logger.info("=" * 60)
    logger.info("MIMIC-IV Cardiovascular Cohort Preparation")
    logger.info("=" * 60)

    preprocessor = MIMICIVPreprocessor(config)
    timelines, outcomes = preprocessor.run_full_pipeline()

    # Save artefacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "timelines.pkl", "wb") as f:
        pickle.dump(timelines, f)
    outcomes.to_csv(output_dir / "outcomes.csv", index=False)

    # Save preprocessor state (scalers, feature stats)
    with open(output_dir / "preprocessor_state.pkl", "wb") as f:
        pickle.dump({
            "scalers": preprocessor.scalers,
            "feature_stats": preprocessor.feature_stats,
        }, f)

    logger.info(f"Data preparation complete. Artefacts saved to {output_dir}")
    logger.info(f"  - Patients: {len(timelines)}")
    logger.info(f"  - Events: {outcomes['event'].sum()}")


if __name__ == "__main__":
    main()
