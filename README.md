# SSTF-XMF (prototype)

**Self-Supervised Temporal Transformer with Explainable Multi-Modal Fusion**
for cardiovascular risk prediction from longitudinal electronic health records.

> **Prototype.** This is the code skeleton of an MSc research framework. The
> model architecture, losses, training loops, evaluation and explainability
> modules are written, and the unit tests run. The model has **not** been
> trained or evaluated on real patient data. There are no results, no trained
> weights and no clinical validation. It is not for clinical use.

## Status

| Part | Status |
|---|---|
| Model components (Time2Vec, structured encoder, hierarchical temporal transformer, cross-modal attention, concept bottleneck) | Built; unit tests pass |
| Full model forward pass (needs `transformers` and Bio_ClinicalBERT) | Built; see test results below |
| MIMIC-IV preprocessing pipeline | Written; not run (needs credentialed PhysioNet access) |
| Self-supervised pre-training and fine-tuning | Written; not run |
| Evaluation, fairness metrics, explanations | Written; not run |
| FHIR deployment service | Written; not run |
| Training on real data, reported metrics | Planned |

### Test results

Run on 19 October 2026 (Python 3.13, PyTorch 2.14 CPU, `transformers` with
Bio_ClinicalBERT downloaded): **7 passed**, including the full-model forward
pass and the parameter-count test.

- Without `transformers` installed, the two full-model tests are skipped
  (5 passed, 2 skipped).
- Fixed on 19 October 2026: the forward-pass test's dummy text IDs went up to
  30,000, beyond Bio_ClinicalBERT's 28,996-token vocabulary, which caused an
  `IndexError`. The test now draws IDs within the vocabulary.

## What the design intends

1. **Hierarchical temporal transformer.** A within-visit transformer, then an
   across-visit transformer with continuous-time (Time2Vec) encodings, for
   irregularly sampled clinical timelines.
2. **Self-supervised pre-training.** Masked clinical event modelling plus
   temporal contrastive learning, on unlabelled EHR sequences.
3. **Explainable multi-modal fusion.** Structured labs and vitals, clinical
   notes (Bio_ClinicalBERT), medications and imaging features are fused with
   cross-modal attention and modality gating. A concept-bottleneck layer of 15
   clinical risk concepts sits before the survival-style risk head.
4. **Fairness-aware fine-tuning.** A fairness penalty term in the loss.

These are design goals. None of them has been demonstrated on real data yet.

## How to run the tests

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install torch numpy pandas scikit-learn scipy pyyaml tqdm requests pytest
pip install transformers         # optional: enables the two full-model tests
python -m pytest tests/ -v
```

`requirements.txt` lists the full set of dependencies, including optional ONNX
and FHIR packages.

## Using real data (Planned)

The pipeline is written for MIMIC-IV, which needs credentialed access from
PhysioNet. Patient data must never be committed to this repository:
`data/` and `outputs/` are git-ignored.

```bash
python scripts/prepare_data.py --config configs/default.yaml
python -m src.main pretrain --config configs/default.yaml
python -m src.main finetune --config configs/default.yaml --pretrained outputs/<run>/pretrain_best.pt
python -m src.main evaluate --config configs/default.yaml --checkpoint outputs/<run>/finetune_best.pt
```

## Folder layout

```
sstf-xmf/
├── configs/default.yaml        # model, training and data settings
├── scripts/prepare_data.py     # data preparation entry point
├── src/
│   ├── config.py               # configuration dataclasses
│   ├── main.py                 # CLI: pretrain / finetune / evaluate / predict
│   ├── data/                   # MIMIC-IV preprocessing, PyTorch dataset
│   ├── models/sstf_xmf.py      # model architecture
│   ├── training/               # losses and trainer
│   ├── evaluation/             # clinical metrics
│   ├── explainability/         # attention, SHAP and concept explanations
│   └── deployment/             # FHIR inference service
├── tests/test_model.py         # unit tests
├── requirements.txt
└── setup.py
```

## Notes

- Developed with AI assistance (Claude).
- The accompanying dissertation is not included in this repository.

## Licence

All rights reserved. Shared for academic reference only.
