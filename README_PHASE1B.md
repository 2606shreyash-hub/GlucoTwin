# GlucoTwin Phase 1B — Causal Feature Engineering

Add these files to the existing GlucoTwin project root. The feature engine uses only information available at or before prediction time `t`.

Files:
- `src/features/__init__.py`
- `src/features/feature_engineering.py`
- `tests/test_features.py`
- `scripts/build_features.py`

The build script expects the existing `src.labels.hypoglycemia.make_hypoglycemia_label` function from Phase 1.
