# Locked TORGO cohort metadata

- `torgo_manifest_v1.csv` is derived from the surviving 8,216-row index by retaining
  head-mounted-microphone rows with `usable_prompt=True` and `duration_sec>=0.05`.
- `torgo_folds_v1.csv` expands each retained sample across the five recovered
  train/validation/test speaker folds.
- `torgo_manifest_v1.meta.json` records the source and derived SHA-256 hashes.

The `path` column is repository-relative and becomes valid after selective archive recovery.
The `recovered_path` column is retained only as forensic provenance and is expected to be
stale. Model code must identify samples by `sample_id` and `rel_path`, never by row number.
