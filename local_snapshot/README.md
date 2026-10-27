# Local ignored-content snapshot

This directory preserves the repository content that is intentionally excluded by
`.gitignore`, including the Python environment, raw TORGO data, downloaded archives,
experiment artifacts and checkpoints, caches, and temporary build outputs.

The snapshot was created on 2026-10-27 from these ignored roots:

- `.DS_Store`
- `.pytest_cache/`
- `.ruff_cache/`
- `.venv/`
- `artifacts/`
- `data/archives/`
- `data/raw/`
- Python `__pycache__/` directories under `scripts/`, `src/`, and `tests/`
- `src/timeentropy.egg-info/`
- `tmp/`

The compressed tar stream contains 67,284 entries and is split so every part remains
below GitHub's per-file Git LFS limit. Git LFS is required to clone the binary content.

Verify the downloaded parts from the repository root:

```bash
shasum -a 256 -c local_snapshot/SHA256SUMS
cat local_snapshot/timeentropy-ignored-20261027.tar.gz.part-* | gzip -t
```

Inspect the snapshot before extraction:

```bash
cat local_snapshot/timeentropy-ignored-20261027.tar.gz.part-* | tar -tzf -
```

Restore it into a clean checkout from the repository root:

```bash
cat local_snapshot/timeentropy-ignored-20261027.tar.gz.part-* | tar -xzf -
```

Extraction can overwrite files at the listed ignored paths, so use a clean checkout
when restoring the snapshot.
