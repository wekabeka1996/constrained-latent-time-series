# Public Evidence Snapshot

This directory contains small, sanitized publication evidence summaries.
Full reproduction outputs are generated locally under `results/reproduction_*`
and are intentionally not committed.

To regenerate the full evidence:

```bash
python -m src.reproduce --config configs/demo.yaml --stages 1,2,3,4,5,6,7
```
