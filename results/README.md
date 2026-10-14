# Reproducibility records

This directory contains small, Git-friendly records from the audited successful
run. Large checkpoints, teacher caches, raw predictions, and datasets are not
stored in the repository.

- `FINAL_TEST_COMPARISON.json`: exact paper-baseline and FOCER-v2 AP values;
- `run_configuration.json`: source, detector, annotation, split, and protocol
  identity;
- `selection_record.json`: validation-selected epoch count and checkpoint
  identity.

The complete artifact was independently checked for source integrity, split-ID
isolation, optimizer scope, frozen DETR tensor equality, and COCO AP
recomputation. See `../docs/AUDIT_REPORT.md` for limitations and interpretation.
