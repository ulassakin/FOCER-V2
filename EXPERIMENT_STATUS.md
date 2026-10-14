# FOCER-v2 experiment status

Last updated: 2026-10-13

## Completed

- Implemented fixed 15-stop, per-query top-3 exposure routing in every DETR
  decoder layer.
- Implemented strict detector freezing and optimizer safety guards.
- Implemented frozen-teacher per-object exposure targets, confidence weighting,
  sparse-routing curriculum, and capped rare-class sampling.
- Completed selection training on 1,264 HDR4RTT images with validation on 168
  images.
- Selected epoch 11 by validation pooled four-category COCO mAP.
- Freshly refitted the router for 11 epochs on 1,432 train+validation images.
- Evaluated once on the predefined 150-image test split.
- Independently recomputed the recorded COCO AP values.
- Verified 458/458 inherited DETR tensors remained bit-identical to the official
  checkpoint and the optimizer contained only 34 router tensors.
- Passed all 15 unit/protocol tests and artifact integrity checks.

## Recorded result

- Paper fixed-exposure baseline: `41.4649` mean AP.
- FOCER-v2: `50.0826` mean AP.
- Difference: `+8.6178` AP points.

## Required follow-up

The current result is same-source evidence. HDR4RTT contains closely related
frames and camera views across the fixed split, although no byte-identical file
crosses a split. Before making a broad generalization claim:

1. define a fully scene/camera-disjoint protocol;
2. run multiple seeds;
3. report paired, scene-clustered uncertainty;
4. run teacher, radiometric-prior, global-routing, and routing-budget ablations.

See `docs/AUDIT_REPORT.md` for the complete legitimacy and leakage audit.
