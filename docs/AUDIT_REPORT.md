# FOCER-v2 legitimacy and leakage audit

Audit date: 2026-10-13  
Audited result archive SHA-256:
`8bd01be4d20c2475bede2ff89c541abb8da5844a8480990f6f0426b793e7ad6e`

## Verdict

The saved +8.6178 AP-point result is not fabricated by a metric bug, changed
DETR weights, overlapping image IDs, an incorrect checkpoint, or mismatched
test predictions. The exact predictions reproduce both reported mAP values.
The architecture and optimizer trained only the intended new router.

There is nevertheless an important dataset-split limitation. HDR4RTT contains
repeated video/scene content. The split has no byte-identical image across
train/validation/test, but many test frames have nearly identical scene and
camera content in train/validation. Consequently, the current result is valid
for this exact same-source split but is not sufficient by itself to establish
generalization to new scenes or cameras.

## Checks that passed

### Exact source and artifacts

- The source embedded in the result archive and the source package both have
  SHA-256 `6d84b8564...e01009b`.
- All 56 extracted source files match that immutable archive.
- The official DETR-R50 file matches the expected SHA-256
  `e632da11...784eadc`.
- The master HDR4RTT COCO annotation file matches the expected SHA-256
  `47ba1074...85b38b5`.
- Source compiles and all 15 included unit/regression tests pass under a current
  CPU PyTorch/torchvision installation.

### Parameter-freeze proof

- Saved model tensors: 495.
- New router tensors: 34, totaling exactly 1,251,328 parameters.
- Official DETR-derived tensors compared: 458/458 exactly equal, with zero
  changed values.
- Optimizer state entries: 34.
- Optimizer parameter IDs: 34 in one group.
- Therefore the optimizer did not update ResNet-50, the DETR encoder, original
  decoder self/cross-attention and FFNs, object queries, class head, or box head.

The trainable tensors are six copies each of `route_query`, `route_evidence`,
and `route_stop`, plus the shared two-layer `decoder.stop_encoder`.

### Split and execution order

- The immutable assignment exactly partitions all 1,596 image IDs into 1,264
  train, 168 validation, 150 test, and 14 guard images.
- The five regenerated split JSON hashes exactly match the recorded manifest.
- There is no filename or image-ID overlap among the four partitions.
- The selection log has 20 epochs. Its maximum validation mAP occurs at epoch
  11, matching `selection_record.json`.
- The final checkpoint is a fresh 11-epoch refit on the 1,432 train+validation
  images; its log contains no validation or test pass.
- Archive timestamps are in the expected order: selection completed at 13:13,
  refit at 14:11, FOCER test at 14:12, and fixed baseline test at 14:15 on
  2026-10-11.

### Saved result recomputation

The FOCER detections contain exactly 15,000 records (100 DETR queries for each
of 150 images). The fixed baseline contains 57,443 post-NMS detections. Both
cover exactly the 150 test IDs, with finite scores and boxes.

Independent `pycocotools` recomputation produces exactly:

| Category | Paper fixed AP | FOCER-v2 AP | Delta |
|---|---:|---:|---:|
| person | 43.0866 | 52.4401 | +9.3535 |
| bicycle | 33.3159 | 36.9049 | +3.5890 |
| car | 49.3943 | 52.7202 | +3.3259 |
| truck | 40.0627 | 58.2653 | +18.2027 |
| mean | 41.4649 | 50.0826 | +8.6178 |

### Rendering and evaluation fairness

Both methods use the same normalized HDR data, fixed stops 0 through 14, image
size, official detector, test annotations, category IDs, and COCO AP evaluator.
The paper baseline uses its exact NumPy/OpenCV preprocessing and class-wise NMS;
FOCER uses the differentiable PyTorch renderer and one prediction set without
late NMS, which is part of the architectural difference.

Across 30 sampled image/stop combinations, the two renderer implementations
had mean normalized-pixel absolute difference `2.76e-6`, maximum sampled p99
difference `1.81e-4`, and maximum individual difference `7.54e-4`. This is a
small implementation difference, not a plausible explanation for +8.62 AP.

## Findings that limit the claim

### 1. Same-scene similarity across splits

Full file SHA-256 found 1,572 unique images among 1,596 files. The 24 exact
duplicate groups are contained within a single split; **zero exact duplicate
groups cross a split**.

The stronger structural audit is less reassuring:

- 36/150 test images have a train/validation nearest similarity >= 0.99;
- 27/150 are >= 0.995;
- 14/150 are >= 0.999;
- the maximum is 0.9999748.

Visual inspection confirms repeated restaurant tables, people, lighting, and
camera viewpoints with small pose/time differences. See
[`assets/top_20_scene_similarity_pairs.jpg`](../assets/top_20_scene_similarity_pairs.jpg).

This is not literal duplicate-file leakage, but it can allow the router to
learn scene-specific exposure preferences. A publishable generalization claim
needs a new split grouped by complete video/scene/camera—not 100-frame blocks or
camera/day—or an external scene-disjoint test set.

### 2. Original full teacher-cache procedure touched test entries

The successful notebook reused a teacher cache built over the 1,596-image
master set and then copied only train, validation, and train+validation entries
into separate caches. Each cache entry is computed independently by a frozen
DETR; there are no learned weights or global cache statistics, and the training
loaders require an exact image-ID set. Therefore no test entry had a numerical
path into an optimizer step.

Still, the broad metadata phrase `test_images_accessed: false` is procedurally
too strong: test images and labels were read when the full cache was originally
built. The repository-level `run_clean.py` fixes this by building the parent
cache from train+validation only. This does not change the architecture.

### 3. Split construction is only partially self-verifying

The code proves ID disjointness, cardinality, category counts, and immutable
hashes. It stores human-readable source/temporal group names, but the released
master JSON does not contain source sequence metadata linking each ID back to a
group. Therefore the group-to-ID relationship cannot be independently proved
from the released annotations alone. The content audit above is the relevant
independent check, and it found same-scene overlap.

### 4. Statistical uncertainty remains

This is one seed and one 150-image test split. The result archive contains no
multi-seed confidence interval or scene-cluster bootstrap. The large point
estimate is encouraging, but the rare bicycle/truck classes and repeated
scenes make uncertainty reporting important.

## What is and is not safe to say

Safe:

> On the predefined 150-image HDR4RTT test split, with the official DETR-R50
> frozen, FOCER-v2 improved four-class mean COCO AP from 41.46 to 50.08
> (+8.62 points). The saved predictions independently reproduce this result.

Not yet safe:

> FOCER-v2 generalizes by +8.62 AP to unseen scenes/cameras.

Recommended next evidence: define scene/camera groups without consulting model
scores, freeze the protocol, train at least three seeds, evaluate the paper
baseline and FOCER once on that untouched split, and report a scene-clustered
paired bootstrap confidence interval.
