# FOCER-v2 design and research contract

## Hypothesis

The fixed stops already reveal most annotated objects somewhere. The research
question is therefore not which global exposure set to generate, but whether
each object query can retrieve the most useful evidence from the existing fixed
branches before classification and box regression.

## Factorized object-conditioned routing

For each decoder layer, query `q` attends to each exposure memory `k`
independently. Branches are folded into the batch dimension during spatial
attention, so tokens from different exposures cannot interact there. Only
after producing one evidence vector per query/exposure does the router combine
them.

The route also receives a continuous numeric stop embedding and a query-local
radiometric reliability value. Reliability is obtained by weighting a soft
darkness/clipping map with that query's spatial attention. It is a physical
prior, not a supervision label.

Sparse top-k routing is straight-through: the prediction path uses the
renormalized top-k weights, while the backward path retains the dense-softmax
gradient so an excluded exposure can later enter the route.

## Frozen-teacher routing targets

For every labelled object and every fixed exposure, an untouched official
DETR-R50 is matched to ground truth with its standard Hungarian matcher. The
cached utility is:

```text
utility = log P(correct class) - 5 * L1(box) - 2 * (1 - GIoU(box))
```

The utilities become a soft target distribution across exposures. Target
entropy defines confidence: if the teacher considers every exposure equally
useful, the target contributes little or nothing. The loss is applied to the
dense routing probabilities of the student query matched to the same object.

This is privileged training supervision, not inference-time computation. At
test time the frozen teacher and labels are absent.

## Parameter contract

Trainable:

- six decoder layers' query, evidence, and stop routing projections;
- the continuous fixed-stop encoder.

Frozen and held in evaluation mode:

- ResNet-50 backbone and input projection;
- DETR encoder;
- DETR object-query embeddings;
- all copied decoder self-attention, cross-attention, feed-forward, norm, and
  dropout modules;
- DETR class and box heads.

The code validates this contract before building the optimizer. This isolation
is essential: any improvement can then be attributed to routing much more
cleanly than in V1, where about 9.65M pretrained detector parameters were also
fine-tuned.

## Curriculum and imbalance control

The default 20-epoch schedule is:

1. epochs 1-3: all 15 branches, dense soft route;
2. epochs 4-6: top-5 route;
3. epochs 7-20: top-3 deployment route.

Temperature linearly anneals from 1.5 to 0.7. Validation always uses top-3 at
0.7, preventing checkpoint selection from benefiting from an easier temporary
training budget.

Training uses capped inverse-square-root image sampling over person, bicycle,
car, and truck. The image weight is the largest weight of a category present
in that image and is capped at 4x. Validation and test loaders are never
resampled.

## Required comparisons

All methods must use the same train/validation/test split, fixed stops,
official detector weights, seed set, evaluation categories, and COCO AP code.

Minimum ablations:

1. original paper fixed-exposure detectors plus NMS;
2. V2 without teacher loss (`--teacher-routing-weight 0`);
3. V2 without radiometric prior (`--radiometric-strength 0`);
4. V2 without balanced sampling (`--no-balanced-sampling`);
5. dense, top-5, top-3, and top-1 routing;
6. V2 with shuffled object/exposure teacher targets as a negative control.

The strongest causal comparison is frozen detector + identical training
pipeline, changing only the routing component under study.

## Go/no-go gates

Before a full run:

- source tests and both data preflights pass;
- startup reports zero non-router trainable parameters;
- teacher caches match their exact annotations and detector fingerprint;
- a full-resolution optimizer-step smoke test has finite router gradients and
  no frozen gradients.

During training:

- teacher confidence is nonzero for a meaningful number of objects;
- teacher top-1 agreement rises above random (`1/15 = 6.67%`);
- routes do not collapse to the same stops for every object;
- validation detection loss improves under the final top-3 configuration.

Only after choosing the checkpoint from validation may the untouched paper test
set be evaluated. Test AP must never select epochs or hyperparameters.

## Limitations

- Fixed exposure branches still require repeated backbone/encoder computation.
  V2 removes repeated prediction sets and decoder heads, not the cost of all
  exposure features.
- Frozen DETR improves attribution but may cap adaptation to HDRDCL appearance.
- Teacher utility inherits errors and biases of the official DETR; entropy
  confidence reduces but does not eliminate this limitation.
- HDRDCL validation has only 53 images and the paper test set has 125 images.
  Report per-image paired bootstrap uncertainty, not only a point estimate.
- FOCER is a research hypothesis until controlled experiments support it.
