"""Measure object-level fixed-exposure routing headroom before long training.

For every ground-truth object, this diagnostic finds the fixed exposure whose
pretrained DETR prediction has the largest same-class score x IoU.  It compares
an optimistic per-object exposure oracle with the best single exposure for each
scene and reports how often objects in the same scene prefer different stops.

This script is a diagnostic only: it uses ground truth to estimate headroom and
must never be reported as a deployable detector result.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.hdr_mat import HDRMatDetection, collate_hdr
from focer_main import (
    OFFICIAL_DETR,
    base_detr_arguments,
    load_checkpoint,
    parse_stops,
    select_device,
)
from models import build_model
from models.focer import FixedExposureRenderer
from util.box_ops import box_cxcywh_to_xyxy, box_iou


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--mat-dir", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", default=OFFICIAL_DETR)
    parser.add_argument("--stops", type=parse_stops, default=parse_stops("0:14:1"))
    parser.add_argument("--short-side", type=int, default=800)
    parser.add_argument("--max-side", type=int, default=1333)
    parser.add_argument("--branch-chunk-size", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--classes", default="1,2,3,8")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--hdr-scale", type=float, default=255.0)
    parser.add_argument("--mat-key", default="rgb_dem")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("oracle_output/routing_headroom.json"))
    config = parser.parse_args()
    config.classes = tuple(int(item) for item in config.classes.split(",") if item)
    if not config.classes or config.branch_chunk_size <= 0:
        parser.error("classes and --branch-chunk-size must be nonempty/positive")
    if not 0 <= config.score_threshold <= 1 or not 0 <= config.iou_threshold <= 1:
        parser.error("score and IoU thresholds must lie in [0, 1]")
    return config


def ground_truth(target, categories):
    keep = torch.tensor(
        [int(label) in categories for label in target["labels"]], dtype=torch.bool
    )
    boxes = box_cxcywh_to_xyxy(target["boxes"][keep])
    height, width = target["orig_size"].tolist()
    boxes = boxes * boxes.new_tensor([width, height, width, height])
    return boxes, target["labels"][keep]


@torch.no_grad()
def detect_branches(detector, postprocessor, renderer, hdr, target, chunk_size):
    rendered = renderer(hdr[None]).squeeze(0)
    target_size = target["orig_size"].to(hdr.device)[None]
    results = []
    for start in range(0, rendered.shape[0], chunk_size):
        images = list(rendered[start:start + chunk_size])
        outputs = detector(images)
        results.extend(postprocessor(outputs, target_size.expand(len(images), -1)))
    return results


def object_branch_matrices(branch_results, gt_boxes, gt_labels, score_threshold,
                           iou_threshold):
    objects = gt_boxes.shape[0]
    branches = len(branch_results)
    utility = torch.zeros(objects, branches)
    detected = torch.zeros(objects, branches, dtype=torch.bool)
    matched_score = torch.zeros(objects, branches)
    matched_iou = torch.zeros(objects, branches)
    for branch, predictions in enumerate(branch_results):
        boxes = predictions["boxes"].cpu()
        scores = predictions["scores"].cpu()
        labels = predictions["labels"].cpu()
        if not boxes.numel() or not objects:
            continue
        ious, _ = box_iou(gt_boxes.cpu(), boxes)
        for obj in range(objects):
            same_class = labels == gt_labels[obj].cpu()
            if not same_class.any():
                continue
            class_scores = scores[same_class]
            class_ious = ious[obj, same_class]
            values = class_scores * class_ious
            best = int(values.argmax())
            utility[obj, branch] = values[best]
            matched_score[obj, branch] = class_scores[best]
            matched_iou[obj, branch] = class_ious[best]
            detected[obj, branch] = (
                class_scores[best] >= score_threshold
                and class_ious[best] >= iou_threshold
            )
    return utility, detected, matched_score, matched_iou


def percentile(values, quantile):
    return None if not values else float(np.quantile(np.asarray(values), quantile))


def summarize(records, stops, thresholds, image_count):
    object_records = [item for image in records for item in image["objects"]]
    total_objects = len(object_records)
    oracle_detected = sum(item["detected_any"] for item in object_records)
    best_global_detected = sum(image["best_global_detected"] for image in records)
    multi = [image for image in records if len(image["objects"]) >= 2]
    spreads = [image["preferred_stop_spread"] for image in multi]
    branch_histogram = Counter(str(item["preferred_stop"]) for item in object_records)
    conflicting = sum(
        image["oracle_detected"] > image["best_global_detected"] for image in multi
    )
    return {
        "diagnostic": "ground_truth_informed_object_exposure_oracle_not_a_detector_result",
        "images": image_count,
        "objects": total_objects,
        "fixed_stops": list(map(float, stops)),
        "thresholds": thresholds,
        "per_object_oracle_recall": oracle_detected / max(total_objects, 1),
        "best_single_exposure_per_scene_recall": best_global_detected / max(total_objects, 1),
        "oracle_minus_best_scene_exposure_recall": (
            oracle_detected - best_global_detected
        ) / max(total_objects, 1),
        "multi_object_images": len(multi),
        "multi_object_images_where_oracle_beats_every_single_exposure": conflicting,
        "conflicting_exposure_fraction": conflicting / max(len(multi), 1),
        "preferred_stop_spread": {
            "median": percentile(spreads, 0.5),
            "p75": percentile(spreads, 0.75),
            "p90": percentile(spreads, 0.9),
            "fraction_at_least_2_stops": sum(value >= 2 for value in spreads) / max(len(spreads), 1),
            "fraction_at_least_3_stops": sum(value >= 3 for value in spreads) / max(len(spreads), 1),
        },
        "preferred_stop_histogram": dict(sorted(branch_histogram.items(), key=lambda item: float(item[0]))),
        "images_detail": records,
    }


def main():
    config = arguments()
    device = select_device(config.device)
    dataset = HDRMatDetection(
        config.annotations,
        config.mat_dir,
        short_side=config.short_side,
        max_side=config.max_side,
        mat_key=config.mat_key,
        hdr_scale=config.hdr_scale,
    )
    dataset.validate_paths()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_hdr)
    detector, _, postprocessors = build_model(base_detr_arguments(device))
    checkpoint = load_checkpoint(config.detector_checkpoint)
    detector.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    detector.to(device).eval()
    renderer = FixedExposureRenderer(
        config.stops, short_side=config.short_side, max_side=config.max_side
    ).to(device)

    records = []
    limit = min(config.limit_images or len(loader), len(loader))
    for index, (images, targets) in enumerate(loader, start=1):
        if index > limit:
            break
        hdr = images[0].to(device)
        target = targets[0]
        gt_boxes, gt_labels = ground_truth(target, config.classes)
        branch_results = detect_branches(
            detector, postprocessors["bbox"], renderer, hdr, target,
            config.branch_chunk_size,
        )
        utility, detected, scores, ious = object_branch_matrices(
            branch_results, gt_boxes, gt_labels,
            config.score_threshold, config.iou_threshold,
        )
        preferred = utility.argmax(dim=1) if utility.numel() else torch.empty(0, dtype=torch.long)
        objects = []
        for obj in range(gt_boxes.shape[0]):
            branch = int(preferred[obj])
            objects.append({
                "category_id": int(gt_labels[obj]),
                "preferred_branch": branch,
                "preferred_stop": float(config.stops[branch]),
                "utility": float(utility[obj, branch]),
                "matched_score": float(scores[obj, branch]),
                "matched_iou": float(ious[obj, branch]),
                "detected_any": bool(detected[obj].any()),
                "detected_stops": [
                    float(config.stops[b]) for b in range(len(config.stops))
                    if bool(detected[obj, b])
                ],
            })
        branch_counts = detected.sum(dim=0) if detected.numel() else torch.zeros(len(config.stops))
        preferred_stops = [item["preferred_stop"] for item in objects]
        records.append({
            "image_id": int(target["image_id"]),
            "objects": objects,
            "oracle_detected": int(detected.any(dim=1).sum()) if detected.numel() else 0,
            "best_global_detected": int(branch_counts.max()) if branch_counts.numel() else 0,
            "best_global_stop": float(config.stops[int(branch_counts.argmax())]) if objects else None,
            "preferred_stop_spread": (
                max(preferred_stops) - min(preferred_stops) if len(preferred_stops) >= 2 else 0.0
            ),
        })
        if index == 1 or index % 10 == 0:
            print("oracle image={}/{} objects={} preferred_stops={}".format(
                index, limit, len(objects), preferred_stops
            ), flush=True)

    report = summarize(
        records,
        config.stops,
        {"score": config.score_threshold, "iou": config.iou_threshold},
        limit,
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = dict(report)
    printable.pop("images_detail")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()

