"""Train and compare a learned exposure policy against the fixed HDRDet branches.

The DETR detector and class-wise NMS are shared between both evaluation modes.
Only the exposure branch generator changes.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from datasets.hdr_mat import HDRMatDetection, collate_hdr
from main import get_args_parser
from models import build_model
from models.adaptive_exposure import AdaptiveExposureDETR


OFFICIAL_DETR = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
POLICY_ARCHITECTURE = "paper_bounded_ordered_gaps_v3"


def paper_stop_max(branches):
    """Largest MEL stop for each branch-count interpretation in the paper."""
    return float(2 * (branches - 1) if branches in (7, 8) else branches - 1)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--annotations", required=True, type=Path,
                        help="COCO JSON for this split")
    parser.add_argument("--mat-dir", required=True, type=Path,
                        help="directory containing HDR rgb_dem .mat files")
    parser.add_argument("--mat-key", default="rgb_dem",
                        help="MAT variable containing an H x W x 3 HDR image")
    parser.add_argument("--hdr-scale", type=float, default=255.0,
                        help="divide raw MAT values by this before clipping; 255 matches the paper")
    parser.add_argument("--source-scale", action="append", default=[], metavar="SOURCE=SCALE",
                        help="override --hdr-scale for a top-level assembled source; repeat for mixed datasets")
    parser.add_argument("--val-annotations", type=Path,
                        help="disjoint labeled HDR validation COCO JSON; required for normal training")
    parser.add_argument("--val-mat-dir", type=Path,
                        help="validation MAT directory; defaults to --mat-dir")
    parser.add_argument("--allow-train-only", action="store_true",
                        help="explicitly train without validation (not recommended for final results)")
    parser.add_argument("--detector-checkpoint", default=OFFICIAL_DETR,
                        help="use the SAME complete DETR checkpoint for fixed and adaptive runs")
    parser.add_argument("--policy-checkpoint", type=Path,
                        help="trained policy for evaluation, or weights-only warm start for training")
    parser.add_argument("--resume", type=Path,
                        help="resume optimizer, epoch, and best validation loss from policy_latest.pth")
    parser.add_argument("--output-dir", type=Path, default=Path("adaptive_output"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--branches", type=int, default=15,
                        choices=(7, 8, 12, 14, 15, 16, 17),
                        help="Number of rendered images. Default 15 follows Table 5's n=0..14; prose says N=14; released script example gives 12.")
    parser.add_argument("--branch-batch-size", type=int, default=2,
                        help="number of exposure branches per gradient pass; loss selection and evaluation batch all branches")
    parser.add_argument("--dataset-mode", choices=("ours", "rod"), default="ours",
                        help="Paper evaluates classes person/bicycle/car/truck on Ours; car/truck on ROD")
    parser.add_argument("--short-side", type=int, default=800,
                        help="same default training/evaluation resolution as the released detector")
    parser.add_argument("--max-side", type=int, default=1333)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--coverage-weight", type=float, default=0.05)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--baseline-only", action="store_true",
                        help="evaluate the paper's fixed gains without a trained policy")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-samples", type=int, default=3,
                        help="number of MAT contents to inspect in preflight; all file paths are checked")
    parser.add_argument("--limit-train-images", type=int, default=0,
                        help="pilot only: process this many training images per epoch; 0 means all")
    parser.add_argument("--limit-val-images", type=int, default=0,
                        help="pilot only: process this many validation images; 0 means all")
    parsed = parser.parse_args()
    parsed.source_scales = {}
    for item in parsed.source_scale:
        if "=" not in item:
            parser.error("--source-scale must have the form SOURCE=SCALE")
        source, raw_scale = item.split("=", 1)
        if source in parsed.source_scales:
            parser.error("duplicate --source-scale for {}".format(source))
        try:
            scale = float(raw_scale)
        except ValueError:
            parser.error("invalid --source-scale value: {}".format(item))
        if not source or "/" in source or "\\" in source or not math.isfinite(scale) or scale <= 0:
            parser.error("invalid --source-scale value: {}".format(item))
        parsed.source_scales[source] = scale
    return parsed


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(source):
    if str(source).startswith(("https://", "http://")):
        return torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=True)
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # Older supported PyTorch versions
        return torch.load(source, map_location="cpu")


def image_sizes(config):
    return config.short_side, config.max_side


def detector_fingerprint(weights):
    """Path-independent identity for a complete set of detector weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(config, device):
    detr_args = get_args_parser().parse_args([])
    detr_args.device = str(device)
    detr_args.lr_backbone = 0
    detr_args.aux_loss = False
    detr_args.intermediate = False
    detector, criterion, postprocessors = build_model(detr_args)
    checkpoint = load_checkpoint(config.detector_checkpoint)
    weights = checkpoint.get("model", checkpoint)
    detector.load_state_dict(weights, strict=True)
    short_side, max_side = image_sizes(config)
    model = AdaptiveExposureDETR(detector, branches=config.branches,
                                 stop_min=0.0,
                                 stop_max=paper_stop_max(config.branches),
                                 short_side=short_side, max_side=max_side).to(device)
    model.detector_fingerprint = detector_fingerprint(weights)
    model.freeze_detector()
    criterion = criterion.to(device).eval()
    return model, criterion, postprocessors


def make_loader(config, annotations=None, mat_dir=None, shuffle=None):
    short_side, max_side = image_sizes(config)
    dataset = HDRMatDetection(annotations or config.annotations, mat_dir or config.mat_dir,
                              short_side=short_side, max_side=max_side,
                              mat_key=config.mat_key, hdr_scale=config.hdr_scale,
                              source_scales=getattr(config, "source_scales", None))
    dataset.validate_paths()
    if shuffle is None:
        shuffle = config.action == "train"
    return DataLoader(dataset, batch_size=1, shuffle=shuffle,
                      num_workers=config.workers, collate_fn=collate_hdr)


def verify_disjoint(train_data, validation_data):
    # Compare basenames as well as paths.  Assembled source directories can
    # differ while still referring to a duplicated underlying frame.
    train_names = {Path(info["file_name"]).name for info in train_data.images}
    val_names = {Path(info["file_name"]).name for info in validation_data.images}
    overlap = train_names & val_names
    if overlap:
        raise ValueError("Training/validation image leakage: {} shared images, e.g. {}".format(
            len(overlap), sorted(overlap)[:5]))


def preflight(config):
    loader = make_loader(config, shuffle=False)
    data = loader.dataset
    count = min(config.preflight_samples, len(data))
    indices = sorted({round(i * (len(data) - 1) / max(count - 1, 1))
                      for i in range(count)})
    samples = []
    for index in indices:
        info = data.images[index]
        path = data._mat_path(info["file_name"])
        raw = data.load_raw(info["file_name"])
        image, target = data[index]
        effective_scale = data.scale_for(info)
        samples.append({"file": str(path), "shape": list(raw.shape),
                        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                        "normalized_max": float(image.max()),
                        "effective_scale": effective_scale,
                        "negative_fraction": float((raw < 0).mean()),
                        "clipped_fraction": float((raw >= effective_scale).mean()),
                        "objects": int(target["labels"].numel())})
    report = {"images": len(data), "annotations": len(data.coco["annotations"]),
              "categories": {str(cat["id"]): cat["name"] for cat in data.coco.get("categories", [])},
              "mat_key": data.mat_key, "hdr_scale": data.hdr_scale,
              "source_scales": data.source_scales,
              "samples": samples}
    print(json.dumps(report, indent=2), flush=True)
    if config.hdr_scale == 255 and samples and all(item["raw_max"] <= 1 for item in samples):
        print("WARNING: samples appear to be in [0,1]; --hdr-scale 255 may make them 255x too dark.", flush=True)
    if any(item["clipped_fraction"] > 0.25 for item in samples):
        print("WARNING: more than 25% of a sampled image clips; review --hdr-scale before training.", flush=True)
    if any(item["negative_fraction"] > 0.01 for item in samples):
        print("WARNING: more than 1% of a sampled image is negative and will be clipped to zero.", flush=True)


def detection_loss(criterion, outputs, target):
    terms = criterion(outputs, [target])
    return sum(terms[name] * weight for name, weight in criterion.weight_dict.items()
               if name in terms)


def detector_output_at(outputs, index):
    """Select one image from a batched DETR output for per-branch matching."""
    selected = {}
    for name, value in outputs.items():
        if name == "aux_outputs":
            selected[name] = [detector_output_at(item, index) for item in value]
        elif torch.is_tensor(value):
            selected[name] = value[index:index + 1]
        else:
            selected[name] = value
    return selected


def branch_detection_losses(criterion, outputs, target):
    """Return one Hungarian-matched detection loss per exposure branch."""
    count = outputs["pred_logits"].shape[0]
    return torch.stack([
        detection_loss(criterion, detector_output_at(outputs, index), target)
        for index in range(count)
    ])


def smooth_best_branch(branch_losses, temperature):
    return -temperature * (torch.logsumexp(-branch_losses / temperature, dim=0)
                           - math.log(branch_losses.numel()))


@torch.no_grad()
def validation_loss(config, model, criterion, loader, device):
    model.eval()
    total = 0.0
    seen = 0
    for index, (images, targets) in enumerate(loader, start=1):
        if getattr(config, "limit_val_images", 0) and index > config.limit_val_images:
            break
        hdr = images[0].to(device)[None]
        target = {key: value.to(device) for key, value in targets[0].items()}
        if target["labels"].numel() == 0:
            continue
        stops = model.policy(hdr)
        outputs = model.detect_branches(hdr, stops)
        losses = branch_detection_losses(criterion, outputs, target)
        coverage = model.policy.coverage_loss(hdr, stops)
        objective = smooth_best_branch(losses, config.temperature) + config.coverage_weight * coverage
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite validation objective")
        total += float(objective)
        seen += 1
    if not seen:
        raise ValueError("No labeled objects in validation annotations")
    return total / seen


def save_checkpoint(state, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def train(config, model, criterion, loader, device, val_loader=None):
    if getattr(config, "policy_checkpoint", None) and getattr(config, "resume", None):
        raise ValueError("Use either --policy-checkpoint for a warm start or --resume, not both")
    if getattr(config, "policy_checkpoint", None):
        checkpoint = load_checkpoint(config.policy_checkpoint)
        if checkpoint.get("policy_architecture") != POLICY_ARCHITECTURE:
            raise ValueError("Warm-start checkpoint uses an older exposure-policy architecture")
        if checkpoint.get("branches") != config.branches:
            raise ValueError("Warm-start policy branch count differs from --branches")
        if (checkpoint.get("detector_fingerprint") is not None and
                checkpoint["detector_fingerprint"] != getattr(model, "detector_fingerprint", None)):
            raise ValueError("Warm-start policy was trained with different detector weights")
        model.policy.load_state_dict(checkpoint["policy"])
    optimizer = torch.optim.AdamW(model.policy.parameters(),
                                  lr=config.learning_rate, weight_decay=1e-4)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_train = math.inf
    best_val = math.inf
    start_epoch = 0
    if getattr(config, "resume", None):
        checkpoint = load_checkpoint(config.resume)
        if checkpoint.get("policy_architecture") != POLICY_ARCHITECTURE:
            raise ValueError("Resume checkpoint uses an older exposure-policy architecture")
        if checkpoint.get("branches") != config.branches:
            raise ValueError("Resume checkpoint branch count differs from --branches")
        if checkpoint.get("detector_fingerprint") != getattr(model, "detector_fingerprint", None):
            raise ValueError("Resume checkpoint was trained with different detector weights")
        if "optimizer" not in checkpoint:
            raise ValueError("This checkpoint has no optimizer state; use --policy-checkpoint to warm start")
        for name, current in (("learning_rate", config.learning_rate),
                              ("temperature", config.temperature),
                              ("coverage_weight", config.coverage_weight),
                              ("short_side", loader.dataset.short_side),
                              ("max_side", loader.dataset.max_side),
                              ("hdr_scale", getattr(config, "hdr_scale", 255.0)),
                              ("source_scales", getattr(config, "source_scales", {})),
                              ("mat_key", getattr(config, "mat_key", "rgb_dem")),
                              ("limit_train_images", getattr(config, "limit_train_images", 0)),
                              ("limit_val_images", getattr(config, "limit_val_images", 0)),
                              ("stop_min", 0.0),
                              ("stop_max", paper_stop_max(config.branches))):
            if checkpoint.get(name) != current:
                raise ValueError("Resume setting {} differs from the saved run".format(name))
        model.policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch = checkpoint["epoch"]
        best_train = checkpoint.get("best_train_loss", math.inf)
        best_val = checkpoint.get("best_validation_loss", math.inf)
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    if start_epoch >= config.epochs:
        raise ValueError("--epochs is the total desired epoch count and must exceed the saved epoch")

    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.monotonic()
        model.policy.train()
        model.detector.eval()
        epoch_loss = 0.0
        seen = 0
        for step, (images, targets) in enumerate(loader, start=1):
            if getattr(config, "limit_train_images", 0) and step > config.limit_train_images:
                break
            hdr = images[0].to(device)[None]
            target = {key: value.to(device) for key, value in targets[0].items()}
            if target["labels"].numel() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            statistics = model.policy._statistics(hdr)

            # The soft minimum of branch detection losses has derivative
            # softmax(-loss / temperature) times each branch gradient. Estimate
            # these weights without a graph in one paper-style detector batch,
            # then rerun small branch micro-batches with gradients.  This is the
            # same objective as the serial derivative but uses GPU batch
            # parallelism without retaining all K detector graphs at once.
            with torch.no_grad():
                stops = model.policy(hdr, statistics)
                outputs = model.detect_branches(hdr, stops)
                branch_losses = branch_detection_losses(criterion, outputs, target)
                if not torch.isfinite(branch_losses).all():
                    raise FloatingPointError("Non-finite branch detection loss")
                weights = torch.softmax(-branch_losses / config.temperature, dim=0)
                smooth_min = smooth_best_branch(branch_losses, config.temperature)

            branch_batch_size = getattr(config, "branch_batch_size", 1)
            for first in range(0, config.branches, branch_batch_size):
                branches = list(range(first, min(first + branch_batch_size,
                                                 config.branches)))
                stops = model.policy(hdr, statistics)
                outputs = model.detect_branches(hdr, stops, branches)
                losses = branch_detection_losses(criterion, outputs, target)
                loss = sum(losses[index] * weights[branch]
                           for index, branch in enumerate(branches))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
            stops = model.policy(hdr, statistics)
            coverage = model.policy.coverage_loss(hdr, stops)
            if not torch.isfinite(coverage):
                raise FloatingPointError("Non-finite exposure coverage loss")
            (config.coverage_weight * coverage).backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            total = smooth_min.item() + config.coverage_weight * coverage.detach().item()
            epoch_loss += total
            seen += 1
            if step == 1 or step % 20 == 0:
                print("epoch={} image={}/{} loss={:.4f} coverage={:.4f} stops={}".format(
                    epoch + 1, step, len(loader), total, coverage.detach().item(),
                    [round(value, 2) for value in stops.detach()[0].tolist()]), flush=True)

        if seen == 0:
            raise ValueError("No labeled objects in the training annotations")
        average = epoch_loss / seen
        held_out = (validation_loss(config, model, criterion, val_loader, device)
                    if val_loader is not None else None)
        best_train = min(best_train, average)
        if held_out is not None:
            is_best_val = held_out < best_val
            best_val = min(best_val, held_out)
        else:
            is_best_val = False
        state = {
            "policy": model.policy.state_dict(),
            "policy_architecture": POLICY_ARCHITECTURE,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "average_train_loss": average,
            "validation_loss": held_out,
            "best_train_loss": best_train,
            "best_validation_loss": best_val,
            "detector_checkpoint": config.detector_checkpoint,
            "detector_fingerprint": getattr(model, "detector_fingerprint", None),
            "branches": config.branches,
            "stop_min": 0.0,
            "stop_max": paper_stop_max(config.branches),
            "training_annotations": str(config.annotations),
            "training_mat_dir": str(config.mat_dir),
            "validation_annotations": str(config.val_annotations) if getattr(config, "val_annotations", None) else None,
            "validation_mat_dir": str(config.val_mat_dir) if getattr(config, "val_mat_dir", None) else None,
            "hdr_scale": getattr(config, "hdr_scale", 255.0),
            "source_scales": getattr(config, "source_scales", {}),
            "mat_key": getattr(config, "mat_key", "rgb_dem"),
            "short_side": loader.dataset.short_side,
            "max_side": loader.dataset.max_side,
            "seed": config.seed,
            "learning_rate": config.learning_rate,
            "temperature": config.temperature,
            "coverage_weight": config.coverage_weight,
            "branch_batch_size": getattr(config, "branch_batch_size", 1),
            "limit_train_images": getattr(config, "limit_train_images", 0),
            "limit_val_images": getattr(config, "limit_val_images", 0),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if device.type == "cuda":
            state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        save_checkpoint(state, config.output_dir / "policy_latest.pth")
        if average <= best_train:
            save_checkpoint(state, config.output_dir / "policy_best_train_loss.pth")
        if is_best_val:
            save_checkpoint(state, config.output_dir / "policy_best_val_loss.pth")
        print("epoch={} train_loss={:.4f} validation_loss={} seconds={:.1f} pilot_limits={}/{}".format(
            epoch + 1, average, "n/a" if held_out is None else "{:.4f}".format(held_out),
            time.monotonic() - epoch_started, getattr(config, "limit_train_images", 0),
            getattr(config, "limit_val_images", 0)), flush=True)


def fixed_stops(branches, device):
    # The paper is inconsistent about whether N counts images or is the last
    # inclusive index. Support both readings of its 14/16 and 7 settings.
    if branches in (7, 8):
        return torch.arange(0, 2 * branches, 2, device=device,
                            dtype=torch.float32)[None]
    if branches in (12, 14, 15, 16, 17):
        return torch.arange(branches, device=device, dtype=torch.float32)[None]
    raise ValueError("Unsupported fixed-baseline branch count")


def paper_detect_branches(model, normalized_hdr, stops, device, branches=None):
    """Reproduce the released NumPy/OpenCV MEL preprocessing at evaluation.

    Training uses the differentiable PyTorch equivalent. Final evaluation has
    no gradient requirement, so both fixed and adaptive policies can use the
    paper's exact tone-map, resize, normalization, and float-cast order.
    """
    import cv2

    if branches is None:
        branches = range(stops.shape[1])
    else:
        branches = list(branches)
    height, width = normalized_hdr.shape[:2]
    size = model.short_side
    if max(height, width) / min(height, width) * size > model.max_side:
        size = int(round(model.max_side * min(height, width) / max(height, width)))
    if width < height:
        new_height, new_width = int(size * height / width), size
    else:
        new_height, new_width = size, int(size * width / height)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float64)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float64)
    rendered = []
    for branch in branches:
        gain = 2.0 ** float(stops[0, branch])
        ldr = np.clip(normalized_hdr * gain, 0, 1) ** (1 / 2.2)
        ldr = cv2.resize(ldr, (new_width, new_height))
        ldr = ((ldr - mean) / std).astype(np.float32)
        rendered.append(torch.from_numpy(np.ascontiguousarray(
            ldr.transpose(2, 0, 1))).to(device))
    if hasattr(model, "detector"):
        return model.detector(rendered)
    # Lightweight evaluation stubs in the regression tests do not expose the
    # shared detector. Their branch outputs are already deterministic.
    return model.detect_branches(
        torch.from_numpy(normalized_hdr.astype(np.float32)).permute(2, 0, 1)[None].to(device),
        stops, branches)


@torch.no_grad()
def evaluate_variant(config, model, postprocessors, loader, device, variant):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = []
    exposure_log = []
    model.eval()
    for index, (images, targets) in enumerate(loader, start=1):
        hdr = images[0].to(device)[None]
        target = targets[0]
        stops = (fixed_stops(config.branches, device) if variant == "fixed"
                 else model.policy(hdr))
        target_size = target["orig_size"].to(device)[None]
        info = loader.dataset.images[index - 1]
        raw = loader.dataset.load_raw(info["file_name"])
        paper_hdr = np.clip(raw / loader.dataset.scale_for(info), 0, 1).astype(np.float64)
        outputs = paper_detect_branches(model, paper_hdr, stops, device)
        branch_results = postprocessors["bbox"](
            outputs, target_size.expand(config.branches, -1))
        boxes = torch.cat([item["boxes"] for item in branch_results])
        scores = torch.cat([item["scores"] for item in branch_results])
        labels = torch.cat([item["labels"] for item in branch_results])
        keep = scores >= config.score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.numel():
            # Reproduce the released late_fusion implementation literally:
            # move float32 boxes/scores to CPU, group by class in first-seen
            # order, and call torchvision NMS separately for each class.
            # batched_nms is usually equivalent, but its coordinate-offset
            # trick can differ for borderline boxes because of float32
            # rounding at large class offsets.
            boxes_cpu = boxes.cpu()
            scores_cpu = scores.cpu()
            labels_cpu = labels.cpu()
            kept = []
            for category in dict.fromkeys(labels_cpu.tolist()):
                category_indices = torch.nonzero(
                    labels_cpu == category, as_tuple=False).flatten()
                local = torchvision.ops.nms(
                    boxes_cpu[category_indices], scores_cpu[category_indices],
                    config.nms_iou)
                kept.extend(category_indices[local].tolist())
            for idx in kept:
                x0, y0, x1, y1 = boxes[idx].tolist()
                results.append({"image_id": target["image_id"].item(),
                                "category_id": int(labels[idx]),
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "score": float(scores[idx])})
        exposure_log.append({"image_id": target["image_id"].item(),
                             "stops": [float(value) for value in stops[0].tolist()]})
        if index == 1 or index % 20 == 0:
            print("{} image={}/{} stops={}".format(
                variant, index, len(loader),
                [round(value, 2) for value in exposure_log[-1]["stops"]]), flush=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / (variant + "_detections.json")).open("w") as stream:
        json.dump(results, stream)
    with (config.output_dir / (variant + "_stops.json")).open("w") as stream:
        json.dump(exposure_log, stream, indent=2)

    if not results:
        raise RuntimeError("No detections produced; check the checkpoint and dataset")
    coco_gt = COCO(str(config.annotations))
    coco_dt = coco_gt.loadRes(results)
    category_ids = [1, 2, 3, 8] if config.dataset_mode == "ours" else [3, 8]
    missing = set(category_ids) - set(coco_gt.getCatIds())
    if missing:
        raise ValueError("Annotation JSON is missing paper evaluation categories: {}".format(sorted(missing)))
    empty = [category for category in category_ids if not coco_gt.getAnnIds(catIds=[category])]
    if empty:
        raise ValueError("Evaluation split has no ground-truth objects for categories: {}".format(empty))
    ap_per_category = {}
    for category in category_ids:
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.catIds = [category]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        ap_per_category[str(category)] = float(evaluator.stats[0] * 100)
    report = {"variant": variant, "images": len(loader),
              "dataset_mode": config.dataset_mode,
              "branch_count": config.branches,
              "fixed_gains": [int(2 ** stop) for stop in fixed_stops(config.branches, device)[0].tolist()],
              "mean_AP_percent": sum(ap_per_category.values()) / len(ap_per_category),
              "AP_percent_by_category": ap_per_category,
              "detector_checkpoint": config.detector_checkpoint,
              "detector_fingerprint": getattr(model, "detector_fingerprint", None),
              "hdr_scale": getattr(config, "hdr_scale", 255.0),
              "source_scales": getattr(config, "source_scales", {}),
              "mat_key": getattr(config, "mat_key", "rgb_dem"),
              "nms_iou": config.nms_iou,
              "evaluation_renderer": "paper_numpy_opencv",
              "short_side": loader.dataset.short_side,
              "max_side": loader.dataset.max_side}
    with (config.output_dir / (variant + "_metrics.json")).open("w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    config = arguments()
    if config.branches < 3 or config.branch_batch_size < 1:
        raise ValueError("--branches must be at least 3 and --branch-batch-size must be positive")
    if config.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if config.coverage_weight < 0:
        raise ValueError("--coverage-weight cannot be negative")
    if not math.isfinite(config.hdr_scale) or config.hdr_scale <= 0:
        raise ValueError("--hdr-scale must be positive and finite")
    if config.short_side <= 0 or config.max_side < config.short_side:
        raise ValueError("--short-side and --max-side must be positive and ordered")
    if config.epochs <= 0 or config.learning_rate <= 0:
        raise ValueError("--epochs and --learning-rate must be positive")
    if not 0 <= config.score_threshold <= 1 or not 0 <= config.nms_iou <= 1:
        raise ValueError("--score-threshold and --nms-iou must lie in [0, 1]")
    if (config.preflight_samples <= 0 or config.workers < 0 or
            config.limit_train_images < 0 or config.limit_val_images < 0):
        raise ValueError("Preflight samples must be positive; worker and pilot limits cannot be negative")
    if config.action == "preflight":
        preflight(config)
        return
    if config.action == "train" and not config.val_annotations and not config.allow_train_only:
        raise ValueError("Training needs --val-annotations; use --allow-train-only only for a smoke test")
    if config.val_mat_dir and not config.val_annotations:
        raise ValueError("--val-mat-dir requires --val-annotations")
    if config.resume and config.action != "train":
        raise ValueError("--resume is only used with train")
    if config.action == "evaluate" and not config.baseline_only and not config.policy_checkpoint:
        raise ValueError("--policy-checkpoint is required for adaptive evaluation")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = select_device(config.device)
    print("device={}".format(device), flush=True)
    loader = make_loader(config)
    val_loader = None
    if config.action == "train" and config.val_annotations:
        val_loader = make_loader(config, config.val_annotations,
                                 config.val_mat_dir or config.mat_dir, shuffle=False)
        verify_disjoint(loader.dataset, val_loader.dataset)
    model, criterion, postprocessors = make_model(config, device)
    if config.action == "train":
        train(config, model, criterion, loader, device, val_loader)
    else:
        checkpoint = None
        if not config.baseline_only:
            checkpoint = load_checkpoint(config.policy_checkpoint)
            if checkpoint.get("policy_architecture") != POLICY_ARCHITECTURE:
                raise ValueError("Policy checkpoint uses an older exposure-policy architecture")
            if checkpoint["branches"] != config.branches:
                raise ValueError("Policy checkpoint branch count differs from --branches")
            saved_fingerprint = checkpoint.get("detector_fingerprint")
            if saved_fingerprint is not None:
                if saved_fingerprint != model.detector_fingerprint:
                    raise ValueError("Policy checkpoint was trained with different DETR weights")
            elif checkpoint["detector_checkpoint"] != config.detector_checkpoint:
                raise ValueError("Use the detector checkpoint used to train this older policy")
        fixed = evaluate_variant(config, model, postprocessors, loader, device, "fixed")
        if not config.baseline_only:
            model.policy.load_state_dict(checkpoint["policy"])
            adaptive = evaluate_variant(config, model, postprocessors, loader, device,
                                        "adaptive")
            print("adaptive minus fixed mean AP: {:+.3f} percentage points".format(
                adaptive["mean_AP_percent"] - fixed["mean_AP_percent"]))


if __name__ == "__main__":
    main()
