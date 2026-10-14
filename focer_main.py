"""Train and evaluate teacher-guided, detector-frozen FOCER-v2.

The exposure stops and all official DETR parameters are immutable.  A frozen
DETR teacher first produces per-object exposure utility targets.  Training then
updates only the object-conditioned routing projections and stop encoder.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.hdr_mat import HDRMatDetection, collate_hdr
from models import build_model
from models.focer import (
    FixedExposureRenderer,
    FixedStopQueryRouter,
    matched_routing_entropy,
    teacher_guided_routing_loss,
)
from util import box_ops


OFFICIAL_DETR = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
ARCHITECTURE = "focer_v2_frozen_detr_teacher_guided_query_routing"
TEACHER_CACHE_VERSION = 1
PAPER_CATEGORIES = (1, 2, 3, 8)


def parse_stops(specification):
    """Parse ``0:14:1`` or ``0,3.5,7,10.5,14`` into immutable stops."""
    specification = str(specification).strip()
    if ":" in specification:
        parts = specification.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("stop ranges use start:end:step")
        start, end, step = map(float, parts)
        if not step > 0 or end <= start:
            raise argparse.ArgumentTypeError("stop range must be increasing")
        count = int(round((end - start) / step))
        values = [start + index * step for index in range(count + 1)]
        if not math.isclose(values[-1], end, rel_tol=0, abs_tol=1e-6):
            raise argparse.ArgumentTypeError("stop range must include its endpoint exactly")
    else:
        try:
            values = [float(item) for item in specification.split(",") if item.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("stops must be numeric") from exc
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("at least two finite stops are required")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("stops must be strictly increasing")
    return values


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("preflight", "build_teacher_cache", "train", "fit", "evaluate", "inspect"),
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--mat-dir", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--val-mat-dir", type=Path)
    parser.add_argument("--mat-key", default="rgb_dem")
    parser.add_argument("--hdr-scale", type=float, default=255.0)
    parser.add_argument("--detector-checkpoint", default=OFFICIAL_DETR)
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--val-teacher-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("focer_output"))
    parser.add_argument("--stops", type=parse_stops, default=parse_stops("0:14:1"))
    parser.add_argument("--short-side", type=int, default=800)
    parser.add_argument("--max-side", type=int, default=1333)
    parser.add_argument("--feature-chunk-size", type=int, default=3)
    parser.add_argument("--route-temperature", type=float, default=0.7,
                        help="final/evaluation routing temperature")
    parser.add_argument("--route-temperature-start", type=float, default=1.5)
    parser.add_argument("--route-top-k", type=int, default=3,
                        help="maximum exposures used by each query; 0 keeps dense soft routing")
    parser.add_argument("--curriculum-soft-epochs", type=int, default=3)
    parser.add_argument("--curriculum-middle-epochs", type=int, default=3)
    parser.add_argument("--curriculum-middle-top-k", type=int, default=5)
    parser.add_argument("--curriculum-total-epochs", type=int, default=0,
                        help="temperature schedule length; 0 uses --epochs")
    parser.add_argument("--radiometric-strength", type=float, default=0.5)
    parser.add_argument("--routing-entropy-weight", type=float, default=0.0)
    parser.add_argument("--teacher-routing-weight", type=float, default=1.0)
    parser.add_argument("--teacher-target-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-min-confidence", type=float, default=0.05)
    parser.add_argument("--select-by-validation-map", action="store_true",
                        help="select train checkpoint by validation COCO mAP instead of loss")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--routing-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-max-norm", type=float, default=0.1)
    parser.set_defaults(balanced_sampling=True)
    parser.add_argument("--balanced-sampling", dest="balanced_sampling", action="store_true")
    parser.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    parser.add_argument("--max-sample-weight", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train-images", type=int, default=0)
    parser.add_argument("--limit-val-images", type=int, default=0)
    parser.add_argument("--limit-eval-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--routing-log-threshold", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=20,
                        help="print detailed progress every N images")
    parser.add_argument("--preflight-samples", type=int, default=3)
    config = parser.parse_args(argv)
    if config.val_mat_dir and not config.val_annotations:
        parser.error("--val-mat-dir requires --val-annotations")
    if config.action == "train":
        if not config.val_annotations:
            parser.error("training requires a disjoint --val-annotations split")
        if not config.teacher_cache or not config.val_teacher_cache:
            parser.error("training requires --teacher-cache and --val-teacher-cache")
    if config.action == "fit" and not config.teacher_cache:
        parser.error("fixed-epoch fitting requires --teacher-cache")
    if config.action == "build_teacher_cache" and not config.teacher_cache:
        parser.error("build_teacher_cache requires --teacher-cache as its output path")
    if config.action in ("evaluate", "inspect") and not config.router_checkpoint:
        parser.error("evaluate/inspect requires --router-checkpoint")
    if config.resume and config.action not in ("train", "fit"):
        parser.error("--resume is only valid for training/fitting")
    if config.short_side <= 0 or config.max_side < config.short_side:
        parser.error("invalid image dimensions")
    if config.feature_chunk_size < 0:
        parser.error("--feature-chunk-size cannot be negative")
    if (config.route_temperature <= 0 or config.route_temperature_start <= 0
            or config.radiometric_strength < 0
            or config.route_top_k < 0):
        parser.error("invalid routing hyperparameters")
    if (config.route_top_k > len(config.stops)
            or config.curriculum_middle_top_k > len(config.stops)):
        parser.error("routing budgets cannot exceed the number of fixed stops")
    if config.curriculum_soft_epochs < 0 or config.curriculum_middle_epochs < 0:
        parser.error("curriculum epoch counts cannot be negative")
    if (config.curriculum_total_epochs < 0
            or (config.curriculum_total_epochs
                and config.curriculum_total_epochs < config.epochs)):
        parser.error("--curriculum-total-epochs must be zero or at least --epochs")
    if (config.routing_entropy_weight < 0 or config.teacher_routing_weight < 0
            or config.teacher_target_temperature <= 0
            or not 0 <= config.teacher_min_confidence < 1):
        parser.error("invalid routing-loss hyperparameters")
    if config.max_sample_weight < 1:
        parser.error("--max-sample-weight must be at least one")
    if config.log_every <= 0:
        parser.error("--log-every must be positive")
    return config


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(enabled):
    """Use the current AMP API while remaining compatible with older PyTorch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled):
    """Construct a CUDA gradient scaler across old and new PyTorch releases."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def base_detr_arguments(device):
    """Arguments matching the official DETR-R50 checkpoint."""
    return SimpleNamespace(
        device=str(device),
        lr_backbone=0.0,
        aux_loss=True,
        intermediate=False,
        masks=False,
        backbone="resnet50",
        dilation=False,
        position_embedding="sine",
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        enc_layers=6,
        dec_layers=6,
        pre_norm=False,
        num_queries=100,
        dataset_file="coco",
        frozen_weights=None,
        set_cost_class=1.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        mask_loss_coef=1.0,
        dice_loss_coef=1.0,
        eos_coef=0.1,
    )


def load_checkpoint(source):
    if str(source).startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(
            str(source), map_location="cpu", check_hash=True
        )
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def tensor_fingerprint(weights):
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(config, device):
    detr_config = base_detr_arguments(device)
    detector, criterion, postprocessors = build_model(detr_config)
    checkpoint = load_checkpoint(config.detector_checkpoint)
    detector_weights = checkpoint.get("model", checkpoint)
    detector.load_state_dict(detector_weights, strict=True)
    model = FixedStopQueryRouter(
        detector,
        stops=config.stops,
        short_side=config.short_side,
        max_side=config.max_side,
        feature_chunk_size=config.feature_chunk_size,
        route_temperature=config.route_temperature,
        radiometric_strength=config.radiometric_strength,
        route_top_k=config.route_top_k,
    )
    model.freeze_pretrained_detector()
    model.detector_fingerprint = tensor_fingerprint(detector_weights)
    model = model.to(device)
    criterion = criterion.to(device)
    if config.router_checkpoint:
        routed = load_checkpoint(config.router_checkpoint)
        validate_router_checkpoint(routed, config, model)
        model.load_state_dict(routed["model"], strict=True)
    return model, criterion, postprocessors


def validate_router_checkpoint(checkpoint, config, model):
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint is not a FOCER-v2 model")
    saved_stops = checkpoint.get("fixed_stops")
    if saved_stops is None or len(saved_stops) != len(config.stops):
        raise ValueError("Checkpoint uses a different fixed-stop list")
    if not np.allclose(saved_stops, config.stops, rtol=0, atol=1e-6):
        raise ValueError("Checkpoint uses a different fixed-stop list")
    fingerprint = checkpoint.get("detector_fingerprint")
    if fingerprint and fingerprint != model.detector_fingerprint:
        raise ValueError("Checkpoint was initialized from different DETR weights")
    saved_config = checkpoint.get("config", {})
    expected = {
        "short_side": config.short_side,
        "max_side": config.max_side,
        "route_temperature": config.route_temperature,
        "route_top_k": config.route_top_k,
        "radiometric_strength": config.radiometric_strength,
        "strict_detector_freeze": True,
    }
    for name, value in expected.items():
        if name in saved_config:
            saved_value = saved_config[name]
            if isinstance(value, bool):
                equal = bool(saved_value) == value
            else:
                equal = math.isclose(
                    float(saved_value), float(value), rel_tol=0, abs_tol=1e-8
                )
            if not equal:
                raise ValueError("Checkpoint {} differs from the requested value".format(name))


def balanced_image_weights(dataset, maximum_weight):
    """Capped inverse-square-root sampling for the four evaluated categories."""
    category_counts = {category: 0 for category in PAPER_CATEGORIES}
    labels_by_image = {}
    for info in dataset.images:
        labels = [
            ann["category_id"]
            for ann in dataset.annotations.get(info["id"], [])
            if ann["category_id"] in category_counts
        ]
        labels_by_image[info["id"]] = labels
        for label in labels:
            category_counts[label] += 1
    nonzero = [count for count in category_counts.values() if count]
    if not nonzero:
        raise ValueError("Balanced sampling found no paper-category objects")
    reference = max(nonzero)
    category_weights = {
        category: min(maximum_weight, math.sqrt(reference / max(count, 1)))
        for category, count in category_counts.items()
    }
    weights = []
    for info in dataset.images:
        labels = labels_by_image[info["id"]]
        weights.append(max([1.0] + [category_weights[label] for label in labels]))
    summary = {
        "category_counts": category_counts,
        "category_sampling_weights": category_weights,
        "minimum_image_weight": min(weights),
        "maximum_image_weight": max(weights),
    }
    return torch.as_tensor(weights, dtype=torch.double), summary


def make_loader(config, annotations=None, mat_dir=None, shuffle=False,
                balanced=False):
    dataset = HDRMatDetection(
        annotations or config.annotations,
        mat_dir or config.mat_dir,
        short_side=config.short_side,
        max_side=config.max_side,
        mat_key=config.mat_key,
        hdr_scale=config.hdr_scale,
    )
    dataset.validate_paths()
    sampler = None
    if balanced:
        weights, summary = balanced_image_weights(dataset, config.max_sample_weight)
        generator = torch.Generator().manual_seed(config.seed)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True, generator=generator
        )
        print(json.dumps({
            "event": "balanced_sampling",
            **summary,
        }, indent=2), flush=True)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=config.workers,
        collate_fn=collate_hdr,
    )


def teacher_utility_logits(outputs, target, matcher):
    """Return frozen-DETR utility logits in ``[objects, exposures]`` order."""
    branches = outputs["pred_logits"].shape[0]
    objects = target["labels"].numel()
    utility = outputs["pred_logits"].new_full((objects, branches), -50.0)
    if not objects:
        return utility
    for branch in range(branches):
        branch_output = {
            "pred_logits": outputs["pred_logits"][branch:branch + 1],
            "pred_boxes": outputs["pred_boxes"][branch:branch + 1],
        }
        query_indices, target_indices = matcher(branch_output, [target])[0]
        query_indices = query_indices.to(outputs["pred_logits"].device)
        target_indices = target_indices.to(outputs["pred_logits"].device)
        predicted_boxes = branch_output["pred_boxes"][0, query_indices]
        target_boxes = target["boxes"][target_indices]
        probabilities = branch_output["pred_logits"][0, query_indices].softmax(-1)
        class_probability = probabilities[
            torch.arange(query_indices.numel(), device=probabilities.device),
            target["labels"][target_indices],
        ]
        l1 = (predicted_boxes - target_boxes).abs().sum(dim=-1)
        giou = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(predicted_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        ))
        # This mirrors DETR's relative localization coefficients while using
        # log class probability so teacher targets remain well-scaled.
        score = class_probability.clamp_min(1e-8).log() - 5.0 * l1 - 2.0 * (1.0 - giou)
        utility[target_indices, branch] = score
    return utility


@torch.no_grad()
def build_teacher_cache(config, loader, device):
    """Run the untouched official DETR once and cache per-object branch utility."""
    detr_config = base_detr_arguments(device)
    teacher, criterion, _ = build_model(detr_config)
    checkpoint = load_checkpoint(config.detector_checkpoint)
    detector_weights = checkpoint.get("model", checkpoint)
    teacher.load_state_dict(detector_weights, strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    renderer = FixedExposureRenderer(
        config.stops, short_side=config.short_side, max_side=config.max_side
    ).to(device).eval()
    entries = {}
    started = time.monotonic()
    object_count = 0
    for index, (images, raw_targets) in enumerate(loader, start=1):
        image = images[0].to(device)
        target = move_target(raw_targets[0], device)
        rendered = renderer(image.unsqueeze(0))[0]
        chunks = []
        chunk_size = config.feature_chunk_size or rendered.shape[0]
        for start in range(0, rendered.shape[0], chunk_size):
            prediction = teacher(list(rendered[start:start + chunk_size]))
            chunks.append({
                "pred_logits": prediction["pred_logits"],
                "pred_boxes": prediction["pred_boxes"],
            })
        outputs = {
            name: torch.cat([chunk[name] for chunk in chunks], dim=0)
            for name in ("pred_logits", "pred_boxes")
        }
        utilities = teacher_utility_logits(outputs, target, criterion.matcher)
        image_id = str(int(raw_targets[0]["image_id"]))
        entries[image_id] = {
            "labels": raw_targets[0]["labels"].clone(),
            "boxes": raw_targets[0]["boxes"].clone(),
            "utilities": utilities.cpu(),
        }
        object_count += int(raw_targets[0]["labels"].numel())
        if index == 1 or index % config.log_every == 0 or index == len(loader):
            elapsed = time.monotonic() - started
            print(
                "teacher cache image={}/{} objects={} elapsed={} eta={}".format(
                    index, len(loader), object_count, format_duration(elapsed),
                    format_duration(elapsed / index * (len(loader) - index)),
                ),
                flush=True,
            )
    payload = {
        "version": TEACHER_CACHE_VERSION,
        "architecture": ARCHITECTURE,
        "annotations_sha256": file_sha256(loader.dataset.annotations_path),
        "detector_fingerprint": tensor_fingerprint(detector_weights),
        "fixed_stops": list(map(float, config.stops)),
        "short_side": config.short_side,
        "max_side": config.max_side,
        "hdr_scale": config.hdr_scale,
        "mat_key": config.mat_key,
        "images": len(loader.dataset),
        "objects": object_count,
        "entries": entries,
    }
    config.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.teacher_cache.with_suffix(config.teacher_cache.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(config.teacher_cache)
    print(json.dumps({
        "event": "teacher_cache_complete",
        "path": str(config.teacher_cache),
        "images": payload["images"],
        "objects": payload["objects"],
        "bytes": config.teacher_cache.stat().st_size,
        "detector_fingerprint": payload["detector_fingerprint"],
    }, indent=2), flush=True)
    return payload


def load_teacher_cache(path, dataset, config, detector_fingerprint):
    cache = load_checkpoint(path)
    expected = {
        "version": TEACHER_CACHE_VERSION,
        "architecture": ARCHITECTURE,
        "annotations_sha256": file_sha256(dataset.annotations_path),
        "detector_fingerprint": detector_fingerprint,
        "fixed_stops": list(map(float, config.stops)),
        "short_side": config.short_side,
        "max_side": config.max_side,
        "hdr_scale": config.hdr_scale,
        "mat_key": config.mat_key,
        "images": len(dataset),
    }
    for name, value in expected.items():
        actual = cache.get(name)
        if name == "fixed_stops":
            equal = actual is not None and np.allclose(actual, value, rtol=0, atol=1e-6)
        elif isinstance(value, float):
            equal = actual is not None and math.isclose(
                float(actual), value, rel_tol=0, abs_tol=1e-8
            )
        else:
            equal = actual == value
        if not equal:
            raise ValueError(
                "Teacher cache {} mismatch: expected {!r}, found {!r}".format(
                    name, value, actual
                )
            )
    if set(cache.get("entries", {})) != {str(info["id"]) for info in dataset.images}:
        raise ValueError("Teacher cache image IDs do not match the dataset")
    return cache


def teacher_utilities_for_targets(cache, targets):
    utilities = []
    for target in targets:
        entry = cache["entries"].get(str(int(target["image_id"].detach().cpu())))
        if entry is None:
            raise KeyError("Teacher cache is missing an image")
        labels = target["labels"].detach().cpu()
        boxes = target["boxes"].detach().cpu()
        if not torch.equal(entry["labels"], labels) or not torch.allclose(
                entry["boxes"], boxes, rtol=0, atol=1e-6):
            raise ValueError("Teacher cache object order/content differs from annotations")
        utilities.append(entry["utilities"])
    return utilities


def verify_disjoint(train_data, validation_data):
    train_names = {Path(item["file_name"]).name for item in train_data.images}
    validation_names = {Path(item["file_name"]).name for item in validation_data.images}
    overlap = train_names & validation_names
    if overlap:
        raise ValueError("Training/validation leakage: {} shared filenames".format(len(overlap)))


def move_target(target, device):
    return {name: value.to(device) if torch.is_tensor(value) else value
            for name, value in target.items()}


def detection_loss(criterion, outputs, targets):
    terms = criterion(outputs, targets)
    weighted = {
        name: value * criterion.weight_dict[name]
        for name, value in terms.items()
        if name in criterion.weight_dict
    }
    return sum(weighted.values()), terms


def routing_statistics(outputs):
    weights = outputs["routing_weights"].detach().clamp_min(1e-8)
    entropy = -(weights * weights.log()).sum(dim=-1)
    normalized = entropy / math.log(weights.shape[-1])
    marginal = weights.mean(dim=(0, 1))
    return {
        "routing_entropy": float(normalized.mean().cpu()),
        "routing_peak": float(weights.max(dim=-1)[0].mean().cpu()),
        "effective_branches": float(entropy.exp().mean().cpu()),
        "branch_marginal": [float(value) for value in marginal.cpu()],
    }


def format_duration(seconds):
    """Human-readable duration for progress and ETA messages."""
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "{}h {:02d}m {:02d}s".format(hours, minutes, seconds)
    if minutes:
        return "{}m {:02d}s".format(minutes, seconds)
    return "{}s".format(seconds)


def scalar_term(terms, name):
    """Read a scalar DETR loss term without retaining its graph."""
    value = terms.get(name)
    return float(value.detach().cpu()) if value is not None else float("nan")


def top_branch_summary(marginal, stops, count=5):
    """Describe the most-used fixed exposure branches."""
    pairs = sorted(
        zip(stops, marginal), key=lambda item: item[1], reverse=True
    )[:count]
    return ",".join("{:.1f}:{:.3f}".format(stop, weight)
                    for stop, weight in pairs)


def optimizer_for(config, model):
    routing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not model.is_router_parameter(name):
            raise RuntimeError("Pretrained detector parameter is trainable: {}".format(name))
        routing.append(parameter)
    if not routing:
        raise RuntimeError("No routing parameters are trainable")
    return torch.optim.AdamW(
        [{"params": routing, "lr": config.routing_lr}],
        weight_decay=config.weight_decay,
    )


def curriculum_state(config, epoch):
    """Soft -> top-middle -> final-top-k routing, with temperature annealing."""
    schedule_epochs = getattr(config, "curriculum_total_epochs", 0) or config.epochs
    if schedule_epochs <= 1:
        fraction = 1.0
    else:
        fraction = min(max(epoch / (schedule_epochs - 1), 0.0), 1.0)
    temperature = (
        config.route_temperature_start
        + fraction * (config.route_temperature - config.route_temperature_start)
    )
    if epoch < config.curriculum_soft_epochs:
        top_k = 0
        stage = "soft_all"
    elif epoch < config.curriculum_soft_epochs + config.curriculum_middle_epochs:
        top_k = config.curriculum_middle_top_k
        stage = "middle_topk"
    else:
        top_k = config.route_top_k
        stage = "final_topk"
    return {
        "stage": stage,
        "top_k": top_k,
        "temperature": float(temperature),
    }


def validation_map_summary(annotations, detections, evaluated_image_ids):
    """Compute the paper-category COCO AP from an already completed val pass."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        raise RuntimeError("Validation produced no detections")
    coco_gt = COCO(str(annotations))
    coco_dt = coco_gt.loadRes(detections)
    available = [
        category for category in PAPER_CATEGORIES
        if coco_gt.getAnnIds(imgIds=evaluated_image_ids, catIds=[category])
    ]
    if not available:
        raise RuntimeError("Validation contains none of the paper categories")
    ap = {}
    for category in available:
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = list(evaluated_image_ids)
        evaluator.params.catIds = [category]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        ap[str(category)] = float(evaluator.stats[0] * 100)
    return {
        "validation_AP_percent_by_category": ap,
        "validation_mean_AP_percent": sum(ap.values()) / len(ap),
        "validation_map_categories": available,
    }


def run_loss_epoch(config, model, criterion, loader, device, teacher_cache,
                   optimizer=None, limit=0, scaler=None, epoch=0,
                   compute_map=False):
    training = optimizer is not None
    phase = "train" if training else "validation"
    model.train(training)
    criterion.train(training)
    expected = min(len(loader), limit) if limit else len(loader)
    totals = []
    detection_totals = []
    teacher_totals = []
    teacher_confidences = []
    teacher_agreements = []
    teacher_object_count = 0
    matched_entropies = []
    route_entropies = []
    route_peaks = []
    effective_branches = []
    branch_marginals = []
    ce_losses = []
    bbox_losses = []
    giou_losses = []
    class_errors = []
    object_count = 0
    overflow_steps = 0
    map_detections = []
    map_image_ids = []
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        "\n[{}] epoch {}/{} | images={} | amp={} | log_every={}".format(
            phase, epoch + 1, config.epochs, expected,
            bool(config.amp and device.type == "cuda"), config.log_every,
        ),
        flush=True,
    )
    for index, (images, raw_targets) in enumerate(loader, start=1):
        if limit and index > limit:
            break
        images = [image.to(device) for image in images]
        targets = [move_target(target, device) for target in raw_targets]
        object_count += sum(int(target["labels"].numel()) for target in targets)
        if training:
            optimizer.zero_grad()
        amp_enabled = bool(config.amp and device.type == "cuda")
        gradient_norm = float("nan")
        amp_scale = float(scaler.get_scale()) if scaler is not None else 1.0
        with torch.set_grad_enabled(training):
            with autocast_context(amp_enabled):
                outputs = model(images)
                loss, loss_terms = detection_loss(criterion, outputs, targets)
                route_entropy = matched_routing_entropy(outputs, targets, criterion.matcher)
                teacher_loss, teacher_stats = teacher_guided_routing_loss(
                    outputs,
                    targets,
                    criterion.matcher,
                    teacher_utilities_for_targets(teacher_cache, targets),
                    temperature=config.teacher_target_temperature,
                    minimum_confidence=config.teacher_min_confidence,
                )
                total = (
                    loss
                    + config.teacher_routing_weight * teacher_loss
                    + config.routing_entropy_weight * route_entropy
                )
            if training:
                if scaler is not None and amp_enabled:
                    scale_before = float(scaler.get_scale())
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    # An AMP overflow is expected occasionally while the
                    # dynamic scaler calibrates.  Avoid clipping non-finite
                    # gradients; scaler.step will safely skip that update.
                    finite_gradients = all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                    if finite_gradients:
                        gradient_norm = float(clip_grad_norm_(
                            model.parameters(), config.clip_max_norm
                        ).detach().cpu())
                    scaler.step(optimizer)
                    scaler.update()
                    amp_scale = float(scaler.get_scale())
                    overflow_steps += int(amp_scale < scale_before)
                else:
                    total.backward()
                    gradient_norm = float(clip_grad_norm_(
                        model.parameters(), config.clip_max_norm
                    ).detach().cpu())
                    optimizer.step()
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite training/evaluation loss")
        total_value = float(total.detach().cpu())
        detection_value = float(loss.detach().cpu())
        teacher_value = float(teacher_loss.detach().cpu())
        matched_entropy = float(route_entropy.detach().cpu())
        statistics = routing_statistics(outputs)
        totals.append(total_value)
        detection_totals.append(detection_value)
        teacher_totals.append(teacher_value)
        teacher_confidences.append(teacher_stats["teacher_confidence"])
        teacher_agreements.append(teacher_stats["teacher_top1_agreement"])
        teacher_object_count += teacher_stats["teacher_objects"]
        matched_entropies.append(matched_entropy)
        route_entropies.append(statistics["routing_entropy"])
        route_peaks.append(statistics["routing_peak"])
        effective_branches.append(statistics["effective_branches"])
        branch_marginals.append(statistics["branch_marginal"])
        ce_losses.append(scalar_term(loss_terms, "loss_ce"))
        bbox_losses.append(scalar_term(loss_terms, "loss_bbox"))
        giou_losses.append(scalar_term(loss_terms, "loss_giou"))
        class_errors.append(scalar_term(loss_terms, "class_error"))
        if compute_map:
            probabilities = outputs["pred_logits"].softmax(-1)[0, :, :-1]
            scores, labels = probabilities.max(dim=-1)
            boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][0])
            height, width = raw_targets[0]["orig_size"].tolist()
            boxes = boxes * boxes.new_tensor([width, height, width, height])
            image_id = int(raw_targets[0]["image_id"])
            map_image_ids.append(image_id)
            for query in range(scores.numel()):
                x0, y0, x1, y1 = boxes[query].detach().cpu().tolist()
                map_detections.append({
                    "image_id": image_id,
                    "category_id": int(labels[query].detach().cpu()),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(scores[query].detach().cpu()),
                })
        if index == 1 or index % config.log_every == 0 or index == expected:
            elapsed = time.monotonic() - started
            seconds_per_image = elapsed / index
            eta = seconds_per_image * max(expected - index, 0)
            progress = 100.0 * index / expected
            lr_text = "-"
            if training:
                lr_text = ",".join("{:.2e}".format(group["lr"])
                                   for group in optimizer.param_groups)
            memory_text = "n/a"
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
                peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                memory_text = "{:.2f}/{:.2f}GiB".format(allocated, peak)
            top_routes = top_branch_summary(
                statistics["branch_marginal"], config.stops
            )
            print(
                "[{}] epoch={}/{} image={}/{} ({:.1f}%) | "
                "loss current/mean={:.4f}/{:.4f} det={:.4f} teacher={:.4f} | "
                "ce={:.4f} bbox={:.4f} giou={:.4f} class_err={:.1f}%".format(
                    phase, epoch + 1, config.epochs, index, expected, progress,
                    total_value, sum(totals) / len(totals), detection_value,
                    teacher_value,
                    ce_losses[-1], bbox_losses[-1], giou_losses[-1],
                    class_errors[-1],
                ),
                flush=True,
            )
            print(
                "[{}] routing matched_H={:.3f} all_H={:.3f} peak={:.3f} "
                "effective={:.2f} teacher_conf={:.3f} teacher_top1={:.1f}% "
                "teacher_objects={} | top stop:weight [{}]".format(
                    phase, matched_entropy, statistics["routing_entropy"],
                    statistics["routing_peak"],
                    statistics["effective_branches"],
                    teacher_stats["teacher_confidence"],
                    100.0 * teacher_stats["teacher_top1_agreement"],
                    teacher_stats["teacher_objects"],
                    top_routes,
                ),
                flush=True,
            )
            print(
                "[{}] speed={:.2f}s/image elapsed={} eta={} | "
                "lr={} grad_norm={} amp_scale={:.0f} overflows={} "
                "gpu_alloc/peak={}".format(
                    phase, seconds_per_image, format_duration(elapsed),
                    format_duration(eta), lr_text,
                    "{:.3f}".format(gradient_norm) if training else "n/a",
                    amp_scale, overflow_steps, memory_text,
                ),
                flush=True,
            )
    if not totals:
        raise RuntimeError("No images were processed")
    seconds = time.monotonic() - started
    mean_marginal = np.asarray(branch_marginals, dtype=np.float64).mean(axis=0)
    summary = {
        "images": len(totals),
        "objects": object_count,
        "loss": float(np.mean(totals)),
        "detection_loss": float(np.mean(detection_totals)),
        "teacher_routing_loss": float(np.mean(teacher_totals)),
        "teacher_objects": teacher_object_count,
        "teacher_confidence": float(np.mean(teacher_confidences)),
        "teacher_top1_agreement": float(np.mean(teacher_agreements)),
        "loss_ce": float(np.nanmean(ce_losses)),
        "loss_bbox": float(np.nanmean(bbox_losses)),
        "loss_giou": float(np.nanmean(giou_losses)),
        "class_error_percent": float(np.nanmean(class_errors)),
        "matched_routing_entropy": float(np.mean(matched_entropies)),
        "routing_entropy": float(np.mean(route_entropies)),
        "routing_peak": float(np.mean(route_peaks)),
        "effective_branches": float(np.mean(effective_branches)),
        "branch_marginal": [float(value) for value in mean_marginal],
        "top_branches": top_branch_summary(mean_marginal, config.stops),
        "seconds": seconds,
        "seconds_per_image": seconds / len(totals),
        "images_per_hour": 3600.0 * len(totals) / seconds,
        "amp_overflow_steps": overflow_steps,
        "gpu_peak_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3
            if device.type == "cuda" else None
        ),
    }
    if compute_map:
        summary.update(validation_map_summary(
            loader.dataset.annotations_path, map_detections, map_image_ids
        ))
    print("[{}] epoch summary:\n{}".format(
        phase, json.dumps(summary, indent=2)
    ), flush=True)
    return summary


def checkpoint_state(config, model, optimizer, epoch, best_validation,
                     selection_metric="detection_loss"):
    return {
        "architecture": ARCHITECTURE,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_loss": (
            best_validation if selection_metric == "detection_loss" else None
        ),
        "best_validation_map": (
            best_validation if selection_metric == "mean_AP_percent" else None
        ),
        "selection_metric": selection_metric,
        "fixed_stops": list(map(float, config.stops)),
        "detector_fingerprint": model.detector_fingerprint,
        "config": {
            "short_side": config.short_side,
            "max_side": config.max_side,
            "feature_chunk_size": config.feature_chunk_size,
            "route_temperature": config.route_temperature,
            "route_top_k": config.route_top_k,
            "radiometric_strength": config.radiometric_strength,
            "routing_entropy_weight": config.routing_entropy_weight,
            "teacher_routing_weight": config.teacher_routing_weight,
            "teacher_target_temperature": config.teacher_target_temperature,
            "teacher_min_confidence": config.teacher_min_confidence,
            "route_temperature_start": config.route_temperature_start,
            "curriculum_soft_epochs": config.curriculum_soft_epochs,
            "curriculum_middle_epochs": config.curriculum_middle_epochs,
            "curriculum_middle_top_k": config.curriculum_middle_top_k,
            "curriculum_total_epochs": config.curriculum_total_epochs or config.epochs,
            "strict_detector_freeze": True,
        },
    }


def save_checkpoint(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, model, criterion, train_loader, validation_loader, device,
          train_teacher_cache, validation_teacher_cache):
    optimizer = optimizer_for(config, model)
    scaler = make_grad_scaler(config.amp and device.type == "cuda")
    start_epoch = 0
    selection_metric = (
        "mean_AP_percent" if config.select_by_validation_map else "detection_loss"
    )
    best_validation = (
        float("-inf") if config.select_by_validation_map else float("inf")
    )
    if config.resume:
        resumed = load_checkpoint(config.resume)
        validate_router_checkpoint(resumed, config, model)
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"]) + 1
        saved_metric = resumed.get("selection_metric", "detection_loss")
        if saved_metric != selection_metric:
            raise ValueError("Resume checkpoint uses a different selection metric")
        if config.select_by_validation_map:
            best_validation = float(resumed["best_validation_map"])
        else:
            best_validation = float(resumed["best_validation_loss"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_started = time.monotonic()
    train_images = min(len(train_loader), config.limit_train_images) \
        if config.limit_train_images else len(train_loader)
    validation_images = min(len(validation_loader), config.limit_val_images) \
        if config.limit_val_images else len(validation_loader)
    print(json.dumps({
        "event": "training_start",
        "start_epoch": start_epoch + 1,
        "target_epochs": config.epochs,
        "train_images_per_epoch": train_images,
        "validation_images_per_epoch": validation_images,
        "fixed_stops": config.stops,
        "route_top_k": config.route_top_k,
        "feature_chunk_size": config.feature_chunk_size,
        "routing_lr": config.routing_lr,
        "teacher_routing_weight": config.teacher_routing_weight,
        "teacher_target_temperature": config.teacher_target_temperature,
        "teacher_min_confidence": config.teacher_min_confidence,
        "curriculum_soft_epochs": config.curriculum_soft_epochs,
        "curriculum_middle_epochs": config.curriculum_middle_epochs,
        "curriculum_middle_top_k": config.curriculum_middle_top_k,
        "curriculum_total_epochs": config.curriculum_total_epochs or config.epochs,
        "route_temperature_start": config.route_temperature_start,
        "route_temperature_final": config.route_temperature,
        "strict_detector_freeze": True,
        "amp": bool(config.amp and device.type == "cuda"),
        "resume": str(config.resume) if config.resume else None,
        "selection_metric": selection_metric,
        "best_validation_metric_at_start": best_validation,
    }, indent=2), flush=True)
    for epoch in range(start_epoch, config.epochs):
        print("\n========== EPOCH {}/{} ==========".format(
            epoch + 1, config.epochs
        ), flush=True)
        current_routing = curriculum_state(config, epoch)
        model.set_routing_state(
            current_routing["top_k"], current_routing["temperature"]
        )
        print(json.dumps({
            "event": "routing_curriculum",
            "epoch": epoch + 1,
            **current_routing,
        }, indent=2), flush=True)
        train_summary = run_loss_epoch(
            config, model, criterion, train_loader, device,
            teacher_cache=train_teacher_cache,
            optimizer=optimizer, limit=config.limit_train_images, scaler=scaler,
            epoch=epoch,
        )
        # Checkpoint selection always uses the final deployment budget rather
        # than an easier transient curriculum state.
        model.set_routing_state(config.route_top_k, config.route_temperature)
        validation_summary = run_loss_epoch(
            config, model, criterion, validation_loader, device,
            teacher_cache=validation_teacher_cache,
            limit=config.limit_val_images,
            epoch=epoch,
            compute_map=config.select_by_validation_map,
        )
        train_loss = train_summary["loss"]
        validation_loss = validation_summary["detection_loss"]
        validation_metric = (
            validation_summary["validation_mean_AP_percent"]
            if config.select_by_validation_map else validation_loss
        )
        is_best = (
            validation_metric > best_validation
            if config.select_by_validation_map
            else validation_metric < best_validation
        )
        if is_best:
            best_validation = validation_metric
        state = checkpoint_state(
            config, model, optimizer, epoch, best_validation, selection_metric
        )
        save_checkpoint(state, config.output_dir / "focer_v2_latest.pth")
        best_name = (
            "focer_v2_best_val_map.pth" if config.select_by_validation_map
            else "focer_v2_best_val_detection_loss.pth"
        )
        if is_best:
            save_checkpoint(state, config.output_dir / best_name)
        completed_this_session = epoch - start_epoch + 1
        session_elapsed = time.monotonic() - training_started
        mean_epoch_seconds = session_elapsed / completed_this_session
        remaining_seconds = mean_epoch_seconds * (config.epochs - epoch - 1)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_total_objective": validation_summary["loss"],
            "selection_metric": selection_metric,
            "validation_selection_value": validation_metric,
            "routing_curriculum": current_routing,
            "train": train_summary,
            "validation": validation_summary,
            "best_validation_metric": best_validation,
            "new_best_checkpoint": is_best,
            "fixed_stops": config.stops,
            "session_elapsed_seconds": session_elapsed,
            "estimated_remaining_seconds": remaining_seconds,
        }
        with (config.output_dir / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("\n========== EPOCH {}/{} COMPLETE ==========".format(
            epoch + 1, config.epochs
        ), flush=True)
        print(json.dumps(record, indent=2), flush=True)
        print(
            "checkpoint latest={} best={} new_best={} | session elapsed={} "
            "estimated remaining={}".format(
                config.output_dir / "focer_v2_latest.pth",
                config.output_dir / best_name,
                is_best,
                format_duration(session_elapsed),
                format_duration(remaining_seconds),
            ),
            flush=True,
        )


def fit_fixed_epochs(config, model, criterion, train_loader, device,
                     train_teacher_cache):
    """Refit a fresh router on all outer-training data for a fixed epoch count.

    The epoch count must have been selected without using the corresponding
    outer test fold.  This path intentionally accepts no validation data and
    therefore cannot inspect or select against the outer test fold.
    """
    optimizer = optimizer_for(config, model)
    scaler = make_grad_scaler(config.amp and device.type == "cuda")
    start_epoch = 0
    if config.resume:
        resumed = load_checkpoint(config.resume)
        validate_router_checkpoint(resumed, config, model)
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"]) + 1
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(json.dumps({
        "event": "fixed_epoch_refit_start",
        "start_epoch": start_epoch + 1,
        "target_epochs": config.epochs,
        "curriculum_total_epochs": config.curriculum_total_epochs or config.epochs,
        "train_images_per_epoch": len(train_loader),
        "outer_test_data_accessed": False,
        "strict_detector_freeze": True,
        "amp": bool(config.amp and device.type == "cuda"),
        "resume": str(config.resume) if config.resume else None,
    }, indent=2), flush=True)
    for epoch in range(start_epoch, config.epochs):
        print("\n========== REFIT EPOCH {}/{} ==========".format(
            epoch + 1, config.epochs
        ), flush=True)
        current_routing = curriculum_state(config, epoch)
        model.set_routing_state(
            current_routing["top_k"], current_routing["temperature"]
        )
        train_summary = run_loss_epoch(
            config, model, criterion, train_loader, device,
            teacher_cache=train_teacher_cache,
            optimizer=optimizer, limit=config.limit_train_images, scaler=scaler,
            epoch=epoch,
        )
        # Store every refit checkpoint in the final deployment state.
        model.set_routing_state(config.route_top_k, config.route_temperature)
        state = checkpoint_state(
            config, model, optimizer, epoch, float("inf")
        )
        save_checkpoint(state, config.output_dir / "focer_v2_latest.pth")
        record = {
            "epoch": epoch + 1,
            "mode": "outer_train_fixed_epoch_refit",
            "train_loss": train_summary["loss"],
            "routing_curriculum": current_routing,
            "train": train_summary,
            "fixed_stops": config.stops,
            "session_elapsed_seconds": time.monotonic() - started,
        }
        with (config.output_dir / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    latest = config.output_dir / "focer_v2_latest.pth"
    if not latest.is_file():
        raise RuntimeError("Fixed-epoch refit produced no checkpoint")
    final_path = config.output_dir / "focer_v2_fixed_epoch_refit.pth"
    save_checkpoint(load_checkpoint(latest), final_path)
    print(json.dumps({
        "event": "fixed_epoch_refit_complete",
        "epochs": config.epochs,
        "checkpoint": str(final_path),
        "outer_test_data_accessed": False,
    }, indent=2), flush=True)


def box_cxcywh_to_xyxy(boxes):
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack((
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        center_x + 0.5 * width,
        center_y + 0.5 * height,
    ), dim=-1)


@torch.no_grad()
def evaluate(config, model, loader, device):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    detections = []
    routing_records = []
    limit = config.limit_eval_images or len(loader)
    for index, (images, raw_targets) in enumerate(loader, start=1):
        if index > limit:
            break
        images = [image.to(device) for image in images]
        outputs = model(images)
        probabilities = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        scores, labels = probabilities.max(dim=-1)
        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][0])
        height, width = raw_targets[0]["orig_size"].tolist()
        scale = boxes.new_tensor([width, height, width, height])
        boxes = boxes * scale
        routes = outputs["routing_weights"][0]
        qualities = outputs["routing_quality"][0]
        image_id = int(raw_targets[0]["image_id"])
        for query_index in range(scores.numel()):
            score = float(scores[query_index].cpu())
            label = int(labels[query_index].cpu())
            if score >= config.score_threshold:
                x0, y0, x1, y1 = boxes[query_index].cpu().tolist()
                detections.append({
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score,
                })
            if score >= config.routing_log_threshold:
                weights = routes[query_index].cpu()
                quality = qualities[query_index].cpu()
                top_branch = int(weights.argmax())
                routing_records.append({
                    "image_id": image_id,
                    "query": query_index,
                    "category_id": label,
                    "score": score,
                    "top_branch": top_branch,
                    "top_stop": float(config.stops[top_branch]),
                    "weights": [float(value) for value in weights],
                    "radiometric_quality": [float(value) for value in quality],
                })
        if index == 1 or index % 20 == 0:
            stats = routing_statistics(outputs)
            print("evaluate image={}/{} route_peak={:.3f} effective_branches={:.2f}".format(
                index, limit, stats["routing_peak"], stats["effective_branches"]
            ), flush=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "focer_v2_detections.json").write_text(
        json.dumps(detections), encoding="utf-8"
    )
    (config.output_dir / "focer_v2_routing.json").write_text(
        json.dumps(routing_records, indent=2), encoding="utf-8"
    )
    if not detections:
        raise RuntimeError("No detections were produced")
    coco_gt = COCO(str(config.annotations))
    coco_dt = coco_gt.loadRes(detections)
    available = set(coco_gt.getCatIds())
    evaluated_image_ids = [item["id"] for item in loader.dataset.images[:limit]]
    categories = [
        category for category in PAPER_CATEGORIES
        if category in available
        and coco_gt.getAnnIds(imgIds=evaluated_image_ids, catIds=[category])
    ]
    skipped_categories = [category for category in PAPER_CATEGORIES if category not in categories]
    if not categories:
        raise RuntimeError("The evaluated images contain none of the requested paper categories")
    ap_by_category = {}
    for category in categories:
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.catIds = [category]
        if config.limit_eval_images:
            evaluator.params.imgIds = evaluated_image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        ap_by_category[str(category)] = float(evaluator.stats[0] * 100)
    report = {
        "architecture": ARCHITECTURE,
        "images": limit,
        "fixed_stops": list(map(float, config.stops)),
        "AP_percent_by_category": ap_by_category,
        "skipped_categories_without_ground_truth": skipped_categories,
        "mean_AP_percent": sum(ap_by_category.values()) / len(ap_by_category),
        "detections": len(detections),
        "routing_records": len(routing_records),
        "detector_fingerprint": model.detector_fingerprint,
        "nms": False,
    }
    (config.output_dir / "focer_v2_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


@torch.no_grad()
def inspect(config, model, loader, device):
    model.eval()
    images, targets = next(iter(loader))
    outputs = model([images[0].to(device)])
    probabilities = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    scores, labels = probabilities.max(dim=-1)
    order = scores.argsort(descending=True)[:10]
    rows = []
    for query in order.tolist():
        weights = outputs["routing_weights"][0, query].cpu()
        rows.append({
            "query": query,
            "category_id": int(labels[query]),
            "score": float(scores[query]),
            "top_stop": float(config.stops[int(weights.argmax())]),
            "route_weights": [round(float(value), 4) for value in weights],
        })
    print(json.dumps({
        "image_id": int(targets[0]["image_id"]),
        "fixed_stops": config.stops,
        "top_queries": rows,
    }, indent=2))


def preflight(config):
    loader = make_loader(config)
    dataset = loader.dataset
    count = min(config.preflight_samples, len(dataset))
    indices = sorted({
        round(index * (len(dataset) - 1) / max(count - 1, 1))
        for index in range(count)
    })
    samples = []
    for index in indices:
        info = dataset.images[index]
        raw = dataset.load_raw(info["file_name"])
        image, target = dataset[index]
        samples.append({
            "file": info["file_name"],
            "shape": list(raw.shape),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "hdr_scale": dataset.scale_for(info),
            "normalized_min": float(image.min()),
            "normalized_max": float(image.max()),
            "objects": int(target["labels"].numel()),
        })
    print(json.dumps({
        "images": len(dataset),
        "annotations": len(dataset.coco["annotations"]),
        "fixed_stops": config.stops,
        "samples": samples,
    }, indent=2))


def main(argv=None):
    config = arguments(argv)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.action == "preflight":
        preflight(config)
        return
    device = select_device(config.device)
    runtime = {
        "event": "runtime_configuration",
        "device": str(device),
        "architecture": ARCHITECTURE,
        "fixed_stops": config.stops,
        "route_top_k": config.route_top_k,
        "route_temperature": config.route_temperature,
        "radiometric_strength": config.radiometric_strength,
        "feature_chunk_size": config.feature_chunk_size,
        "image_short_side": config.short_side,
        "image_max_side": config.max_side,
        "amp_requested": config.amp,
        "seed": config.seed,
        "strict_detector_freeze": True,
        "balanced_sampling": config.balanced_sampling,
        "annotations": str(config.annotations),
        "mat_dir": str(config.mat_dir),
        "output_dir": str(config.output_dir),
    }
    if device.type == "cuda":
        runtime.update({
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_gib": (
                torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            ),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        })
    print(json.dumps(runtime, indent=2), flush=True)
    loader = make_loader(
        config,
        shuffle=config.action in ("train", "fit"),
        balanced=(config.action in ("train", "fit")
                  and config.balanced_sampling),
    )
    if config.action == "build_teacher_cache":
        build_teacher_cache(config, loader, device)
        return
    model, criterion, _ = make_model(config, device)
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    routing_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if parameter.requires_grad and model.is_router_parameter(name)
    )
    strict_count = model.assert_only_router_trainable()
    if trainable != routing_parameters or trainable != strict_count:
        raise RuntimeError("FOCER-v2 attempted to train a pretrained DETR parameter")
    print(json.dumps({
        "event": "model_ready",
        "dataset_images": len(loader.dataset),
        "parameters_total": total,
        "parameters_trainable": trainable,
        "parameters_routing": routing_parameters,
        "parameters_other_trainable": trainable - routing_parameters,
        "strict_detector_freeze_verified": True,
        "detector_fingerprint": model.detector_fingerprint,
    }, indent=2), flush=True)
    if config.action == "train":
        validation_loader = make_loader(
            config,
            annotations=config.val_annotations,
            mat_dir=config.val_mat_dir or config.mat_dir,
            shuffle=False,
        )
        verify_disjoint(loader.dataset, validation_loader.dataset)
        train_teacher_cache = load_teacher_cache(
            config.teacher_cache, loader.dataset, config, model.detector_fingerprint
        )
        validation_teacher_cache = load_teacher_cache(
            config.val_teacher_cache,
            validation_loader.dataset,
            config,
            model.detector_fingerprint,
        )
        train(
            config, model, criterion, loader, validation_loader, device,
            train_teacher_cache, validation_teacher_cache,
        )
    elif config.action == "fit":
        train_teacher_cache = load_teacher_cache(
            config.teacher_cache, loader.dataset, config, model.detector_fingerprint
        )
        fit_fixed_epochs(
            config, model, criterion, loader, device, train_teacher_cache
        )
    elif config.action == "evaluate":
        evaluate(config, model, loader, device)
    else:
        inspect(config, model, loader, device)


if __name__ == "__main__":
    main()
