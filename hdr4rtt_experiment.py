"""Leak-resistant same-source HDR4RTT train/validation/test experiment.

The FOCER-v2 architecture is unchanged.  This wrapper creates an immutable
source-aware 80/10/10 split, selects an epoch on validation mAP, freshly refits
on train+validation, and compares the untouched test set with the paper's fixed
15-exposure detector.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from split_manifest import MASTER_ANNOTATIONS_SHA256, load_manifest


PAPER_CATEGORIES = (1, 2, 3, 8)
EXPECTED_DETECTOR_SHA256 = "e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc"
EXPECTED_SIZES = {"train": 1264, "val": 168, "test": 150, "guard": 14}
EXPECTED_OBJECTS = {
    "train": {1: 4492, 2: 40, 3: 1342, 8: 35},
    "val": {1: 557, 2: 8, 3: 168, 8: 8},
    "test": {1: 549, 2: 8, 3: 164, 8: 8},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, payload, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=None if compact else 2,
                   separators=(",", ":") if compact else None),
        encoding="utf-8",
    )
    temporary.replace(path)


def split_stats(coco):
    objects = Counter()
    images_by_category = defaultdict(set)
    for annotation in coco["annotations"]:
        category = int(annotation["category_id"])
        if category in PAPER_CATEGORIES:
            objects[category] += 1
            images_by_category[category].add(int(annotation["image_id"]))
    return {
        "images": len(coco["images"]),
        "annotations_all_categories": len(coco["annotations"]),
        "objects_by_paper_category": {str(key): objects[key] for key in PAPER_CATEGORIES},
        "images_by_paper_category": {
            str(key): len(images_by_category[key]) for key in PAPER_CATEGORIES
        },
    }


def subset_coco(master, image_ids, split_name):
    image_ids = set(map(int, image_ids))
    result = {key: value for key, value in master.items()
              if key not in ("images", "annotations")}
    result["images"] = [item for item in master["images"]
                        if int(item["id"]) in image_ids]
    found = {int(item["id"]) for item in result["images"]}
    if found != image_ids:
        raise ValueError("Split references unknown image IDs: {}".format(sorted(image_ids - found)[:10]))
    result["annotations"] = [item for item in master["annotations"]
                             if int(item["image_id"]) in image_ids]
    result["info"] = dict(result.get("info", {}), hdr4rtt_split=split_name)
    return result


def prepare_splits(master_path, split_dir):
    master_path, split_dir = Path(master_path), Path(split_dir)
    if sha256_file(master_path) != MASTER_ANNOTATIONS_SHA256:
        raise ValueError("Master HDR4RTT annotation SHA-256 differs from the audited file")
    master = read_json(master_path)
    manifest = load_manifest()
    assignments = {name: set(map(int, values))
                   for name, values in manifest["assignments"].items()}
    actual_ids = {int(item["id"]) for item in master["images"]}
    if len(actual_ids) != 1596:
        raise ValueError("Expected 1596 unique HDR4RTT images")
    names = ("train", "val", "test", "guard")
    union = set()
    for name in names:
        if len(assignments[name]) != EXPECTED_SIZES[name]:
            raise RuntimeError("Unexpected {} split size".format(name))
        if union & assignments[name]:
            raise RuntimeError("Split overlap involving {}".format(name))
        union |= assignments[name]
    if union != actual_ids:
        raise RuntimeError("Built-in assignments do not exactly partition HDR4RTT")

    split_dir.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for name in names:
        payloads[name] = subset_coco(master, assignments[name], name)
    payloads["trainval"] = subset_coco(
        master, assignments["train"] | assignments["val"], "trainval_refit"
    )
    summary = {
        "protocol": manifest["protocol_version"],
        "seed": manifest["seed"],
        "master_annotations_sha256": MASTER_ANNOTATIONS_SHA256,
        "paper_categories": list(PAPER_CATEGORIES),
        "test_access_policy": "test is not used for epoch selection or fitting",
        "selection": "maximum validation pooled paper-category COCO mAP",
        "refit": "fresh initialization on train+validation for the selected epoch count",
        "guard_radius_original_frames": manifest["cross_split_guard_radius_frames"],
        "group_assignments": manifest["group_assignments"],
        "splits": {},
    }
    for name, payload in payloads.items():
        path = split_dir / (name + ".json")
        write_json(path, payload)
        stats = split_stats(payload)
        if name in EXPECTED_OBJECTS:
            actual = {int(key): value for key, value in stats["objects_by_paper_category"].items()}
            if actual != EXPECTED_OBJECTS[name]:
                raise RuntimeError("{} category counts differ from the audit".format(name))
        if name in ("train", "val", "test") and any(
                stats["objects_by_paper_category"][str(cat)] == 0 for cat in PAPER_CATEGORIES):
            raise RuntimeError("{} is missing a paper category".format(name))
        summary["splits"][name] = {
            **stats, "annotations_sha256": sha256_file(path), "file": path.name,
        }
    summary["guard_image_ids"] = sorted(assignments["guard"])
    write_json(split_dir / "split_manifest.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def load_torch(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def subset_teacher_cache(full_cache_path, annotations_path, output_path):
    import torch
    full = load_torch(full_cache_path)
    if full.get("annotations_sha256") != MASTER_ANNOTATIONS_SHA256:
        raise ValueError("Teacher cache was not built from the audited HDR4RTT master JSON")
    split = read_json(annotations_path)
    ids = {str(item["id"]) for item in split["images"]}
    entries = full.get("entries", {})
    if not ids <= set(entries):
        raise ValueError("Full teacher cache is missing {} split IDs".format(len(ids - set(entries))))
    payload = {key: value for key, value in full.items()
               if key not in ("entries", "images", "objects", "annotations_sha256")}
    payload["annotations_sha256"] = sha256_file(annotations_path)
    payload["images"] = len(ids)
    payload["entries"] = {key: entries[key] for key in sorted(ids, key=int)}
    payload["objects"] = sum(int(entry["labels"].numel())
                             for entry in payload["entries"].values())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)


def prepare_subset_caches(full_cache, split_dir, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "trainval"):
        annotations = Path(split_dir) / (name + ".json")
        destination = cache_dir / (name + "_teacher_v1.pth")
        expected = sha256_file(annotations)
        reusable = destination.is_file() and load_torch(destination).get("annotations_sha256") == expected
        if reusable:
            print("reusing", destination, flush=True)
        else:
            subset_teacher_cache(full_cache, annotations, destination)
            print("created", destination, flush=True)


def completed_epochs(path):
    path = Path(path)
    return int(load_torch(path)["epoch"]) + 1 if path.is_file() else 0


def run_streamed(command, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\nCOMMAND " + " ".join(map(str, command)) + "\n")
        process = subprocess.Popen(
            list(map(str, command)), cwd=cwd, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError("Command failed with exit code {}".format(code))


def run_experiment(args):
    project = Path(__file__).resolve().parent
    split_dir, cache_dir, run_root = map(Path, (args.split_dir, args.cache_dir, args.run_root))
    selection_dir = run_root / "selection_train1264_val168"
    refit_dir = run_root / "refit_trainval1432"
    test_focer = run_root / "test150_focer"
    test_fixed = run_root / "test150_paper_fixed"
    for directory in (selection_dir, refit_dir, test_focer, test_fixed):
        directory.mkdir(parents=True, exist_ok=True)
    required = [
        split_dir / "train.json", split_dir / "val.json", split_dir / "trainval.json",
        split_dir / "test.json", cache_dir / "train_teacher_v1.pth",
        cache_dir / "val_teacher_v1.pth", cache_dir / "trainval_teacher_v1.pth",
        Path(args.detector_checkpoint),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment inputs: {}".format(missing))
    if sha256_file(args.detector_checkpoint) != EXPECTED_DETECTOR_SHA256:
        raise ValueError("Detector is not the audited official DETR-R50 checkpoint")

    common = [
        sys.executable, "-u", "focer_main.py", "--mat-dir", args.mat_dir,
        "--detector-checkpoint", args.detector_checkpoint,
        "--stops", "0:14:1", "--route-top-k", "3",
        "--feature-chunk-size", str(args.feature_chunk_size),
        "--routing-lr", "1e-4", "--device", args.device,
        "--workers", "0", "--seed", str(args.seed),
        "--log-every", str(args.log_every),
        "--curriculum-total-epochs", str(args.selection_epochs),
    ]
    amp = ["--amp"] if args.amp else []
    selection_latest = selection_dir / "focer_v2_latest.pth"
    if completed_epochs(selection_latest) < args.selection_epochs:
        command = common[:3] + ["train"] + common[3:] + [
            "--annotations", split_dir / "train.json",
            "--val-annotations", split_dir / "val.json",
            "--val-mat-dir", args.mat_dir,
            "--teacher-cache", cache_dir / "train_teacher_v1.pth",
            "--val-teacher-cache", cache_dir / "val_teacher_v1.pth",
            "--output-dir", selection_dir,
            "--epochs", str(args.selection_epochs), "--select-by-validation-map",
        ] + amp
        if selection_latest.is_file():
            command += ["--resume", selection_latest]
        run_streamed(command, project, selection_dir / "console.log")
    else:
        print("selection training already complete", flush=True)

    selected = selection_dir / "focer_v2_best_val_map.pth"
    if not selected.is_file():
        raise RuntimeError("No validation-mAP-selected checkpoint was produced")
    selected_epochs = int(load_torch(selected)["epoch"]) + 1
    selection_record = {
        "selected_epochs": selected_epochs,
        "maximum_selection_epochs": args.selection_epochs,
        "selection_criterion": "maximum validation pooled paper-category COCO mAP",
        "selection_checkpoint_sha256": sha256_file(selected),
        "test_images_accessed": False,
    }
    write_json(run_root / "selection_record.json", selection_record)

    refit_configuration = {
        "selected_epochs": selected_epochs,
        "curriculum_total_epochs": args.selection_epochs,
        "images": 1432,
        "fresh_initialization": True,
        "test_images_accessed": False,
    }
    configuration_path = refit_dir / "refit_configuration.json"
    if configuration_path.is_file() and read_json(configuration_path) != refit_configuration:
        raise ValueError("Existing refit directory was created with a different selected epoch")
    write_json(configuration_path, refit_configuration)
    refit_latest = refit_dir / "focer_v2_latest.pth"
    if completed_epochs(refit_latest) < selected_epochs:
        command = common[:3] + ["fit"] + common[3:] + [
            "--annotations", split_dir / "trainval.json",
            "--teacher-cache", cache_dir / "trainval_teacher_v1.pth",
            "--output-dir", refit_dir, "--epochs", str(selected_epochs),
        ] + amp
        if refit_latest.is_file():
            command += ["--resume", refit_latest]
        run_streamed(command, project, refit_dir / "console.log")
    else:
        print("train+validation refit already complete", flush=True)
    refit_checkpoint = refit_dir / "focer_v2_fixed_epoch_refit.pth"
    if not refit_checkpoint.is_file():
        import torch
        temporary = refit_checkpoint.with_suffix(".pth.tmp")
        torch.save(load_torch(refit_latest), temporary)
        temporary.replace(refit_checkpoint)

    focer_metrics = test_focer / "focer_v2_metrics.json"
    if not focer_metrics.is_file():
        command = common[:3] + ["evaluate"] + common[3:] + [
            "--annotations", split_dir / "test.json",
            "--router-checkpoint", refit_checkpoint,
            "--output-dir", test_focer,
        ]
        run_streamed(command, project, test_focer / "console.log")
    else:
        print("FOCER test evaluation already complete", flush=True)

    fixed_metrics = test_fixed / "fixed_metrics.json"
    if not fixed_metrics.is_file():
        command = [
            sys.executable, "-u", "adaptive_main.py", "evaluate",
            "--annotations", split_dir / "test.json", "--mat-dir", args.mat_dir,
            "--detector-checkpoint", args.detector_checkpoint,
            "--output-dir", test_fixed, "--device", args.device,
            "--branches", "15", "--workers", "0", "--seed", str(args.seed),
            "--baseline-only",
        ]
        run_streamed(command, project / "baseline", test_fixed / "console.log")
    else:
        print("paper fixed test evaluation already complete", flush=True)
    summarize(split_dir, run_root)


def summarize(split_dir, run_root):
    split_dir, run_root = Path(split_dir), Path(run_root)
    split_manifest = read_json(split_dir / "split_manifest.json")
    focer = read_json(run_root / "test150_focer" / "focer_v2_metrics.json")
    fixed = read_json(run_root / "test150_paper_fixed" / "fixed_metrics.json")
    per_category = {
        str(category): focer["AP_percent_by_category"][str(category)]
        - fixed["AP_percent_by_category"][str(category)]
        for category in PAPER_CATEGORIES
    }
    report = {
        "protocol": split_manifest["protocol"],
        "test_images": split_manifest["splits"]["test"]["images"],
        "selected_epochs": read_json(run_root / "selection_record.json")["selected_epochs"],
        "paper_fixed": fixed,
        "focer_v2": focer,
        "focer_minus_paper_AP_points_by_category": per_category,
        "focer_minus_paper_mean_AP_points": focer["mean_AP_percent"] - fixed["mean_AP_percent"],
        "interpretation": "positive delta means FOCER-v2 is better",
    }
    write_json(run_root / "FINAL_TEST_COMPARISON.json", report)
    print("\n========== FINAL UNTOUCHED TEST RESULT ==========", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cache_subsets", "run", "summarize"))
    parser.add_argument("--master-annotations", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--full-teacher-cache", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--mat-dir", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--selection-epochs", type=int, default=20)
    parser.add_argument("--feature-chunk-size", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def require(config, *names):
    missing = ["--" + name.replace("_", "-") for name in names
               if getattr(config, name) is None]
    if missing:
        raise ValueError("Missing required arguments: {}".format(", ".join(missing)))


def main(argv=None):
    config = arguments(argv)
    if config.action == "prepare":
        require(config, "master_annotations", "split_dir")
        prepare_splits(config.master_annotations, config.split_dir)
    elif config.action == "cache_subsets":
        require(config, "full_teacher_cache", "split_dir", "cache_dir")
        prepare_subset_caches(config.full_teacher_cache, config.split_dir, config.cache_dir)
    elif config.action == "run":
        require(config, "split_dir", "cache_dir", "mat_dir", "detector_checkpoint", "run_root")
        run_experiment(config)
    else:
        require(config, "split_dir", "run_root")
        summarize(config.split_dir, config.run_root)


if __name__ == "__main__":
    main()
