#!/usr/bin/env python3
"""Run the audited FOCER-v2 HDR4RTT experiment without test teacher targets.

Expected defaults are documented in README.md. The runner builds teacher
supervision only for train+validation, selects the training duration using the
validation split, freshly refits, and accesses the test split only for final
FOCER and fixed-baseline evaluation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUBSET_TOOL = ROOT / "scripts" / "subset_teacher_cache.py"

EXPECTED_MASTER_SHA256 = (
    "47ba1074081ad3b26ed359f1fed2baba59b0779c31a8b860c477e26fb85b38b5"
)
EXPECTED_DETECTOR_SHA256 = (
    "e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=ROOT):
    print("\n$", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    subprocess.run(
        list(map(str, command)), cwd=cwd, env=environment, check=True
    )


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data",
        help="root relative to which annotation file_name paths are resolved",
    )
    parser.add_argument(
        "--master-annotations", type=Path,
        default=ROOT / "data" / "annotations" / "HDRDCL_TRAIN.json",
    )
    parser.add_argument(
        "--detector-checkpoint", type=Path,
        default=ROOT / "weights" / "detr-r50-e632da11.pth",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=ROOT / "runs" / "focer_v2_hdr4rtt_seed42_clean",
    )
    parser.add_argument("--selection-epochs", type=int, default=20)
    parser.add_argument("--feature-chunk-size", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--allow-resume", action="store_true",
        help="continue the same interrupted run after verifying its configuration",
    )
    return parser.parse_args()


def require_hash(path, expected):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Unexpected SHA-256 for {path}: expected {expected}, found {actual}"
        )
    return path


def main():
    config = arguments()
    data_root = config.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    master = require_hash(config.master_annotations, EXPECTED_MASTER_SHA256)
    detector = require_hash(config.detector_checkpoint, EXPECTED_DETECTOR_SHA256)
    if not SUBSET_TOOL.is_file():
        raise FileNotFoundError(SUBSET_TOOL)

    run_root = config.run_root.expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()) and not config.allow_resume:
        raise FileExistsError(
            f"Run directory is not empty: {run_root}. Choose a new --run-root "
            "or pass --allow-resume to continue the same run."
        )

    split_dir = run_root / "splits"
    cache_root = run_root / "teacher_cache"
    subset_dir = cache_root / "subsets"
    trainval_cache = cache_root / "trainval1432_teacher_v1.pth"
    run_root.mkdir(parents=True, exist_ok=True)

    hardened_configuration = {
        "protocol": "hdr4rtt_source_temporal_guard_v1",
        "master_annotations_sha256": EXPECTED_MASTER_SHA256,
        "detector_sha256": EXPECTED_DETECTOR_SHA256,
        "cache_policy": "teacher cache built on train+validation only",
        "selection_epochs": config.selection_epochs,
        "seed": config.seed,
        "fixed_stops": list(range(15)),
        "route_top_k": 3,
        "strict_detector_freeze": True,
    }
    configuration_path = run_root / "HARDENED_RUN_CONFIGURATION.json"
    if configuration_path.is_file():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != hardened_configuration:
            raise ValueError(
                "The existing run directory uses a different configuration"
            )
    else:
        configuration_path.write_text(
            json.dumps(hardened_configuration, indent=2), encoding="utf-8"
        )

    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run([
        sys.executable, "hdr4rtt_experiment.py", "prepare",
        "--master-annotations", master,
        "--split-dir", split_dir,
    ])
    run([
        sys.executable, "focer_main.py", "preflight",
        "--annotations", split_dir / "trainval.json",
        "--mat-dir", data_root,
        "--preflight-samples", "8",
    ])

    if not trainval_cache.is_file():
        trainval_cache.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "-u", "focer_main.py", "build_teacher_cache",
            "--annotations", split_dir / "trainval.json",
            "--mat-dir", data_root,
            "--detector-checkpoint", detector,
            "--teacher-cache", trainval_cache,
            "--feature-chunk-size", config.feature_chunk_size,
            "--log-every", config.log_every,
            "--device", config.device,
        ])
    else:
        print("Reusing train+validation-only teacher cache:", trainval_cache)

    subset_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "trainval"):
        run([
            sys.executable, SUBSET_TOOL,
            "--parent-cache", trainval_cache,
            "--annotations", split_dir / f"{name}.json",
            "--output", subset_dir / f"{name}_teacher_v1.pth",
        ])

    command = [
        sys.executable, "-u", "hdr4rtt_experiment.py", "run",
        "--split-dir", split_dir,
        "--cache-dir", subset_dir,
        "--mat-dir", data_root,
        "--detector-checkpoint", detector,
        "--run-root", run_root,
        "--selection-epochs", config.selection_epochs,
        "--feature-chunk-size", config.feature_chunk_size,
        "--log-every", config.log_every,
        "--seed", config.seed,
        "--device", config.device,
    ]
    if config.amp:
        command.append("--amp")
    run(command)

    print("\nCompleted clean experiment:", run_root)
    print("Final comparison:", run_root / "FINAL_TEST_COMPARISON.json")


if __name__ == "__main__":
    main()
