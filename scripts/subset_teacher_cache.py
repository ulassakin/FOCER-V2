#!/usr/bin/env python3
"""Create an annotation-hash-bound subset from a parent teacher cache."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parent = load(args.parent_cache)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    ids = {str(int(item["id"])) for item in annotations["images"]}
    entries = parent.get("entries", {})
    missing = ids - set(entries)
    if missing:
        raise ValueError(f"Parent cache is missing {len(missing)} requested image IDs")

    payload = {
        key: value for key, value in parent.items()
        if key not in ("entries", "images", "objects", "annotations_sha256")
    }
    payload["annotations_sha256"] = sha256(args.annotations)
    payload["images"] = len(ids)
    payload["entries"] = {key: entries[key] for key in sorted(ids, key=int)}
    payload["objects"] = sum(
        int(entry["labels"].numel()) for entry in payload["entries"].values()
    )

    if args.output.is_file():
        existing = load(args.output)
        if (existing.get("annotations_sha256") == payload["annotations_sha256"]
                and set(existing.get("entries", {})) == ids):
            print("Reusing verified subset cache:", args.output)
            return
        raise ValueError(f"Refusing to overwrite mismatched cache: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print("Created", args.output, "images=", payload["images"],
          "objects=", payload["objects"])


if __name__ == "__main__":
    main()
