"""Read COCO boxes with the paper's MAT files or HDRDCL HDR/EXR images."""

import json
import os
from pathlib import Path

# OpenCV's EXR codec is opt-in in some builds. Set this before importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class HDRMatDetection(Dataset):
    def __init__(self, annotations, mat_dir, short_side=800, max_side=1333,
                 mat_key="rgb_dem", hdr_scale=255.0, source_scales=None):
        self.annotations_path = Path(annotations)
        self.mat_dir = Path(mat_dir)
        if not self.annotations_path.is_file():
            raise FileNotFoundError(self.annotations_path)
        if not self.mat_dir.is_dir():
            raise FileNotFoundError(self.mat_dir)
        with self.annotations_path.open(encoding="utf-8") as stream:
            self.coco = json.load(stream)
        if not np.isfinite(hdr_scale) or hdr_scale <= 0:
            raise ValueError("hdr_scale must be a positive, finite number")
        self.mat_key = mat_key
        self.hdr_scale = float(hdr_scale)
        self.source_scales = {}
        for source, scale in (source_scales or {}).items():
            if not source or "/" in source or "\\" in source:
                raise ValueError("Source-scale keys must be top-level source directory names")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Every source scale must be a positive, finite number")
            self.source_scales[str(source)] = float(scale)
        self.images = self.coco["images"]
        for info in self.images:
            if "hdr_scale" in info and (not np.isfinite(info["hdr_scale"])
                                         or info["hdr_scale"] <= 0):
                raise ValueError("Image hdr_scale must be a positive, finite number")
        image_ids = [info["id"] for info in self.images]
        if not self.images or len(image_ids) != len(set(image_ids)):
            raise ValueError("Annotation JSON needs nonempty images with unique IDs")
        image_id_set = set(image_ids)
        names = [Path(info["file_name"]).as_posix() for info in self.images]
        if len(names) != len(set(names)):
            raise ValueError("Multiple images have the same filename")
        category_ids = {cat["id"] for cat in self.coco.get("categories", [])}
        if not category_ids:
            raise ValueError("COCO annotation JSON is missing category definitions")
        self.annotations = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] not in image_id_set:
                raise ValueError("An annotation references an unknown image ID")
            if not 1 <= ann["category_id"] <= 90:
                raise ValueError("DETR requires COCO category IDs in 1..90; check the label mapping")
            if ann["category_id"] not in category_ids:
                raise ValueError("An annotation uses a category absent from COCO categories")
            if not ann.get("iscrowd", 0):
                self.annotations.setdefault(ann["image_id"], []).append(ann)
        self.short_side = short_side
        self.max_side = max_side

    def __len__(self):
        return len(self.images)

    def validate_paths(self):
        missing = [info["file_name"] for info in self.images
                   if not any(path.is_file() for path in self._mat_candidates(info["file_name"]))]
        if missing:
            raise FileNotFoundError("Missing {} HDR files in {}. Examples: {}".format(
                len(missing), self.mat_dir, missing[:5]))

    def _mat_candidates(self, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Image filename must be a safe path relative to --mat-dir")
        stem = relative.stem
        candidates = []
        if relative.suffix.lower() in {".mat", ".hdr", ".exr"}:
            candidates.append(self.mat_dir / relative)
        # The paper's JSON points to JPEGs, while its HDR MAT files are flat.
        candidates.extend((self.mat_dir / (stem + ".mat"),
                           self.mat_dir / (stem + "_dem.mat")))
        return candidates

    def _mat_path(self, name):
        for path in self._mat_candidates(name):
            if path.is_file():
                return path
        raise FileNotFoundError("No HDR image for {} in {}".format(name, self.mat_dir))

    def scale_for(self, image):
        if isinstance(image, dict):
            if "hdr_scale" in image:
                return float(image["hdr_scale"])
            name = image["file_name"]
        else:
            name = image
        parts = Path(name).parts
        source = parts[0] if len(parts) > 1 else None
        return self.source_scales.get(source, self.hdr_scale)

    def load_raw(self, name):
        path = self._mat_path(name)
        if path.suffix.lower() == ".mat":
            mat = scipy.io.loadmat(path)
            if self.mat_key not in mat:
                raise KeyError("MAT file does not contain {}".format(self.mat_key))
            # Preserve the paper MAT's original float64 values for its exact
            # NumPy/OpenCV evaluation path. __getitem__ converts training
            # tensors to float32 below.
            rgb = np.asarray(mat[self.mat_key])
        else:
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ValueError("OpenCV cannot read HDR image {}".format(path))
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError("HDR image must have three colour channels: {}".format(path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("HDR image must have H x W x 3 shape: {}".format(path))
        if not np.isfinite(rgb).all():
            raise ValueError("Non-finite values in {}".format(path))
        return rgb

    def __getitem__(self, index):
        info = self.images[index]
        rgb = self.load_raw(info["file_name"])
        height, width = rgb.shape[:2]
        if ("height" in info and info["height"] != height) or ("width" in info and info["width"] != width):
            raise ValueError("MAT dimensions differ from annotation dimensions for {}".format(info["file_name"]))
        # Default reproduces the released paper code: clip(rgb_dem / 255, 0, 1).
        # Other HDR datasets must explicitly choose their own radiance scale.
        hdr = (torch.from_numpy(rgb.astype(np.float32, copy=True)).permute(2, 0, 1)
               / self.scale_for(info)).clamp(0, 1)
        labels = []
        boxes = []
        for ann in self.annotations.get(info["id"], []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(width), x + w), min(float(height), y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            labels.append(ann["category_id"])
            boxes.append([((x0 + x1) / 2) / width, ((y0 + y1) / 2) / height,
                          (x1 - x0) / width, (y1 - y0) / height])
        target = {
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1),
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }
        return hdr, target


def collate_hdr(batch):
    return [item[0] for item in batch], [item[1] for item in batch]
