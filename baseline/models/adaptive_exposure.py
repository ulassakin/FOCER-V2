"""Content-adaptive exposure branches for the unchanged DETR detector.

The policy observes low-resolution HDR luminance statistics and redistributes
ordered exposure stops inside the paper baseline's fixed exposure interval.
The detector sees the same gamma-mapped, ImageNet-normalized images as the
original multi-exposure evaluation path. All operations between policy and
detector are differentiable.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveExposurePolicy(nn.Module):
    def __init__(self, branches=15, stop_min=0.0, stop_max=None,
                 min_gap_fraction=0.25, sample_size=64):
        super().__init__()
        if branches < 3:
            raise ValueError("adaptive exposure placement needs at least 3 branches")
        if stop_max is None:
            stop_max = branches - 1
        if not stop_max > stop_min or not 0 < min_gap_fraction < 1:
            raise ValueError("Invalid exposure span or spacing bounds")
        self.branches = branches
        self.stop_min = float(stop_min)
        self.stop_max = float(stop_max)
        self.min_gap_fraction = min_gap_fraction
        self.sample_size = sample_size
        self.register_buffer("quantile_levels", torch.tensor(
            [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]))
        self.network = nn.Sequential(
            nn.Linear(9, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            # K-1 relative gap logits inside the fixed paper interval.
            nn.Linear(64, branches - 1),
        )
        # Start exactly at the paper's evenly spaced fixed-stop baseline.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _statistics(self, hdr):
        # Only policy parameters are learned. CPU quantiles work across CUDA,
        # MPS, and CPU even where an accelerator lacks torch.quantile support.
        small = F.interpolate(hdr.detach().to("cpu"),
                              size=(self.sample_size, self.sample_size),
                              mode="area")
        luminance = (0.2126 * small[:, 0] + 0.7152 * small[:, 1]
                     + 0.0722 * small[:, 2]).clamp_min(0)
        log_luminance = torch.log2(luminance + 1e-6)
        quantiles = []
        dark_fraction = []
        bright_fraction = []
        for image in log_luminance:
            valid = image[image > math.log2(1e-5)]
            if valid.numel() == 0:
                valid = image.flatten()
            q = torch.quantile(valid, self.quantile_levels.cpu())
            quantiles.append(q)
            dark_fraction.append((image < math.log2(0.01)).float().mean())
            # The loader clips radiance to [0, 1], so a threshold of 1.0
            # would make this input feature identically zero.
            bright_fraction.append((image > math.log2(0.8)).float().mean())
        q = torch.stack(quantiles)
        extras = torch.stack((torch.stack(dark_fraction),
                              torch.stack(bright_fraction)), dim=1)
        return q.to(hdr.device), extras.to(hdr.device)

    def forward(self, hdr, statistics=None):
        if hdr.ndim != 4 or hdr.shape[1] != 3:
            raise ValueError("Expected HDR tensor with shape [batch, 3, H, W]")
        q, extras = self._statistics(hdr) if statistics is None else statistics
        features = torch.cat((q.clamp(-24, 24) / 24, extras), dim=1)
        proportions = torch.softmax(self.network(features), dim=1)
        # Every positive gap receives a floor; softmax learns where the
        # remaining fixed span goes. Endpoints therefore stay identical to the
        # paper baseline and branches cannot collapse at either boundary.
        span = q.new_tensor(self.stop_max - self.stop_min)
        gaps = span * (self.min_gap_fraction / (self.branches - 1)
                       + (1 - self.min_gap_fraction) * proportions)
        start = q.new_full((q.shape[0], 1), self.stop_min)
        interior = start + gaps.cumsum(dim=1)[:, :-1]
        end = q.new_full((q.shape[0], 1), self.stop_max)
        # Construct both boundaries explicitly. Besides expressing the fixed
        # action space directly, this avoids a tiny floating-point drift in
        # the final cumulative sum and removes any spurious endpoint gradient.
        return torch.cat((start, interior, end), dim=1)

    def coverage_loss(self, hdr, stops, lower=0.03, upper=0.95,
                      softness=0.35):
        """Softly penalize radiance values missed by every exposure branch.

        This is a coverage preference, not a claim that K finite exposures can
        recover an arbitrary HDR range or information absent from the sensor.
        """
        small = F.interpolate(hdr.detach().to("cpu"),
                              size=(self.sample_size, self.sample_size),
                              mode="area")
        luminance = (0.2126 * small[:, 0] + 0.7152 * small[:, 1]
                     + 0.0722 * small[:, 2]).clamp_min(0)
        log_luminance = torch.log2(luminance + 1e-6).to(stops.device)
        luminance = luminance.to(stops.device)
        exposed_log = log_luminance[:, None] + stops[:, :, None, None]
        # A soft distance to the useful exposure interval retains a gradient
        # even when *every* branch is currently too dark or too bright. The
        # previous product-of-probabilities penalty could saturate there.
        below = F.softplus((math.log2(lower) - exposed_log) / softness) * softness
        above = F.softplus((exposed_log - math.log2(upper)) / softness) * softness
        distance = below + above
        nearest = -softness * torch.logsumexp(-distance / softness, dim=1)
        uncovered = F.softplus(nearest / softness) * softness
        valid = (luminance > 1e-5).float()
        return (uncovered * valid).sum() / valid.sum().clamp_min(1)


class AdaptiveExposureDETR(nn.Module):
    """A shared pretrained DETR applied to K learned exposure renderings."""

    def __init__(self, detector, branches=15, short_side=800, max_side=1333,
                 **policy_kwargs):
        super().__init__()
        self.detector = detector
        self.policy = AdaptiveExposurePolicy(branches=branches, **policy_kwargs)
        self.short_side = short_side
        self.max_side = max_side
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def freeze_detector(self):
        self.detector.eval()
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)

    def render_branch(self, hdr, stops, branch):
        return self._render_at_stop(hdr, stops[:, branch])

    def _render_at_stop(self, hdr, stop):
        gain = torch.exp2(stop).view(-1, 1, 1, 1)
        exposed = (hdr * gain).clamp(0, 1)
        # A fractional power has infinite derivative at zero. Preserve exact
        # black while keeping the exposure gradient finite during training.
        ldr = torch.where(exposed > 0,
                          exposed.clamp_min(1e-8).pow(1 / 2.2),
                          torch.zeros_like(exposed))
        # The released baseline tone maps first, then resizes for DETR.
        height, width = ldr.shape[-2:]
        # Match the integer output-size calculation in result_eval.resizenp.
        size = self.short_side
        if max(height, width) / min(height, width) * size > self.max_side:
            size = int(round(self.max_side * min(height, width) / max(height, width)))
        if width < height:
            new_size = (int(size * height / width), size)
        else:
            new_size = (size, int(size * width / height))
        ldr = F.interpolate(ldr, size=new_size, mode="bilinear",
                            align_corners=False)
        return (ldr - self.imagenet_mean) / self.imagenet_std

    def render_branches(self, hdr, stops, branches=None):
        """Render several exposure branches as one detector batch.

        Training and evaluation currently use one HDR scene at a time.  The
        exposure dimension is therefore the detector batch dimension, matching
        the released MEL implementation's single ``model(inputs)`` call.
        """
        if hdr.ndim != 4 or hdr.shape[0] != 1:
            raise ValueError("Batched exposure rendering expects one HDR image")
        if branches is None:
            branches = range(stops.shape[1])
        else:
            branches = list(branches)
        # Render/resize before stacking, as in the paper's released code.  This
        # avoids materializing K full-resolution HDR tensors at the same time;
        # the expensive detector still receives a parallel K-image batch.
        return torch.cat([self._render_at_stop(hdr, stops[:, branch])
                          for branch in branches])

    def detect_branch(self, hdr_images, stops, branch):
        return self.detect_branches(hdr_images, stops, [branch])

    def detect_branches(self, hdr_images, stops, branches=None):
        """Run several rendered exposures through shared DETR in one call."""
        rendered = self.render_branches(hdr_images, stops, branches)
        return self.detector(list(rendered))

    def forward(self, hdr_images):
        """Return a batched DETR output, preserving exposure-branch order.

        For now training/evaluation use one HDR scene at a time to avoid
        padding bright HDR radiance with invented data.
        """
        if len(hdr_images) != 1:
            raise ValueError("Use batch size 1 for adaptive HDR detection")
        hdr = torch.stack(hdr_images)
        stops = self.policy(hdr)
        return self.detect_branches(hdr, stops), stops
