"""Factorized Object-Conditioned Exposure Routing (FOCER-v2).

FOCER keeps the HDRDet fixed exposure renderings, but removes per-exposure
detector heads and late NMS fusion.  Every DETR object query first extracts an
evidence vector from *each* exposure independently.  It then assigns its own
routing distribution over those evidence vectors.  No globally fused feature
map is constructed.

V2 keeps every pretrained DETR parameter immutable.  Only the stop encoder and
object-conditioned routing projections are trainable.  It also exposes the
dense routing probabilities used by the teacher-guided routing objective while
retaining sparse top-k weights in the actual forward path.

The implementation intentionally supports one HDR scene per batch.  Its
exposure branches are processed as a parallel/shared-weight batch.
"""

import copy
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import nested_tensor_from_tensor_list


def _fixed_stop_features(stops):
    """Continuous Fourier features for numeric exposure stops."""
    if stops.ndim != 1 or stops.numel() < 2:
        raise ValueError("stops must be a one-dimensional tensor with at least two values")
    span = (stops[-1] - stops[0]).clamp_min(1e-6)
    normalized = (stops - stops[0]) / span
    pieces = [normalized, normalized.square()]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        angle = math.pi * frequency * normalized
        pieces.extend((torch.sin(angle), torch.cos(angle)))
    return torch.stack(pieces, dim=-1)


class FixedExposureRenderer(nn.Module):
    """Differentiably render an immutable list of exposure stops."""

    def __init__(self, stops, short_side=800, max_side=1333, gamma=2.2):
        super().__init__()
        stops = torch.as_tensor(stops, dtype=torch.float32)
        if stops.ndim != 1 or stops.numel() < 2:
            raise ValueError("At least two fixed stops are required")
        if not torch.isfinite(stops).all() or not torch.all(stops[1:] > stops[:-1]):
            raise ValueError("Fixed stops must be finite and strictly increasing")
        if short_side <= 0 or max_side < short_side or gamma <= 0:
            raise ValueError("Invalid render size or gamma")
        self.register_buffer("stops", stops)
        self.short_side = int(short_side)
        self.max_side = int(max_side)
        self.gamma = float(gamma)
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def output_size(self, height, width):
        size = self.short_side
        if max(height, width) / min(height, width) * size > self.max_side:
            size = int(round(self.max_side * min(height, width) / max(height, width)))
        if width < height:
            return int(size * height / width), size
        return size, int(size * width / height)

    def forward(self, hdr):
        if hdr.ndim != 4 or hdr.shape[1] != 3:
            raise ValueError("Expected HDR tensor [batch, 3, height, width]")
        if not torch.isfinite(hdr).all():
            raise ValueError("HDR input contains non-finite values")
        height, width = hdr.shape[-2:]
        new_size = self.output_size(height, width)
        rendered = []
        for stop in self.stops:
            linear = (hdr * torch.exp2(stop)).clamp(0, 1)
            ldr = torch.where(
                linear > 0,
                linear.clamp_min(1e-8).pow(1.0 / self.gamma),
                torch.zeros_like(linear),
            )
            ldr = F.interpolate(ldr, size=new_size, mode="bilinear", align_corners=False)
            rendered.append((ldr - self.imagenet_mean) / self.imagenet_std)
        # [B, K, 3, H, W], with no learned stop-generation path.
        return torch.stack(rendered, dim=1)

    def reliability(self, normalized_ldr, feature_size, dark=0.02, bright=0.98,
                    dark_softness=0.01, bright_softness=0.02):
        """Return a soft radiometric reliability map for every exposure.

        Reliability is high where the *linear display exposure* is above the
        nominal noise floor and below clipping.  It is a prior, not a label.
        """
        if normalized_ldr.ndim != 5:
            raise ValueError("Expected rendered images [B, K, 3, H, W]")
        batch, branches = normalized_ldr.shape[:2]
        flat = normalized_ldr.reshape(-1, 3, *normalized_ldr.shape[-2:])
        ldr = (flat * self.imagenet_std + self.imagenet_mean).clamp(0, 1)
        linear = ldr.pow(self.gamma)
        luminance = (
            0.2126 * linear[:, 0:1]
            + 0.7152 * linear[:, 1:2]
            + 0.0722 * linear[:, 2:3]
        )
        channel_peak = linear.amax(dim=1, keepdim=True)
        above_noise = torch.sigmoid((luminance - dark) / dark_softness)
        below_clip = torch.sigmoid((bright - channel_peak) / bright_softness)
        quality = above_noise * below_clip
        quality = F.interpolate(quality, size=feature_size, mode="area")
        return quality.reshape(batch, branches, -1)


class FactorizedRoutingDecoderLayer(nn.Module):
    """DETR decoder layer with exposure-separated evidence then query routing."""

    def __init__(self, detr_layer, d_model, route_temperature=1.0,
                 radiometric_strength=0.5, route_top_k=3):
        super().__init__()
        if getattr(detr_layer, "normalize_before", False):
            raise NotImplementedError("FOCER-v2 currently supports post-norm DETR checkpoints")
        if route_temperature <= 0 or radiometric_strength < 0 or route_top_k < 0:
            raise ValueError("Invalid routing temperature or radiometric strength")

        # Preserve the pretrained DETR decoder components and their semantics.
        self.self_attn = copy.deepcopy(detr_layer.self_attn)
        self.multihead_attn = copy.deepcopy(detr_layer.multihead_attn)
        self.linear1 = copy.deepcopy(detr_layer.linear1)
        self.linear2 = copy.deepcopy(detr_layer.linear2)
        self.norm1 = copy.deepcopy(detr_layer.norm1)
        self.norm2 = copy.deepcopy(detr_layer.norm2)
        self.norm3 = copy.deepcopy(detr_layer.norm3)
        self.dropout = copy.deepcopy(detr_layer.dropout)
        self.dropout1 = copy.deepcopy(detr_layer.dropout1)
        self.dropout2 = copy.deepcopy(detr_layer.dropout2)
        self.dropout3 = copy.deepcopy(detr_layer.dropout3)
        self.activation = detr_layer.activation

        # These projections are the new object-conditioned router.
        self.route_query = nn.Linear(d_model, d_model)
        self.route_evidence = nn.Linear(d_model, d_model)
        self.route_stop = nn.Linear(d_model, d_model, bias=False)
        self.route_temperature = float(route_temperature)
        self.radiometric_strength = float(radiometric_strength)
        self.route_top_k = int(route_top_k)

    def set_routing_state(self, route_top_k, route_temperature):
        """Change only the routing budget/temperature, never model weights."""
        if route_top_k < 0 or route_temperature <= 0:
            raise ValueError("Invalid routing curriculum state")
        self.route_top_k = int(route_top_k)
        self.route_temperature = float(route_temperature)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_key_padding_mask, pos, query_pos,
                stop_embedding, reliability):
        """
        Args:
            tgt: [Q, B, C]
            memory/pos: [L, B, K, C]
            memory_key_padding_mask: [B, K, L]
            stop_embedding: [K, C]
            reliability: [B, K, L]
        """
        queries, batch, channels = tgt.shape
        locations, memory_batch, branches, memory_channels = memory.shape
        if memory_batch != batch or memory_channels != channels:
            raise ValueError("Query and exposure memory dimensions do not agree")
        if stop_embedding.shape != (branches, channels):
            raise ValueError("Stop embedding dimensions do not agree with exposure memory")

        # Standard DETR query self-attention.
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Evidence stage: each query interrogates each exposure separately.
        # Branches are folded into the batch dimension, so spatial attention
        # never mixes exposure feature maps.
        query = self.with_pos_embed(tgt, query_pos)
        query = query.unsqueeze(2).expand(queries, batch, branches, channels)
        query = query.reshape(queries, batch * branches, channels)
        exposure_pos = pos + stop_embedding.view(1, 1, branches, channels)
        key = (memory + exposure_pos).reshape(locations, batch * branches, channels)
        value = memory.reshape(locations, batch * branches, channels)
        padding = memory_key_padding_mask.reshape(batch * branches, locations)
        evidence, spatial_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=padding,
            need_weights=True,
        )
        evidence = evidence.reshape(queries, batch, branches, channels)

        # Query-specific reliability: evaluate clipping/noise only in the
        # spatial region that this query attended to in each exposure.
        spatial_weights = spatial_weights.reshape(batch, branches, queries, locations)
        weighted_quality = (spatial_weights * reliability[:, :, None, :]).sum(dim=-1)
        weight_mass = spatial_weights.sum(dim=-1).clamp_min(1e-6)
        weighted_quality = (weighted_quality / weight_mass).permute(2, 0, 1)

        # Routing stage: compare every query with the evidence it retrieved
        # independently from every exposure.  This is the only cross-exposure
        # aggregation and happens inside each object query.
        route_q = self.route_query(self.with_pos_embed(tgt, query_pos)).unsqueeze(2)
        route_k = self.route_evidence(evidence)
        route_k = route_k + self.route_stop(stop_embedding).view(1, 1, branches, channels)
        route_logits = (route_q * route_k).sum(dim=-1) / math.sqrt(channels)
        if self.radiometric_strength:
            route_logits = route_logits + self.radiometric_strength * torch.log(
                weighted_quality.clamp_min(1e-4)
            )
        soft_weights = torch.softmax(route_logits / self.route_temperature, dim=-1)
        if self.route_top_k and self.route_top_k < branches:
            # The forward pass has a real exposure budget: every query can use
            # at most top-k branches.  The straight-through expression has the
            # same sparse forward value but uses the dense softmax derivative,
            # so an excluded branch can still learn to enter the route later.
            values, indices = soft_weights.topk(self.route_top_k, dim=-1)
            hard_weights = torch.zeros_like(soft_weights).scatter(-1, indices, values)
            hard_weights = hard_weights / hard_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            route_weights = hard_weights.detach() - soft_weights.detach() + soft_weights
        else:
            route_weights = soft_weights
        routed_evidence = (route_weights.unsqueeze(-1) * evidence).sum(dim=2)

        tgt = self.norm2(tgt + self.dropout2(routed_evidence))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return (
            tgt,
            route_weights.permute(1, 0, 2),
            soft_weights.permute(1, 0, 2),
            weighted_quality.permute(1, 0, 2),
        )


class FactorizedRoutingDecoder(nn.Module):
    def __init__(self, detr_decoder, d_model, stop_feature_dim=10,
                 route_temperature=1.0, radiometric_strength=0.5,
                 route_top_k=3):
        super().__init__()
        self.layers = nn.ModuleList([
            FactorizedRoutingDecoderLayer(
                layer,
                d_model=d_model,
                route_temperature=route_temperature,
                radiometric_strength=radiometric_strength,
                route_top_k=route_top_k,
            )
            for layer in detr_decoder.layers
        ])
        self.norm = copy.deepcopy(detr_decoder.norm)
        self.stop_encoder = nn.Sequential(
            nn.Linear(stop_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def set_routing_state(self, route_top_k, route_temperature):
        for layer in self.layers:
            layer.set_routing_state(route_top_k, route_temperature)

    def forward(self, query_embed, memory, memory_key_padding_mask, pos,
                stop_features, reliability):
        queries, batch, channels = query_embed.shape
        output = torch.zeros_like(query_embed)
        stop_embedding = self.stop_encoder(stop_features)
        intermediate = []
        routing = []
        soft_routing = []
        routing_quality = []
        for layer in self.layers:
            output, weights, soft_weights, quality = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_embed,
                stop_embedding=stop_embedding,
                reliability=reliability,
            )
            intermediate.append(self.norm(output) if self.norm is not None else output)
            routing.append(weights)
            soft_routing.append(soft_weights)
            routing_quality.append(quality)
        # DETR convention: [decoder layers, batch, queries, channels].
        hidden = torch.stack(intermediate).transpose(1, 2)
        return (
            hidden,
            torch.stack(routing),
            torch.stack(soft_routing),
            torch.stack(routing_quality),
        )


class FixedStopQueryRouter(nn.Module):
    """One DETR prediction set with a strictly frozen pretrained detector."""

    def __init__(self, detector, stops, short_side=800, max_side=1333,
                 feature_chunk_size=0, route_temperature=1.0,
                 radiometric_strength=0.5, route_top_k=3):
        super().__init__()
        if getattr(detector, "intermediate", False):
            raise ValueError("Build the base DETR with intermediate=False")
        self.renderer = FixedExposureRenderer(
            stops=stops, short_side=short_side, max_side=max_side
        )
        self.backbone = detector.backbone
        self.input_proj = detector.input_proj
        self.encoder = detector.transformer.encoder
        self.query_embed = detector.query_embed
        self.class_embed = detector.class_embed
        self.bbox_embed = detector.bbox_embed
        self.aux_loss = detector.aux_loss
        self.hidden_dim = detector.transformer.d_model
        self.decoder = FactorizedRoutingDecoder(
            detector.transformer.decoder,
            d_model=self.hidden_dim,
            route_temperature=route_temperature,
            radiometric_strength=radiometric_strength,
            route_top_k=route_top_k,
        )
        self.feature_chunk_size = int(feature_chunk_size)
        self.pretrained_detector_frozen = False

    @property
    def branches(self):
        return int(self.renderer.stops.numel())

    @staticmethod
    def is_router_parameter(name):
        return (".route_query." in name or ".route_evidence." in name
                or ".route_stop." in name or name.startswith("decoder.stop_encoder."))

    def freeze_pretrained_detector(self):
        """Freeze all official DETR weights and re-enable only new V2 routing weights."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if self.is_router_parameter(name):
                parameter.requires_grad_(True)
        self.pretrained_detector_frozen = True
        self.assert_only_router_trainable()

    def freeze_feature_extractor(self):
        """Compatibility alias that now enforces the stricter V2 contract."""
        self.freeze_pretrained_detector()

    def assert_only_router_trainable(self):
        unexpected = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and not self.is_router_parameter(name)
        ]
        missing = [
            name for name, parameter in self.named_parameters()
            if self.is_router_parameter(name) and not parameter.requires_grad
        ]
        if unexpected or missing:
            raise RuntimeError(
                "Strict detector freeze violated; unexpected={} missing={}".format(
                    unexpected, missing
                )
            )
        return sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )

    def set_routing_state(self, route_top_k, route_temperature):
        if route_top_k > self.branches:
            raise ValueError("Routing budget exceeds fixed exposure count")
        self.decoder.set_routing_state(route_top_k, route_temperature)

    def train(self, mode=True):
        super().train(mode)
        if self.pretrained_detector_frozen:
            # Frozen dropout must remain disabled.  Router projections and the
            # stop encoder still follow the requested train/eval mode.
            for module in (self.backbone, self.input_proj, self.encoder,
                           self.query_embed, self.class_embed, self.bbox_embed,
                           self.decoder.norm):
                if module is not None:
                    module.eval()
            for layer in self.decoder.layers:
                for module in (
                    layer.self_attn, layer.multihead_attn,
                    layer.linear1, layer.linear2,
                    layer.norm1, layer.norm2, layer.norm3,
                    layer.dropout, layer.dropout1, layer.dropout2, layer.dropout3,
                ):
                    module.eval()
                layer.route_query.train(mode)
                layer.route_evidence.train(mode)
                layer.route_stop.train(mode)
            self.decoder.stop_encoder.train(mode)
        return self

    def _encode_chunk(self, images):
        samples = nested_tensor_from_tensor_list(list(images))
        features, positions = self.backbone(samples)
        src, mask = features[-1].decompose()
        if mask is None:
            raise RuntimeError("Backbone did not return a padding mask")
        src = self.input_proj(src)
        batch, channels, height, width = src.shape
        src_sequence = src.flatten(2).permute(2, 0, 1)
        pos_sequence = positions[-1].flatten(2).permute(2, 0, 1)
        flat_mask = mask.flatten(1)
        memory = self.encoder(
            src_sequence,
            src_key_padding_mask=flat_mask,
            pos=pos_sequence,
        )
        return memory, pos_sequence, flat_mask, (height, width)

    def _encode_exposures(self, rendered):
        batch, branches = rendered.shape[:2]
        flat = rendered.reshape(batch * branches, *rendered.shape[2:])
        chunk_size = self.feature_chunk_size or flat.shape[0]
        memories, positions, masks = [], [], []
        feature_size = None
        context = torch.no_grad if self.pretrained_detector_frozen else nullcontext
        for start in range(0, flat.shape[0], chunk_size):
            with context():
                memory, pos, mask, current_size = self._encode_chunk(
                    flat[start:start + chunk_size]
                )
            if feature_size is not None and feature_size != current_size:
                raise RuntimeError("Exposure chunks produced inconsistent feature sizes")
            feature_size = current_size
            memories.append(memory)
            positions.append(pos)
            masks.append(mask)
        memory = torch.cat(memories, dim=1)
        pos = torch.cat(positions, dim=1)
        mask = torch.cat(masks, dim=0)
        locations, _, channels = memory.shape
        memory = memory.reshape(locations, batch, branches, channels)
        pos = pos.reshape(locations, batch, branches, channels)
        mask = mask.reshape(batch, branches, locations)
        return memory, pos, mask, feature_size

    def forward(self, hdr_images):
        if isinstance(hdr_images, torch.Tensor):
            hdr = hdr_images
        else:
            if len(hdr_images) != 1:
                raise ValueError("FOCER-v2 currently requires one HDR scene per batch")
            hdr = torch.stack(hdr_images)
        if hdr.shape[0] != 1:
            raise ValueError("FOCER-v2 currently requires one HDR scene per batch")

        rendered = self.renderer(hdr)
        memory, pos, mask, feature_size = self._encode_exposures(rendered)
        reliability = self.renderer.reliability(rendered, feature_size)
        stop_features = _fixed_stop_features(self.renderer.stops)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, hdr.shape[0], 1)
        hidden, routing_history, soft_routing_history, quality_history = self.decoder(
            query_embed,
            memory,
            memory_key_padding_mask=mask,
            pos=pos,
            stop_features=stop_features,
            reliability=reliability,
        )
        outputs_class = self.class_embed(hidden)
        outputs_coord = self.bbox_embed(hidden).sigmoid()
        result = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "routing_weights": routing_history[-1],
            "routing_history": routing_history,
            "routing_soft_weights": soft_routing_history[-1],
            "routing_soft_history": soft_routing_history,
            "routing_quality": quality_history[-1],
            "fixed_stops": self.renderer.stops,
        }
        if self.aux_loss:
            result["aux_outputs"] = [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return result


def matched_routing_entropy(outputs, targets, matcher):
    """Normalized routing entropy for ground-truth-matched object queries."""
    matching_input = {
        "pred_logits": outputs["pred_logits"],
        "pred_boxes": outputs["pred_boxes"],
    }
    indices = matcher(matching_input, targets)
    weights = outputs["routing_weights"]
    entropies = []
    for batch_index, (query_indices, _) in enumerate(indices):
        if query_indices.numel():
            selected = weights[batch_index, query_indices].clamp_min(1e-8)
            entropy = -(selected * selected.log()).sum(dim=-1)
            entropy = entropy / math.log(selected.shape[-1])
            entropies.append(entropy)
    if not entropies:
        return weights.sum() * 0
    return torch.cat(entropies).mean()


def teacher_guided_routing_loss(outputs, targets, matcher, teacher_utilities,
                                temperature=1.0, minimum_confidence=0.05):
    """Teach matched object queries which fixed exposures help that object.

    ``teacher_utilities`` contains one ``[objects, branches]`` tensor per
    image.  Utilities are produced offline by the frozen official DETR.  A
    confidence derived from the target distribution's entropy prevents flat,
    uninformative teacher targets from dominating training.
    """
    if temperature <= 0 or not 0 <= minimum_confidence < 1:
        raise ValueError("Invalid teacher-routing loss settings")
    matching_input = {
        "pred_logits": outputs["pred_logits"],
        "pred_boxes": outputs["pred_boxes"],
    }
    indices = matcher(matching_input, targets)
    predictions = outputs["routing_soft_weights"].clamp_min(1e-8)
    weighted_losses = []
    confidence_values = []
    agreements = []
    used_objects = 0
    for batch_index, (query_indices, target_indices) in enumerate(indices):
        utilities = teacher_utilities[batch_index].to(
            device=predictions.device, dtype=predictions.dtype
        )
        expected_shape = (targets[batch_index]["labels"].numel(), predictions.shape[-1])
        if tuple(utilities.shape) != expected_shape:
            raise ValueError(
                "Teacher utility shape {} does not match {}".format(
                    tuple(utilities.shape), expected_shape
                )
            )
        if not query_indices.numel():
            continue
        target_distribution = torch.softmax(
            utilities[target_indices] / temperature, dim=-1
        )
        target_entropy = -(
            target_distribution * target_distribution.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(predictions.shape[-1])
        confidence = (1.0 - target_entropy).clamp(0, 1)
        informative = confidence >= minimum_confidence
        if not informative.any():
            continue
        predicted = predictions[batch_index, query_indices][informative]
        target_distribution = target_distribution[informative]
        confidence = confidence[informative]
        cross_entropy = -(
            target_distribution * predicted.log()
        ).sum(dim=-1)
        weighted_losses.append((cross_entropy * confidence).sum())
        confidence_values.append(confidence.detach())
        agreements.append(
            (predicted.argmax(dim=-1) == target_distribution.argmax(dim=-1)).float().detach()
        )
        used_objects += int(informative.sum())
    if not weighted_losses:
        zero = predictions.sum() * 0
        return zero, {
            "teacher_objects": 0,
            "teacher_confidence": 0.0,
            "teacher_top1_agreement": 0.0,
        }
    confidence_total = torch.cat(confidence_values).sum().clamp_min(1e-8)
    loss = torch.stack(weighted_losses).sum() / confidence_total
    return loss, {
        "teacher_objects": used_objects,
        "teacher_confidence": float(torch.cat(confidence_values).mean().cpu()),
        "teacher_top1_agreement": float(torch.cat(agreements).mean().cpu()),
    }
