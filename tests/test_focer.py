import math
import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from focer_main import (
    balanced_image_weights,
    curriculum_state,
    optimizer_for,
    parse_stops,
    run_loss_epoch,
)
from models.detr import DETR
from models.focer import (
    FactorizedRoutingDecoderLayer,
    FixedExposureRenderer,
    FixedStopQueryRouter,
    matched_routing_entropy,
    teacher_guided_routing_loss,
)
from models.transformer import Transformer, TransformerDecoderLayer
from util.misc import NestedTensor


class TinyBackbone(nn.Module):
    num_channels = 8

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.conv = nn.Conv2d(3, self.num_channels, kernel_size=3, stride=2, padding=1)
        self.hidden_dim = hidden_dim

    def forward(self, nested):
        feature = self.conv(nested.tensors)
        mask = F.interpolate(
            nested.mask[None].float(), size=feature.shape[-2:]
        ).to(torch.bool)[0]
        position = feature.new_zeros(
            feature.shape[0], self.hidden_dim, feature.shape[-2], feature.shape[-1]
        )
        return [NestedTensor(feature, mask)], [position]


def tiny_detector():
    transformer = Transformer(
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        return_intermediate_dec=True,
    )
    return DETR(
        TinyBackbone(), transformer, intermediate=False,
        num_classes=5, num_queries=6, aux_loss=False,
    )


class FixedStopTests(unittest.TestCase):
    def test_stop_parser_and_renderer_are_fixed(self):
        self.assertEqual(parse_stops("0:4:1"), [0, 1, 2, 3, 4])
        self.assertEqual(parse_stops("0,2.5,7"), [0.0, 2.5, 7.0])
        renderer = FixedExposureRenderer([0, 1, 2], short_side=8, max_side=16)
        self.assertFalse(any(name == "stops" for name, _ in renderer.named_parameters()))
        hdr = torch.full((1, 3, 8, 12), 0.05)
        first = renderer(hdr)
        second = renderer(hdr)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 3, 3, 8, 12))

    def test_reliability_discounts_dark_and_clipped_values(self):
        renderer = FixedExposureRenderer([0, 1], short_side=8, max_side=16)
        # Build normalized LDR tensors corresponding to linear values 0.001 and 0.2.
        linear = torch.tensor([0.001, 0.2]).view(1, 2, 1, 1, 1).expand(1, 2, 3, 8, 8)
        ldr = linear.pow(1 / renderer.gamma)
        normalized = (ldr - renderer.imagenet_mean.view(1, 1, 3, 1, 1)) / renderer.imagenet_std.view(1, 1, 3, 1, 1)
        quality = renderer.reliability(normalized, (2, 2))
        self.assertGreater(float(quality[0, 1].mean()), float(quality[0, 0].mean()))


class RoutingLayerTests(unittest.TestCase):
    def test_routing_is_per_query_and_normalized(self):
        torch.manual_seed(3)
        base = TransformerDecoderLayer(16, 4, dim_feedforward=32, dropout=0.0)
        layer = FactorizedRoutingDecoderLayer(base, 16, radiometric_strength=0.5)
        tgt = torch.zeros(4, 1, 16)
        query_pos = torch.randn_like(tgt)
        memory = torch.randn(5, 1, 3, 16)
        pos = torch.zeros_like(memory)
        mask = torch.zeros(1, 3, 5, dtype=torch.bool)
        stops = torch.randn(3, 16)
        reliability = torch.rand(1, 3, 5).clamp_min(0.1)
        output, weights, soft_weights, quality = layer(
            tgt, memory, mask, pos, query_pos, stops, reliability
        )
        self.assertEqual(tuple(output.shape), (4, 1, 16))
        self.assertEqual(tuple(weights.shape), (1, 4, 3))
        self.assertEqual(tuple(quality.shape), (1, 4, 3))
        self.assertEqual(tuple(soft_weights.shape), (1, 4, 3))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(1, 4), atol=1e-6))
        # Distinct object queries have distinct routing distributions.
        self.assertGreater(float((weights[:, 0] - weights[:, 1]).abs().max()), 1e-7)

    def test_radiometric_prior_affects_the_query_route(self):
        torch.manual_seed(4)
        base = TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0)
        layer = FactorizedRoutingDecoderLayer(base, 8, radiometric_strength=2.0)
        for module in (layer.route_query, layer.route_evidence, layer.route_stop):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        tgt = torch.zeros(2, 1, 8)
        query_pos = torch.randn_like(tgt)
        memory = torch.randn(3, 1, 2, 8)
        reliability = torch.tensor([[[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]]])
        _, weights, _, _ = layer(
            tgt,
            memory,
            torch.zeros(1, 2, 3, dtype=torch.bool),
            torch.zeros_like(memory),
            query_pos,
            torch.zeros(2, 8),
            reliability,
        )
        self.assertTrue(torch.all(weights[:, :, 0] > weights[:, :, 1]))

    def test_topk_budget_is_sparse_but_keeps_a_training_gradient(self):
        torch.manual_seed(6)
        base = TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0)
        layer = FactorizedRoutingDecoderLayer(
            base, 8, radiometric_strength=0.0, route_top_k=2
        )
        tgt = torch.zeros(3, 1, 8)
        query_pos = torch.randn_like(tgt)
        memory = torch.randn(4, 1, 5, 8)
        output, weights, soft_weights, _ = layer(
            tgt,
            memory,
            torch.zeros(1, 5, 4, dtype=torch.bool),
            torch.zeros_like(memory),
            query_pos,
            torch.randn(5, 8),
            torch.ones(1, 5, 4),
        )
        self.assertTrue(torch.all((weights > 0).sum(dim=-1) == 2))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(1, 3), atol=1e-6))
        self.assertTrue(torch.allclose(soft_weights.sum(-1), torch.ones(1, 3), atol=1e-6))
        output.sum().backward()
        self.assertIsNotNone(layer.route_query.weight.grad)
        self.assertTrue(torch.isfinite(layer.route_query.weight.grad).all())


class EndToEndShapeTests(unittest.TestCase):
    def test_tiny_model_has_one_prediction_set_and_fixed_routes(self):
        torch.manual_seed(5)
        model = FixedStopQueryRouter(
            tiny_detector(), [0, 2, 4], short_side=16, max_side=24,
            feature_chunk_size=2, radiometric_strength=0.5,
        )
        model.freeze_feature_extractor()
        self.assertEqual(model.assert_only_router_trainable(), sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ))
        output = model([torch.rand(3, 16, 20)])
        self.assertEqual(tuple(output["pred_logits"].shape), (1, 6, 6))
        self.assertEqual(tuple(output["pred_boxes"].shape), (1, 6, 4))
        self.assertEqual(tuple(output["routing_weights"].shape), (1, 6, 3))
        self.assertTrue(torch.equal(output["fixed_stops"], torch.tensor([0.0, 2.0, 4.0])))
        self.assertTrue(torch.allclose(
            output["routing_weights"].sum(-1), torch.ones(1, 6), atol=1e-6
        ))
        # There is one six-query output, not three per-exposure prediction sets.
        self.assertEqual(output["pred_logits"].shape[1], model.query_embed.num_embeddings)
        loss = output["pred_logits"].sum() + output["pred_boxes"].sum()
        loss.backward()
        route_gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if "route_" in name and parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in route_gradients))
        self.assertTrue(all(
            parameter.grad is None for parameter in model.backbone.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad == model.is_router_parameter(name)
            for name, parameter in model.named_parameters()
        ))
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.decoder.layers[0].multihead_attn.training)
        self.assertTrue(model.decoder.layers[0].route_query.training)

    def test_optimizer_rejects_any_unfrozen_detector_parameter(self):
        model = FixedStopQueryRouter(
            tiny_detector(), [0, 1, 2], short_side=16, max_side=24
        )
        model.freeze_pretrained_detector()
        config = SimpleNamespace(routing_lr=1e-4, weight_decay=1e-4)
        optimizer = optimizer_for(config, model)
        optimized = sum(
            parameter.numel()
            for group in optimizer.param_groups for parameter in group["params"]
        )
        self.assertEqual(optimized, model.assert_only_router_trainable())
        model.query_embed.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Pretrained detector"):
            optimizer_for(config, model)

    def test_matched_entropy_uses_only_positive_queries(self):
        outputs = {
            "pred_logits": torch.randn(1, 3, 4),
            "pred_boxes": torch.rand(1, 3, 4),
            "routing_weights": torch.tensor([[
                [0.5, 0.5], [1.0, 0.0], [0.25, 0.75]
            ]]),
        }
        targets = [{"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)}]

        class FirstQueryMatcher:
            def __call__(self, _outputs, _targets):
                return [(torch.tensor([0]), torch.tensor([0]))]

        entropy = matched_routing_entropy(outputs, targets, FirstQueryMatcher())
        self.assertAlmostEqual(float(entropy), 1.0, places=5)

    def test_teacher_loss_uses_matched_object_utility(self):
        outputs = {
            "pred_logits": torch.randn(1, 2, 4),
            "pred_boxes": torch.rand(1, 2, 4),
            "routing_soft_weights": torch.tensor([[
                [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]
            ]], requires_grad=True),
        }
        targets = [{"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)}]

        class FirstQueryMatcher:
            def __call__(self, _outputs, _targets):
                return [(torch.tensor([0]), torch.tensor([0]))]

        utilities = [torch.tensor([[4.0, 0.0, -1.0]])]
        good, stats = teacher_guided_routing_loss(
            outputs, targets, FirstQueryMatcher(), utilities
        )
        changed = dict(outputs)
        changed["routing_soft_weights"] = torch.tensor([[
            [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]
        ]], requires_grad=True)
        bad, _ = teacher_guided_routing_loss(
            changed, targets, FirstQueryMatcher(), utilities
        )
        self.assertLess(float(good), float(bad))
        self.assertEqual(stats["teacher_objects"], 1)
        good.backward()
        self.assertIsNotNone(outputs["routing_soft_weights"].grad)

    def test_curriculum_is_soft_then_middle_then_final(self):
        config = SimpleNamespace(
            epochs=10, route_temperature_start=1.5, route_temperature=0.7,
            curriculum_soft_epochs=2, curriculum_middle_epochs=3,
            curriculum_middle_top_k=5, route_top_k=3,
        )
        self.assertEqual(curriculum_state(config, 0)["top_k"], 0)
        self.assertEqual(curriculum_state(config, 2)["top_k"], 5)
        self.assertEqual(curriculum_state(config, 5)["top_k"], 3)
        self.assertGreater(
            curriculum_state(config, 0)["temperature"],
            curriculum_state(config, 9)["temperature"],
        )


class TrainingLogTests(unittest.TestCase):
    def test_rare_class_sampler_is_capped(self):
        class Dataset:
            images = [{"id": index} for index in range(6)]
            annotations = {
                0: [{"category_id": 1}] * 16,
                1: [{"category_id": 1}] * 16,
                2: [{"category_id": 3}] * 4,
                3: [{"category_id": 3}] * 4,
                4: [{"category_id": 2}],
                5: [{"category_id": 8}],
            }

        weights, summary = balanced_image_weights(Dataset(), 3.0)
        self.assertEqual(float(weights.max()), 3.0)
        self.assertGreater(float(weights[4]), float(weights[0]))
        self.assertGreater(float(weights[5]), float(weights[2]))
        self.assertEqual(summary["category_counts"][2], 1)

    def test_detailed_epoch_summary_is_finite_and_printed(self):
        class TinyLoggingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = nn.Parameter(torch.tensor(0.25))

            def forward(self, images):
                batch = len(images)
                logits = self.value.expand(batch, 2, 4)
                boxes = self.value.sigmoid().expand(batch, 2, 4)
                routes = torch.softmax(torch.stack((
                    self.value, self.value + 0.5, self.value - 0.5
                )), dim=0).expand(batch, 2, 3)
                return {
                    "pred_logits": logits,
                    "pred_boxes": boxes,
                    "routing_weights": routes,
                    "routing_soft_weights": routes,
                }

        class FirstQueryMatcher:
            def __call__(self, outputs, targets):
                return [(torch.tensor([0]), torch.tensor([0]))
                        for _ in range(outputs["pred_logits"].shape[0])]

        class TinyCriterion(nn.Module):
            weight_dict = {"loss_ce": 1.0, "loss_bbox": 2.0, "loss_giou": 1.0}

            def __init__(self):
                super().__init__()
                self.matcher = FirstQueryMatcher()

            def forward(self, outputs, targets):
                return {
                    "loss_ce": outputs["pred_logits"].square().mean(),
                    "loss_bbox": outputs["pred_boxes"].mean(),
                    "loss_giou": (1.0 - outputs["pred_boxes"]).mean(),
                    "class_error": outputs["pred_logits"].new_tensor(25.0),
                }

        model = TinyLoggingModel()
        criterion = TinyCriterion()
        target = {
            "labels": torch.tensor([1]), "boxes": torch.rand(1, 4),
            "image_id": torch.tensor(0),
        }
        loader = [([torch.rand(3, 8, 8)], [target]) for _ in range(2)]
        config = SimpleNamespace(
            amp=False, routing_entropy_weight=0.0, clip_max_norm=0.1,
            log_every=1, epochs=1, stops=[0.0, 1.0, 2.0],
            teacher_routing_weight=1.0, teacher_target_temperature=1.0,
            teacher_min_confidence=0.0,
        )
        teacher_cache = {"entries": {"0": {
            "labels": target["labels"].clone(),
            "boxes": target["boxes"].clone(),
            "utilities": torch.tensor([[2.0, 0.0, -1.0]]),
        }}}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            summary = run_loss_epoch(
                config, model, criterion, loader, torch.device("cpu"),
                teacher_cache=teacher_cache, optimizer=optimizer, epoch=0,
            )
        self.assertEqual(summary["images"], 2)
        self.assertTrue(math.isfinite(summary["loss"]))
        self.assertTrue(math.isfinite(summary["teacher_routing_loss"]))
        self.assertEqual(len(summary["branch_marginal"]), 3)
        logged = stream.getvalue()
        self.assertIn("top stop:weight", logged)
        self.assertIn("gpu_alloc/peak=n/a", logged)
        self.assertIn("epoch summary", logged)


if __name__ == "__main__":
    unittest.main()
