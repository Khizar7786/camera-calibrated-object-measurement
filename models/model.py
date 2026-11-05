"""The permitted two-class Torchvision Mask R-CNN and its actual configuration.

Pretrained construction uses Torchvision's official COCO weights. Checkpoint
reconstruction can use ``pretrained=False`` without any network download; both
paths use the same FrozenBatchNorm backbone structure and two prediction heads.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d


ARCHITECTURE = "maskrcnn_resnet50_fpn"
_BACKBONE_ORDER = ("layer4", "layer3", "layer2", "layer1", "conv1")


def _integer(config: Mapping, key: str, default: int, minimum: int = 1) -> int:
    value = config.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}.")
    return value


def _fixed_threshold(config: Mapping, key: str, expected: float) -> float:
    value = config.get(key, expected)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be the fixed threshold {expected}.")
    if not math.isfinite(value) or float(value) != expected:
        raise ValueError(f"{key} must remain fixed at {expected}.")
    return expected


def _frozen_batch_norm(module: nn.Module) -> None:
    """Make offline construction match the COCO factory's FrozenBN/eps=0.

    The weights=None Torchvision path otherwise creates ordinary BatchNorm,
    whereas COCO_V1 uses FrozenBatchNorm2d and explicitly sets its epsilon to
    zero. Matching the structure avoids changed inference normalization when a
    saved fine-tuned checkpoint is reconstructed without downloading weights.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=0.0)
            frozen = frozen.to(device=child.running_mean.device, dtype=child.running_mean.dtype)
            with torch.no_grad():
                if child.affine:
                    frozen.weight.copy_(child.weight)
                    frozen.bias.copy_(child.bias)
                frozen.running_mean.copy_(child.running_mean)
                frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            _frozen_batch_norm(child)


def _freeze_backbone(model: nn.Module, trainable_layers: int) -> None:
    """Enforce last-N ResNet stages even when the factory ignores that option."""
    allowed = _BACKBONE_ORDER[:trainable_layers]
    if trainable_layers == 5:
        allowed += ("bn1",)
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(any(name == stage or name.startswith(stage + ".") for stage in allowed))
    for parameter in model.backbone.fpn.parameters():
        parameter.requires_grad_(True)


def build_model(config: dict, *, pretrained: bool = True) -> nn.Module:
    """Build only Mask R-CNN ResNet50-FPN, background plus calculator class.

    ``config`` is the flat model section. ``architecture`` is the only model-name
    selector. ImageNet backbone downloads are disabled explicitly when building
    without pretrained detection weights. Internal score/NMS cutoffs are fixed
    across training, validation and checkpoint reconstruction.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Model config must be an object.")
    if type(pretrained) is not bool:
        raise ValueError("pretrained must be a Boolean.")
    if config.get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError(f"Only {ARCHITECTURE} is supported; YOLO/Roboflow models are prohibited.")
    if type(config.get("num_classes", 2)) is not int or config.get("num_classes", 2) != 2:
        raise ValueError("num_classes must be 2: background=0 and calculator=1.")
    trainable = _integer(config, "trainable_backbone_layers", 3, minimum=0)
    if trainable > 5:
        raise ValueError("trainable_backbone_layers must lie between 0 and 5.")
    minimum = _integer(config, "min_size", 512)
    maximum = _integer(config, "max_size", 960)
    if maximum < minimum:
        raise ValueError("max_size must be >= min_size.")
    kwargs = {
        "min_size": minimum,
        "max_size": maximum,
        "rpn_pre_nms_top_n_train": _integer(config, "rpn_pre_nms_top_n_train", 1000),
        "rpn_post_nms_top_n_train": _integer(config, "rpn_post_nms_top_n_train", 500),
        "rpn_pre_nms_top_n_test": _integer(config, "rpn_pre_nms_top_n_test", 1000),
        "rpn_post_nms_top_n_test": _integer(config, "rpn_post_nms_top_n_test", 300),
        "box_batch_size_per_image": _integer(config, "box_batch_size_per_image", 256),
        "box_detections_per_img": _integer(config, "box_detections_per_img", 100),
        "box_score_thresh": _fixed_threshold(config, "box_score_thresh", 0.05),
        "box_nms_thresh": _fixed_threshold(config, "box_nms_thresh", 0.5),
    }
    for phase in ("train", "test"):
        if kwargs[f"rpn_post_nms_top_n_{phase}"] > kwargs[f"rpn_pre_nms_top_n_{phase}"]:
            raise ValueError(f"RPN post-NMS {phase} proposals cannot exceed pre-NMS proposals.")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        # Torchvision forces this to five for weights=None. Passing None avoids
        # its warning; the explicit last-N policy below remains authoritative.
        trainable_backbone_layers=trainable if pretrained else None,
        **kwargs,
    )
    if not pretrained:
        _frozen_batch_norm(model.backbone.body)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, 2)
    _freeze_backbone(model, trainable)
    model._measure_it_model_config = {
        "architecture": ARCHITECTURE,
        "num_classes": 2,
        "trainable_backbone_layers": trainable,
        "pretrained_weights": str(weights) if weights is not None else None,
    }
    return model


def _parameter_counts(module: nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    }


def model_summary(model: nn.Module) -> dict:
    """Return JSON-ready evidence from the built modules, not merely the config."""
    body = model.backbone.body
    stages = {}
    for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"):
        stages[name] = _parameter_counts(getattr(body, name))
        stages[name]["trainable"] = stages[name]["trainable_parameters"] > 0
    frozen_bn = [module for module in body.modules() if isinstance(module, FrozenBatchNorm2d)]
    metadata = getattr(model, "_measure_it_model_config", {})
    return {
        **metadata,
        **_parameter_counts(model),
        "box_classes": model.roi_heads.box_predictor.cls_score.out_features,
        "box_regression_outputs": model.roi_heads.box_predictor.bbox_pred.out_features,
        "mask_classes": model.roi_heads.mask_predictor.mask_fcn_logits.out_channels,
        "backbone_layers": stages,
        "fpn": _parameter_counts(model.backbone.fpn),
        "frozen_batch_norm_layers": len(frozen_bn),
        "frozen_batch_norm_eps": sorted({float(module.eps) for module in frozen_bn}),
        "input_transform": {
            "min_size": list(model.transform.min_size),
            "max_size": model.transform.max_size,
            "image_mean": list(model.transform.image_mean),
            "image_std": list(model.transform.image_std),
        },
        "proposal_limits": {
            "train_pre_nms": model.rpn._pre_nms_top_n["training"],
            "train_post_nms": model.rpn._post_nms_top_n["training"],
            "test_pre_nms": model.rpn._pre_nms_top_n["testing"],
            "test_post_nms": model.rpn._post_nms_top_n["testing"],
        },
        "box_batch_size_per_image": model.roi_heads.fg_bg_sampler.batch_size_per_image,
        "box_detections_per_image": model.roi_heads.detections_per_img,
        "box_score_threshold": float(model.roi_heads.score_thresh),
        "box_nms_threshold": float(model.roi_heads.nms_thresh),
    }
