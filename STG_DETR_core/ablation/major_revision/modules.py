"""Major-revision RPSB, STGF, and DAGS implementations.

The legacy implementations under :mod:`ultralytics` are deliberately left
untouched.  The registration shim in ``bootstrap.py`` exposes these classes to
the existing YAML parser only for the new experiment entry points.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.extra_modules.block import Partial_conv3
from ultralytics.nn.modules.block import BasicBlock, get_activation
from ultralytics.nn.modules.conv import Conv


_ACTIVE_GGAS: dict[str, Any] = {}


def set_active_config(config: Mapping[str, Any] | None) -> None:
    """Set per-model DAGS options consumed by :class:`DAGSLoss` construction."""

    global _ACTIVE_GGAS
    _ACTIVE_GGAS = deepcopy(dict((config or {}).get("ggas", {})))


class PartialConvNormAct(nn.Module):
    """PConv followed by BN and an optional activation.

    Exactly ``C // r`` channels participate in the spatial 3x3 convolution;
    the remaining channels are concatenated without spatial filtering.
    """

    def __init__(self, channels: int, n_div: int, act: str | None) -> None:
        super().__init__()
        if n_div < 1 or channels // n_div < 1:
            raise ValueError(f"Invalid PConv split: channels={channels}, r={n_div}.")
        self.channels = int(channels)
        self.n_div = int(n_div)
        self.spatial_channels = self.channels // self.n_div
        self.pconv = Partial_conv3(self.channels, n_div=self.n_div, forward="split_cat")
        self.norm = nn.BatchNorm2d(self.channels)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pconv(x)))


class _RPSBBase(BasicBlock):
    """Fair RPSB base that leaves every stage-entry block standard."""

    expansion = 1
    first_partial = False
    second_partial = False
    n_div = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        # Construct the same standard block first.  Replacement modules are
        # built in a forked RNG context so later detector layers receive the
        # same initialization stream in every R0--R5 configuration.
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.major_revision_applied = bool(stride == 1 and ch_in == ch_out and shortcut)
        self.rpsb_first_partial = False
        self.rpsb_second_partial = False
        self.rpsb_n_div = int(self.n_div)
        if not self.major_revision_applied:
            return
        with torch.random.fork_rng(devices=[]):
            if self.first_partial:
                self.branch2a = PartialConvNormAct(ch_out, self.n_div, act=act)
                self.rpsb_first_partial = True
            if self.second_partial:
                # The standard second convolution has no pre-addition
                # activation.  Keeping act=None avoids the confound in the
                # legacy BasicBlock_PConv implementation.
                self.branch2b = PartialConvNormAct(ch_out, self.n_div, act=None)
                self.rpsb_second_partial = True


class BasicBlock_R0(_RPSBBase):
    """R0: Full 3x3 + Full 3x3."""


class BasicBlock_R1(_RPSBBase):
    """R1: PConv(r=4) + PConv(r=4), interior residual blocks only."""

    first_partial = True
    second_partial = True
    n_div = 4


class BasicBlock_R2(_RPSBBase):
    """R2: PConv(r=4) + Full 3x3, interior residual blocks only."""

    first_partial = True
    second_partial = False
    n_div = 4


class BasicBlock_R3(_RPSBBase):
    """R3: Full 3x3 + PConv(r=4), interior residual blocks only."""

    first_partial = False
    second_partial = True
    n_div = 4


class BasicBlock_R4(_RPSBBase):
    """R4: Full 3x3 + PConv(r=2), interior residual blocks only."""

    first_partial = False
    second_partial = True
    n_div = 2


class BasicBlock_R5(_RPSBBase):
    """R5: Full 3x3 + PConv(r=8), interior residual blocks only."""

    first_partial = False
    second_partial = True
    n_div = 8


class ECAGate(nn.Module):
    """Local channel interaction used by STGF; this is not claimed as novel."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel size must be a positive odd integer.")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.pool(x).squeeze(-1).transpose(-1, -2)
        return self.conv(weight).transpose(-1, -2).unsqueeze(-1).sigmoid()


class FullConv3(nn.Module):
    """Full-channel local 3x3 mixer matched to PConv's no-BN/no-act form."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MajorRevisionSTGF(nn.Module):
    """Controlled STGF variants sharing projections, width, and local modeling.

    ``mode`` is one of ``add``, ``concat``, ``spatial``, ``channel``, or
    ``full``.  ``local_conv`` is ``partial`` or ``full``.  All variants retain
    independent shallow/deep/value local mixers and the same residual output
    projection.  S6--S8 vary only ``gamma_init``.
    """

    VALID_MODES = {"add", "concat", "spatial", "channel", "full"}
    VALID_LOCAL = {"partial", "full"}

    def __init__(
        self,
        channels,
        hidden_channels: int = 64,
        n_div: int = 4,
        eca_kernel: int = 3,
        mode: str = "full",
        local_conv: str = "partial",
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"STGF expects [P3, P4] channels, got {channels}.")
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown STGF mode {mode!r}; expected {sorted(self.VALID_MODES)}.")
        if local_conv not in self.VALID_LOCAL:
            raise ValueError(f"Unknown local_conv {local_conv!r}; expected {sorted(self.VALID_LOCAL)}.")
        if hidden_channels < n_div:
            raise ValueError("hidden_channels must be no smaller than n_div.")

        shallow_channels, deep_channels = map(int, channels)
        self.mode = mode
        self.local_conv = local_conv
        self.hidden_channels = int(hidden_channels)
        self.n_div = int(n_div)

        # Do not perturb the initialization stream of the shared detector.
        # Shared projections are built before variant-specific components so
        # they receive the same initialization in S1--S8 (including S5).
        with torch.random.fork_rng(devices=[]):
            self.shallow_proj = Conv(shallow_channels, hidden_channels, 1, act=nn.ReLU())
            self.deep_proj = Conv(deep_channels, hidden_channels, 1, act=nn.ReLU())
            self.out_proj = Conv(hidden_channels, shallow_channels, 1, act=False)
            channel_gate = ECAGate(eca_kernel)
            mixer = (lambda: Partial_conv3(hidden_channels, n_div=n_div, forward="split_cat")) \
                if local_conv == "partial" else (lambda: FullConv3(hidden_channels))
            self.texture_mixer = mixer()
            key_mixer = mixer()
            self.value_mixer = mixer()
            # S3 deletes the channel gate and S4 deletes the spatial/key
            # branch structurally.  The candidates above are nevertheless
            # instantiated in fixed order, preserving common initialization
            # for the modules that remain.
            self.key_mixer = key_mixer if mode in {"spatial", "full"} else None
            self.channel_gate = channel_gate if mode in {"channel", "full"} else None
            self.concat_proj = (
                nn.Conv2d(2 * hidden_channels, hidden_channels, 1, bias=False)
                if mode == "concat"
                else None
            )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    def forward(self, inputs):
        p3, p4 = inputs
        texture = self.texture_mixer(self.shallow_proj(p3))
        context = self.deep_proj(p4)
        context = F.interpolate(context, size=texture.shape[-2:], mode="nearest")
        # K and V are parallel local projections of the same resized deep
        # context, matching the audited legacy STGF equation.
        value = self.value_mixer(context)

        if self.mode == "add":
            compensation = texture + value
        elif self.mode == "concat":
            compensation = self.concat_proj(torch.cat((texture, value), dim=1))
        elif self.mode == "spatial":
            key = self.key_mixer(context)
            spatial_gate = (texture * key).mean(dim=1, keepdim=True).sigmoid()
            compensation = value * spatial_gate
        elif self.mode == "channel":
            compensation = value * self.channel_gate(texture + value)
        else:  # full
            key = self.key_mixer(context)
            spatial_gate = (texture * key).mean(dim=1, keepdim=True).sigmoid()
            compensation = value * spatial_gate
            compensation = compensation * self.channel_gate(texture + compensation)
        return p3 + self.gamma.to(dtype=p3.dtype) * self.out_proj(compensation)


class MajorRevisionGGASHead(nn.Module):
    """Disposable, training-only activation head shared by D1--D4/H0--H10."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, grad_scale: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= float(grad_scale) <= 1.0:
            raise ValueError("GGASHead grad_scale must be in [0, 1].")
        self.grad_scale = float(grad_scale)
        with torch.random.fork_rng(devices=[]):
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
            self.bn = nn.BatchNorm2d(in_channels)
            self.pointwise = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
            self.act = nn.SiLU(inplace=True)
            self.out = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The deployment graph takes the identity path.  ``strip_dags_heads``
        # can physically replace this module before export/checkpoint release.
        if not self.training:
            return x
        if self.grad_scale != 1.0:
            detached = x.detach()
            x = detached + self.grad_scale * (x - detached)
        return self.out(self.act(self.pointwise(self.act(self.bn(self.depthwise(x))))))


class DAGSLoss(nn.Module):
    """FP32 Gaussian supervision with MAE, balanced, or occupancy weighting."""

    VALID_MODES = {"mae", "balanced", "occupancy"}

    def __init__(
        self,
        small_area: float = 1024.0,
        sigma_ratio: float = 0.5,
        loss_weight: float = 0.25,
        truncate_sigma: float = 3.0,
        background_weight: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        cfg = _ACTIVE_GGAS
        self.tau = float(cfg.get("tau", cfg.get("small_area", small_area)))
        self.eta = float(cfg.get("eta", cfg.get("sigma_ratio", sigma_ratio)))
        self.loss_weight = float(cfg.get("lambda_g", cfg.get("loss_weight", loss_weight)))
        self.kappa = float(cfg.get("kappa", cfg.get("truncate_sigma", truncate_sigma)))
        self.sigma_min = float(cfg.get("sigma_min", 1.0))
        self.eps = float(cfg.get("eps", eps))
        self.loss_mode = str(cfg.get("loss_mode", "occupancy"))
        self.target_merge = str(cfg.get("target_merge", "sum_clip"))
        if self.loss_mode not in self.VALID_MODES:
            raise ValueError(f"Unknown DAGS loss_mode {self.loss_mode!r}.")
        if self.target_merge != "sum_clip":
            raise ValueError("Major-revision DAGS requires target_merge='sum_clip'.")
        if min(self.tau, self.eta, self.loss_weight, self.kappa, self.sigma_min, self.eps) <= 0:
            raise ValueError("DAGS tau/eta/lambda_g/kappa/sigma_min/eps must be positive.")
        # Retained only to make accidental reliance on the legacy parameter
        # visible during inspection; it is not part of the revision formula.
        self.legacy_background_weight = float(background_weight)
        self.last_rho: torch.Tensor | None = None

    @torch.no_grad()
    def build_targets(self, bboxes, batch_idx, batch_size, feature_size, image_size, device):
        """Build a summed-and-clipped Gaussian target directly on the P3 grid."""

        feature_h, feature_w = map(int, feature_size)
        image_h, image_w = map(int, image_size)
        targets = torch.zeros((batch_size, 1, feature_h, feature_w), device=device, dtype=torch.float32)
        if bboxes.numel() == 0:
            return targets

        boxes = bboxes.detach().to(device=device, dtype=torch.float32)
        indices = batch_idx.detach().to(device=device, dtype=torch.long).view(-1)
        for box, image_index_tensor in zip(boxes, indices):
            cx, cy, width, height = box
            # Boxes are normalized after the training resize/augmentation path.
            if float(width * image_w * height * image_h) >= self.tau:
                continue
            image_index = int(image_index_tensor.item())
            center_x = float((cx * feature_w).clamp(0, max(feature_w - 1, 0)))
            center_y = float((cy * feature_h).clamp(0, max(feature_h - 1, 0)))
            sigma_x = max(float(width * feature_w * self.eta), self.sigma_min)
            sigma_y = max(float(height * feature_h * self.eta), self.sigma_min)
            radius_x, radius_y = self.kappa * sigma_x, self.kappa * sigma_y
            x0 = max(int(math.floor(center_x - radius_x)), 0)
            x1 = min(int(math.ceil(center_x + radius_x)) + 1, feature_w)
            y0 = max(int(math.floor(center_y - radius_y)), 0)
            y1 = min(int(math.ceil(center_y + radius_y)) + 1, feature_h)
            if x0 >= x1 or y0 >= y1:
                continue
            yy = torch.arange(y0, y1, device=device, dtype=torch.float32)
            xx = torch.arange(x0, x1, device=device, dtype=torch.float32)
            try:
                grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            except TypeError:  # PyTorch < 1.10
                grid_y, grid_x = torch.meshgrid(yy, xx)
            gaussian = torch.exp(
                -0.5 * (((grid_x - center_x) / sigma_x) ** 2 + ((grid_y - center_y) / sigma_y) ** 2)
            )
            current = targets[image_index, 0, y0:y1, x0:x1]
            targets[image_index, 0, y0:y1, x0:x1] = (current + gaussian).clamp_(0.0, 1.0)
        return targets

    def forward(self, logits, bboxes, batch_idx, image_size):
        if logits is None:
            return bboxes.new_zeros(())
        batch_size, _, feature_h, feature_w = logits.shape
        targets = self.build_targets(
            bboxes,
            batch_idx,
            batch_size,
            (feature_h, feature_w),
            image_size,
            logits.device,
        )
        # Target generation and weighting remain FP32 even under AMP.
        prediction = logits.float().sigmoid()
        error = (prediction - targets).abs()
        mask = (targets > 0).to(dtype=torch.float32)
        rho = mask.mean(dim=(1, 2, 3))
        self.last_rho = rho.detach()

        if self.loss_mode == "mae":
            per_image = error.mean(dim=(1, 2, 3))
        elif self.loss_mode == "balanced":
            foreground_count = mask.sum(dim=(1, 2, 3))
            background_mask = 1.0 - mask
            background_count = background_mask.sum(dim=(1, 2, 3))
            foreground = (mask * error).sum(dim=(1, 2, 3)) / foreground_count.clamp_min(1.0)
            background = (background_mask * error).sum(dim=(1, 2, 3)) / background_count.clamp_min(1.0)
            both = 0.5 * (foreground + background)
            per_image = torch.where(foreground_count > 0, both, background)
            per_image = torch.where(background_count > 0, per_image, foreground)
        else:
            # Per-image rho is the non-zero Gaussian support occupancy on P3.
            # Keep the no-small-object case as an explicit branch so it
            # continues to penalize background pseudo-activations.
            weights = mask / (rho[:, None, None, None] + self.eps) + (1.0 - mask)
            occupancy = (weights * error).mean(dim=(1, 2, 3))
            background_only = error.mean(dim=(1, 2, 3))
            has_foreground = mask.sum(dim=(1, 2, 3)) > 0
            per_image = torch.where(has_foreground, occupancy, background_only)
        return self.loss_weight * per_image.mean()


class _InferenceIdentity(nn.Module):
    def forward(self, x):
        return x


def strip_dags_heads(model: nn.Module) -> int:
    """Physically replace disposable DAGS heads for deployment/export."""

    count = 0
    owner = model
    sequence = None
    # Accept either the internal RTDETRDetectionModel or the common outer
    # ultralytics.RTDETR wrapper (wrapper.model.model is the Sequential graph).
    for _ in range(3):
        candidate = getattr(owner, "model", None)
        if isinstance(candidate, nn.Sequential):
            sequence = candidate
            break
        if not isinstance(candidate, nn.Module):
            break
        owner = candidate
    if sequence is None:
        return count
    for index, module in enumerate(sequence):
        if isinstance(module, MajorRevisionGGASHead):
            replacement = _InferenceIdentity()
            for attr in ("i", "f", "type"):
                if hasattr(module, attr):
                    setattr(replacement, attr, getattr(module, attr))
            replacement.np = 0
            sequence[index] = replacement
            count += 1
    return count


RPSB_CLASSES = {
    cls.__name__: cls
    for cls in (BasicBlock_R0, BasicBlock_R1, BasicBlock_R2, BasicBlock_R3, BasicBlock_R4, BasicBlock_R5)
}


__all__ = (
    "BasicBlock_R0",
    "BasicBlock_R1",
    "BasicBlock_R2",
    "BasicBlock_R3",
    "BasicBlock_R4",
    "BasicBlock_R5",
    "MajorRevisionSTGF",
    "MajorRevisionGGASHead",
    "DAGSLoss",
    "set_active_config",
    "strip_dags_heads",
)
