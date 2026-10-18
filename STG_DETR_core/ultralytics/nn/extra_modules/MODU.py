


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv
from .block import Partial_conv3

__all__ = ('LiteCSAG', 'GGASHead', 'GGASLoss')


class ECAGate(nn.Module):
    """Efficient channel gate from ECA-Net without channel reduction."""

    def __init__(self, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('ECA kernel_size must be odd.')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        weight = self.pool(x).squeeze(-1).transpose(-1, -2)
        weight = self.conv(weight).transpose(-1, -2).unsqueeze(-1).sigmoid()
        return weight


class LiteCSAG(nn.Module):
    """Lightweight cross-scale attention guidance from a deep to a shallow feature.

    Args:
        channels (Sequence[int]): Channels of ``[shallow, deep]`` feature maps.
        hidden_channels (int): Projection width used for cross-scale interaction.
        n_div (int): Fraction denominator for partial 3x3 convolution.
        eca_kernel (int): Kernel size of the ECA channel gate.
    """

    def __init__(self, channels, hidden_channels=64, n_div=4, eca_kernel=3):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f'LiteCSAG expects [shallow, deep] channels, got {channels}.')
        shallow_channels, deep_channels = channels
        if hidden_channels < n_div:
            raise ValueError('hidden_channels must be no smaller than n_div.')

        self.shallow_proj = Conv(shallow_channels, hidden_channels, 1, act=nn.ReLU())
        self.deep_proj = Conv(deep_channels, hidden_channels, 1, act=nn.ReLU())
        self.query_mixer = Partial_conv3(hidden_channels, n_div=n_div)
        self.key_mixer = Partial_conv3(hidden_channels, n_div=n_div)
        self.value_mixer = Partial_conv3(hidden_channels, n_div=n_div)
        self.channel_gate = ECAGate(eca_kernel)
        self.out_proj = Conv(hidden_channels, shallow_channels, 1, act=False)
        # A small initial residual scale protects pretrained/shallow texture features.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        shallow, deep = x
        query = self.query_mixer(self.shallow_proj(shallow))
        context = self.deep_proj(deep)
        context = F.interpolate(context, size=query.shape[-2:], mode='nearest')
        key = self.key_mixer(context)

        # Parameter-free spatial correlation replaces costly local multi-head attention.
        spatial_gate = (query * key).mean(dim=1, keepdim=True).sigmoid()
        value = self.value_mixer(context) * spatial_gate
        value = value * self.channel_gate(query + value)
        return shallow + self.residual_scale * self.out_proj(value)


class GGASHead(nn.Module):
    """Training-only lightweight activation head for Gaussian supervision.

    ``grad_scale`` leaves the forward activations unchanged, but scales the
    auxiliary gradient sent back to the detector feature.  This prevents GGAS
    from overwhelming the STGF feature on dense scenes.
    """

    def __init__(self, in_channels, hidden_channels=32, grad_scale=1.0):
        super().__init__()
        if not 0.0 <= float(grad_scale) <= 1.0:
            raise ValueError('GGASHead grad_scale must be in [0, 1].')
        self.grad_scale = float(grad_scale)
        # The disposable branch must not advance the global RNG; otherwise all
        # detector layers declared after it receive different initial weights
        # than the matching no-DAGS ablation, even under the same seed.
        with torch.random.fork_rng(devices=[]):
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
            self.bn = nn.BatchNorm2d(in_channels)
            self.pointwise = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
            self.act = nn.SiLU(inplace=True)
            self.out = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x):
        # At inference/export this branch is an identity and its convolutions are not executed.
        if not self.training:
            return x
        if self.grad_scale != 1.0:
            detached = x.detach()
            x = detached + self.grad_scale * (x - detached)
        x = self.act(self.bn(self.depthwise(x)))
        x = self.act(self.pointwise(x))
        return self.out(x)


class GGASLoss(nn.Module):
    """Density-normalized MAE for truncated anisotropic Gaussian maps."""

    def __init__(self,
                 small_area=1024.0,
                 sigma_ratio=1.0 / 6.0,
                 loss_weight=0.10,
                 truncate_sigma=3.0,
                 background_weight=0.25,
                 eps=1e-6):
        super().__init__()
        self.small_area = float(small_area)
        self.sigma_ratio = float(sigma_ratio)
        self.loss_weight = float(loss_weight)
        self.truncate_sigma = float(truncate_sigma)
        self.background_weight = float(background_weight)
        self.eps = float(eps)

    @torch.no_grad()
    def build_targets(self, bboxes, batch_idx, batch_size, feature_size, image_size, device):
        """Build P3-resolution heatmaps from normalized xywh labels."""
        feature_h, feature_w = feature_size
        image_h, image_w = image_size
        targets = torch.zeros((batch_size, 1, feature_h, feature_w), device=device, dtype=torch.float32)

        if bboxes.numel() == 0:
            return targets

        boxes = bboxes.detach().to(device=device, dtype=torch.float32)
        indices = batch_idx.detach().to(device=device, dtype=torch.long).view(-1)
        for box, image_index in zip(boxes, indices):
            cx, cy, width, height = box
            if width * image_w * height * image_h >= self.small_area:
                continue

            center_x = float(cx.clamp(0, 1) * max(feature_w - 1, 0))
            center_y = float(cy.clamp(0, 1) * max(feature_h - 1, 0))
            sigma_x = max(float(width * feature_w * self.sigma_ratio), 0.5)
            sigma_y = max(float(height * feature_h * self.sigma_ratio), 0.5)
            radius_x = self.truncate_sigma * sigma_x
            radius_y = self.truncate_sigma * sigma_y
            x0 = max(int(math.floor(center_x - radius_x)), 0)
            x1 = min(int(math.ceil(center_x + radius_x)) + 1, feature_w)
            y0 = max(int(math.floor(center_y - radius_y)), 0)
            y1 = min(int(math.ceil(center_y + radius_y)) + 1, feature_h)
            if x0 >= x1 or y0 >= y1:
                continue

            yy = torch.arange(y0, y1, device=device, dtype=torch.float32)
            xx = torch.arange(x0, x1, device=device, dtype=torch.float32)
            try:
                grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
            except TypeError:  # PyTorch < 1.10 compatibility
                grid_y, grid_x = torch.meshgrid(yy, xx)
            gaussian = torch.exp(-0.5 * (((grid_x - center_x) / sigma_x) ** 2 +
                                         ((grid_y - center_y) / sigma_y) ** 2))
            target_slice = targets[image_index, 0, y0:y1, x0:x1]
            # Max fusion preserves individual peaks in crowded scenes.  Summing
            # then clipping produces large saturated plateaus on VisDrone.
            targets[image_index, 0, y0:y1, x0:x1] = torch.maximum(target_slice, gaussian)
        return targets

    def forward(self, logits, bboxes, batch_idx, image_size):
        if logits is None:
            return bboxes.new_zeros(())
        batch_size, _, feature_h, feature_w = logits.shape
        targets = self.build_targets(bboxes,
                                     batch_idx,
                                     batch_size,
                                     (feature_h, feature_w),
                                     image_size,
                                     logits.device)
        prediction = logits.float().sigmoid()
        error = (prediction - targets).abs()
        # Use the Gaussian amplitude itself as the foreground weight.  The old
        # ``targets > 0`` mask treated every tiny 3-sigma tail as full-strength
        # foreground, which is especially harmful when small objects overlap.
        foreground_weight = targets
        background_weight = 1.0 - targets
        foreground_mass = foreground_weight.sum(dim=(1, 2, 3))
        background_mass = background_weight.sum(dim=(1, 2, 3)).clamp_min(self.eps)

        foreground_loss = (foreground_weight * error).sum(dim=(1, 2, 3)) \
            / foreground_mass.clamp_min(self.eps)
        foreground_loss = torch.where(foreground_mass > self.eps,
                                      foreground_loss,
                                      torch.zeros_like(foreground_loss))
        background_loss = (background_weight * error).sum(dim=(1, 2, 3)) / background_mass
        return self.loss_weight * (foreground_loss + self.background_weight * background_loss).mean()
