# STG-DETR Core Source Code

This package contains the core source code for **STG-DETR: A Lightweight Semantic-Texture Guided Detector for Tiny Defects in Power Inspection**, together with development versions.

## Core components

- **RPSB**: Residual Partial Spatial Block for reducing backbone spatial computation.
- **STGF**: Semantic-Texture Gated Fusion for cross-scale feature compensation.
- **DAGS**: Density-Aware Gaussian Supervision, including the auxiliary head, Gaussian target generation, and auxiliary loss.

## Code locations

| Content | Location |
| --- | --- |
| Original RPSB and partial convolution | `ultralytics/nn/extra_modules/block.py` |
| Original STGF and DAGS | `ultralytics/nn/extra_modules/MODU.py` |
| Revised RPSB, STGF, and DAGS | `ablation/major_revision/modules.py` |
| Model construction and loss integration | `ultralytics/nn/tasks.py` |
| Model configurations | `ablation/configs/` and `ablation/major_revision/configs/` |

## Version and usage notes

Earlier implementations, ablation configurations, and backups are retained as development history. The original and revised DAGS implementations use different target-generation and loss formulations; use each configuration with its matching implementation.

Revision configurations require registration through `ablation/major_revision/bootstrap.py` before model construction. This archive is a core source supplement and relies on the original RT-DETR project and its dependencies for execution. The DAGS auxiliary branch is used during training and bypassed during inference.
