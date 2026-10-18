"""Generate the controlled R/S/D/H YAML matrix without touching A0--A7."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from variants import CONFIG_DIR, VARIANTS


def backbone(block_name: str):
    return [
        [-1, 1, "ConvNormLayer", [32, 3, 2, None, False, "relu"]],
        [-1, 1, "ConvNormLayer", [32, 3, 1, None, False, "relu"]],
        [-1, 1, "ConvNormLayer", [64, 3, 1, None, False, "relu"]],
        [-1, 1, "nn.MaxPool2d", [3, 2, 1]],
        [-1, 1, "Blocks", [64, block_name, 2, 2, "relu"]],
        [-1, 1, "Blocks", [128, block_name, 2, 3, "relu"]],
        [-1, 1, "Blocks", [256, block_name, 2, 4, "relu"]],
        [-1, 1, "Blocks", [512, block_name, 2, 5, "relu"]],
    ]


def detector_tail(p3_source: int, offset: int):
    """Return the unchanged RT-DETR hybrid encoder/decoder after optional nodes."""

    base = 8 + offset
    # Local indices: p5proj, aifi, p5out, up, p4proj, cat, p4out,
    # p4reduce, up, p3proj, cat, p3out, down, cat, p4pan, down, cat,
    # p5pan, decoder.
    return [
        [7, 1, "Conv", [256, 1, 1, None, 1, 1, False]],                    # base+0
        [-1, 1, "AIFI", [1024, 8]],                                        # base+1
        [-1, 1, "Conv", [256, 1, 1]],                                      # base+2
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],                     # base+3
        [6, 1, "Conv", [256, 1, 1, None, 1, 1, False]],                    # base+4
        [[-2, -1], 1, "Concat", [1]],                                      # base+5
        [-1, 3, "RepC3", [256, 0.5]],                                      # base+6
        [-1, 1, "Conv", [256, 1, 1]],                                      # base+7
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],                     # base+8
        [p3_source, 1, "Conv", [256, 1, 1, None, 1, 1, False]],            # base+9
        [[-2, -1], 1, "Concat", [1]],                                      # base+10
        [-1, 3, "RepC3", [256, 0.5]],                                      # base+11
        [-1, 1, "Conv", [256, 3, 2]],                                      # base+12
        [[-1, base + 7], 1, "Concat", [1]],                                # base+13
        [-1, 3, "RepC3", [256, 0.5]],                                      # base+14
        [-1, 1, "Conv", [256, 3, 2]],                                      # base+15
        [[-1, base + 2], 1, "Concat", [1]],                                # base+16
        [-1, 3, "RepC3", [256, 0.5]],                                      # base+17
        [[base + 11, base + 14, base + 17], 1, "RTDETRDecoder", ["nc", 256, 300, 4, 8, 3]],
    ]


def head(stgf: dict | None = None, dags_source: str | None = None):
    prefix = []
    p3_source = 5
    if stgf is not None:
        prefix.append(
            [[5, 6], 1, "LiteCSAG", [64, 4, 3, stgf["mode"], stgf["local_conv"], stgf["gamma"]]]
        )
        p3_source = 8
    if dags_source is not None:
        source_index = p3_source if dags_source == "enhanced_p3" else 5
        prefix.append([source_index, 1, "GGASHead", [32, 1.0]])
    return prefix + detector_tail(p3_source=p3_source, offset=len(prefix))


def base_config(key: str, block_name: str, stgf=None, ggas=None, dags_source=None):
    variant = VARIANTS[key]
    config = {
        "nc": 4,
        "scales": {"l": [1.0, 1.0, 1024]},
        "major_revision": {
            "id": key,
            "part": variant.part,
            "setting": variant.setting,
            "imgsz": 640,
            "epochs": 200,
            "pretrained": False,
            "rpsb_transition_rule": "standard_stage_entry; modify shortcut stride1 same-channel blocks only",
        },
        "backbone": backbone(block_name),
        "head": head(stgf=stgf, dags_source=dags_source),
    }
    if stgf is not None:
        config["major_revision"]["stgf"] = deepcopy(stgf)
    if ggas is not None:
        config["ggas"] = deepcopy(ggas)
        config["major_revision"]["dags_source"] = dags_source
    return config


DEFAULT_STGF = {"mode": "full", "local_conv": "partial", "gamma": 0.1}
DEFAULT_GGAS = {
    "enabled": True,
    "loss_mode": "occupancy",
    "tau": 1024.0,
    "eta": 0.50,
    "lambda_g": 0.25,
    "kappa": 3.0,
    "sigma_min": 1.0,
    "eps": 1.0e-6,
    "target_merge": "sum_clip",
}


def all_configs():
    configs = {}
    for index in range(6):
        key = f"R{index}"
        configs[key] = base_config(key, f"BasicBlock_R{index}")

    configs["S0"] = base_config("S0", "BasicBlock_R3")
    s_specs = {
        "S1": {"mode": "add", "local_conv": "partial", "gamma": 0.1},
        "S2": {"mode": "concat", "local_conv": "partial", "gamma": 0.1},
        "S3": {"mode": "spatial", "local_conv": "partial", "gamma": 0.1},
        "S4": {"mode": "channel", "local_conv": "partial", "gamma": 0.1},
        "S5": {"mode": "full", "local_conv": "full", "gamma": 0.1},
        "S6": {"mode": "full", "local_conv": "partial", "gamma": 0.0},
        "S7": {"mode": "full", "local_conv": "partial", "gamma": 0.1},
        "S8": {"mode": "full", "local_conv": "partial", "gamma": 1.0},
    }
    for key, spec in s_specs.items():
        configs[key] = base_config(key, "BasicBlock_R3", stgf=spec)

    configs["D0"] = base_config("D0", "BasicBlock_R3", stgf=DEFAULT_STGF)
    d_specs = {
        "D1": ("mae", "enhanced_p3"),
        "D2": ("balanced", "enhanced_p3"),
        "D3": ("occupancy", "enhanced_p3"),
        "D4": ("occupancy", "raw_p3"),
    }
    for key, (mode, source) in d_specs.items():
        ggas = deepcopy(DEFAULT_GGAS)
        ggas["loss_mode"] = mode
        configs[key] = base_config(key, "BasicBlock_R3", stgf=DEFAULT_STGF, ggas=ggas, dags_source=source)

    sensitivity = {
        "H0": {},
        "H1": {"tau": 16.0**2},
        "H2": {"tau": 48.0**2},
        "H3": {"eta": 0.25},
        "H4": {"eta": 0.75},
        "H5": {"lambda_g": 0.10},
        "H6": {"lambda_g": 0.50},
        "H7": {"kappa": 2.0},
        "H8": {"kappa": 4.0},
        "H9": {"sigma_min": 0.5},
        "H10": {"sigma_min": 1.5},
    }
    for key, override in sensitivity.items():
        ggas = deepcopy(DEFAULT_GGAS)
        ggas.update(override)
        configs[key] = base_config(
            key,
            "BasicBlock_R3",
            stgf=DEFAULT_STGF,
            ggas=ggas,
            dags_source="enhanced_p3",
        )
    return configs


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = all_configs()
    for key, config in configs.items():
        path = VARIANTS[key].config
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True, width=120)

    blocked = {
        "major_revision": {
            "id": "S9",
            "part": "STGF",
            "setting": "FPF reproduction",
            "runnable": False,
            "reason": "No official or independently verifiable FPF source/complete implementation details were found.",
        }
    }
    with VARIANTS["S9"].config.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(blocked, handle, sort_keys=False, allow_unicode=True, width=120)
    print(f"Wrote {len(configs)} runnable configs and one blocked S9 descriptor to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
