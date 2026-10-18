"""Measure inference parameters and 640x640 GFLOPs for all runnable variants."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.major_revision.bootstrap import register_major_revision
from ablation.major_revision.variants import OVERALL_CONFIGS, RUNS_ROOT, VARIANTS, parse_variants


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default=str(RUNS_ROOT / "summary" / "complexity.csv"))
    parser.add_argument("--include-overall", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.imgsz != 640:
        raise ValueError("The revision protocol requires a 640x640 complexity input.")

    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel
    from ultralytics.utils.torch_utils import get_flops
    from ultralytics.nn.extra_modules.fpf_lite import GGASHead as LegacyGGASHead
    from ablation.major_revision.modules import MajorRevisionGGASHead

    rows = []
    if args.include_overall:
        # Build the untouched A0--A7 configs before the opt-in parser shim is
        # registered, so these measurements use exactly the legacy classes.
        for key, (setting, config) in OVERALL_CONFIGS.items():
            torch.manual_seed(0)
            model = RTDETRDetectionModel(str(config), nc=4, verbose=False).eval()
            total = sum(parameter.numel() for parameter in model.parameters())
            dags_ids = {
                id(parameter)
                for module in model.modules()
                if isinstance(module, LegacyGGASHead)
                for parameter in module.parameters()
            }
            dags = sum(parameter.numel() for parameter in model.parameters() if id(parameter) in dags_ids)
            gflops = float(get_flops(model, imgsz=args.imgsz))
            rows.append(
                {
                    "variant": key,
                    "part": "Overall",
                    "setting": setting,
                    "config": str(config),
                    "config_sha256": sha256(config),
                    "Train_Params": total,
                    "Train_Params_M": total / 1e6,
                    "Params": total - dags,
                    "Params_M": (total - dags) / 1e6,
                    "DAGS_Params": dags,
                    "GFLOPs": gflops,
                    "imgsz": args.imgsz,
                    "nc": 4,
                    "notes": "Untouched legacy A0-A7 config; complexity rebuilt with CWPS nc=4.",
                }
            )
            print(f"{key:>3} Params={(total-dags)/1e6:.6f}M Train={total/1e6:.6f}M GFLOPs={gflops:.6f}")
            del model
            gc.collect()

    for key in parse_variants(args.variants):
        variant = VARIANTS[key]
        with variant.config.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if cfg.get("major_revision", {}).get("pretrained") is not False:
            raise AssertionError(f"{key}: pretrained must be false.")
        register_major_revision(cfg)
        torch.manual_seed(0)
        model = RTDETRDetectionModel(str(variant.config), nc=int(cfg["nc"]), verbose=False).eval()
        total = sum(parameter.numel() for parameter in model.parameters())
        dags_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, MajorRevisionGGASHead)
            for parameter in module.parameters()
        }
        dags = sum(parameter.numel() for parameter in model.parameters() if id(parameter) in dags_ids)
        gflops = float(get_flops(model, imgsz=args.imgsz))
        row = {
            "variant": key,
            "part": variant.part,
            "setting": variant.setting,
            "config": str(variant.config),
            "config_sha256": sha256(variant.config),
            "Train_Params": total,
            "Train_Params_M": total / 1e6,
            "Params": total - dags,
            "Params_M": (total - dags) / 1e6,
            "DAGS_Params": dags,
            "GFLOPs": gflops,
            "imgsz": args.imgsz,
            "nc": int(cfg["nc"]),
            "notes": "DAGS head excluded from deploy Params and inactive in eval forward." if dags else "",
        }
        rows.append(row)
        print(f"{key:>3} Params={row['Params_M']:.6f}M Train={row['Train_Params_M']:.6f}M GFLOPs={gflops:.6f}")
        del model
        gc.collect()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Complexity report: {output}")


if __name__ == "__main__":
    main()
