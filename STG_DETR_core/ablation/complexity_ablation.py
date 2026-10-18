"""Measure parameters and inference GFLOPs for each ablation config."""

import argparse
import csv
import gc
import sys
from pathlib import Path

from variants import DEFAULT_PROJECT, ROOT, VARIANTS, config_path, parse_variants, variant_flags


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nc", type=int, required=True, help="数据集类别数，例如 VisDrone 为 10。")
    parser.add_argument("--output", default=str(DEFAULT_PROJECT / "ablation_complexity.csv"))
    parser.add_argument("--dry-run", action="store_true", help="只打印配置列表，不导入 PyTorch。")
    return parser.parse_args()


def main():
    args = parse_args()
    keys = parse_variants(args.variants)
    if args.nc < 1 or args.imgsz < 32:
        raise ValueError("nc 必须大于 0，imgsz 必须不小于 32。")
    for key in keys:
        print(f"{key.upper():<3} {VARIANTS[key][0]:<18} {config_path(key)}")
    if args.dry_run:
        return

    sys.path.insert(0, str(ROOT))
    from ultralytics.nn.extra_modules.fpf_lite import GGASHead
    from ultralytics.nn.tasks import RTDETRDetectionModel
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    rows = []
    for key in keys:
        model = RTDETRDetectionModel(str(config_path(key)), nc=args.nc, verbose=False).eval()
        train_params = int(get_num_params(model))
        dags_parameter_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, GGASHead)
            for parameter in module.parameters()
        }
        dags_params = sum(parameter.numel() for parameter in model.parameters() if id(parameter) in dags_parameter_ids)
        rows.append(
            {
                "variant": key,
                "name": VARIANTS[key][0],
                **variant_flags(key),
                "train_parameters": train_params,
                "deploy_active_parameters": train_params - dags_params,
                "dags_training_parameters": dags_params,
                "FP32_train_weights_MiB": train_params * 4 / 1024**2,
                "inference_GFLOPs": float(get_flops(model, imgsz=args.imgsz)),
            }
        )
        del model
        gc.collect()

    print(f"\nInput={args.imgsz}x{args.imgsz}, nc={args.nc}")
    print(f"{'ID':<3} {'R':>1} {'S':>1} {'D':>1} {'Train Params':>14} {'Deploy Params':>14} {'GFLOPs':>9}")
    for row in rows:
        print(
            f"{row['variant'].upper():<3} {row['RPSB']:>1} {row['STGF']:>1} {row['DAGS']:>1} "
            f"{row['train_parameters']:>14,} {row['deploy_active_parameters']:>14,} "
            f"{row['inference_GFLOPs']:>9.1f}"
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"复杂度表已保存：{output}")
    print("DAGS 在 eval 模式下不执行辅助卷积，但未裁剪的检查点仍保存其训练参数。")


if __name__ == "__main__":
    main()
