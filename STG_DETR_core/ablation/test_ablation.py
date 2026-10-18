"""Evaluate ablation checkpoints on val/test and aggregate seeds."""

import argparse
import csv
import statistics
import sys
from pathlib import Path

from variants import (
    DEFAULT_DATA,
    DEFAULT_PROJECT,
    ROOT,
    VARIANTS,
    parse_seeds,
    parse_variants,
    run_name,
    variant_flags,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="train_ablation.py 的输出目录。")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", help="跳过缺失 best.pt 的任务。")
    parser.add_argument("--dry-run", action="store_true", help="只打印待评估权重，不导入 PyTorch。")
    return parser.parse_args()


def scalar(value):
    return float(value.item()) if hasattr(value, "item") else float(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    summary = []
    for key in VARIANTS:
        records = [row for row in rows if row["variant"] == key]
        if not records:
            continue
        result = {
            "variant": key,
            "name": VARIANTS[key][0],
            **variant_flags(key),
            "runs": len(records),
        }
        numeric_keys = [
            field
            for field in records[0]
            if field not in {"variant", "name", "seed", "weights", "RPSB", "STGF", "DAGS"}
            and all(isinstance(record.get(field), (int, float)) for record in records)
        ]
        for field in numeric_keys:
            values = [float(record[field]) for record in records]
            result[f"{field}_mean"] = statistics.fmean(values)
            result[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def main():
    args = parse_args()
    keys = parse_variants(args.variants)
    seeds = parse_seeds(args.seeds)
    data = Path(args.data).expanduser().resolve()
    project = Path(args.project).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"没有找到数据集配置：{data}")

    checkpoints = []
    for seed in seeds:
        for key in keys:
            name = run_name(key, seed)
            weights = project / name / "weights" / "best.pt"
            checkpoints.append((key, seed, name, weights))
            print(f"{name:<24} -> {weights}")
    if args.dry_run:
        missing = [str(weights) for _, _, _, weights in checkpoints if not weights.is_file()]
        print(f"\n共 {len(checkpoints)} 个任务，缺少 {len(missing)} 个 best.pt。")
        return

    sys.path.insert(0, str(ROOT))
    from ultralytics import RTDETR

    eval_project = project / f"{args.split}_evaluation"
    rows = []
    for key, seed, name, weights in checkpoints:
        if not weights.is_file():
            if args.skip_missing:
                print(f"跳过缺失权重：{weights}")
                continue
            raise FileNotFoundError(f"缺少权重：{weights}")

        print(f"\n===== 评估 {name}: {weights} =====")
        model = RTDETR(str(weights))
        metrics = model.val(
            data=str(data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(eval_project),
            name=name,
            plots=args.plots,
            exist_ok=True,
        )
        row = {
            "variant": key,
            "name": VARIANTS[key][0],
            **variant_flags(key),
            "seed": seed,
            "weights": str(weights),
        }
        row.update({metric_name: scalar(value) for metric_name, value in metrics.results_dict.items()})
        if hasattr(metrics, "box") and hasattr(metrics.box, "maps"):
            names = getattr(metrics, "names", model.names)
            for class_id, ap in enumerate(metrics.box.maps):
                class_name = names[class_id] if isinstance(names, dict) else names[class_id]
                row[f"AP50-95_class/{class_name}"] = scalar(ap)
        rows.append(row)

    if not rows:
        raise RuntimeError("没有可汇总的评估结果。")
    detail_output = project / f"ablation_{args.split}_results.csv"
    summary_output = project / f"ablation_{args.split}_summary.csv"
    write_csv(detail_output, rows)
    write_csv(summary_output, aggregate_rows(rows))
    print(f"\n逐次评估结果：{detail_output}")
    print(f"跨种子均值与标准差：{summary_output}")
    print("注意：逐类别 AP50-95 不等于按目标面积计算的 COCO AP_S。")


if __name__ == "__main__":
    main()
