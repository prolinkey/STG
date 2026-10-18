"""Select each run's best validation epoch and aggregate seeds."""

import argparse
import csv
import statistics
from pathlib import Path

from variants import DEFAULT_PROJECT, VARIANTS, parse_seeds, parse_variants, run_name, variant_flags


METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--output", default=None, help="验证集汇总 CSV；默认写入 project。")
    parser.add_argument("--allow-missing", action="store_true", help="跳过缺少 results.csv 的运行。")
    return parser.parse_args()


def read_best(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(file)]
    if not rows:
        raise RuntimeError(f"结果文件为空：{path}")
    missing = [metric for metric in METRICS if metric not in rows[0]]
    if missing:
        raise KeyError(f"{path} 缺少指标列：{missing}")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {"epoch": int(float(best["epoch"])), **{metric: float(best[metric]) for metric in METRICS}}


def mean_std(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def short_metric(metric):
    return metric.removeprefix("metrics/").removesuffix("(B)")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def calculate_main_effects(aggregates):
    if {row["variant"] for row in aggregates} != set(VARIANTS):
        return []
    effects = []
    for module in ("RPSB", "STGF", "DAGS"):
        enabled = [row for row in aggregates if row[module] == 1]
        disabled = [row for row in aggregates if row[module] == 0]
        row = {"module": module}
        for metric in METRICS:
            name = short_metric(metric)
            row[f"{name}_main_effect"] = statistics.fmean(item[f"{name}_mean"] for item in enabled) - statistics.fmean(
                item[f"{name}_mean"] for item in disabled
            )
        effects.append(row)
    return effects


def main():
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    seeds = parse_seeds(args.seeds)
    output = Path(args.output).expanduser().resolve() if args.output else project / "ablation_val_summary.csv"
    aggregates = []

    for key in parse_variants(args.variants):
        records = []
        for seed in seeds:
            results = project / run_name(key, seed) / "results.csv"
            if not results.is_file():
                if args.allow_missing:
                    print(f"跳过缺失结果：{results}")
                    continue
                raise FileNotFoundError(f"缺少训练结果：{results}")
            records.append(read_best(results))
        if not records:
            continue

        row = {
            "variant": key,
            "name": VARIANTS[key][0],
            **variant_flags(key),
            "runs": len(records),
        }
        epoch_mean, epoch_std = mean_std([record["epoch"] for record in records])
        row.update({"best_epoch_mean": epoch_mean, "best_epoch_std": epoch_std})
        for metric in METRICS:
            mean, std = mean_std([record[metric] for record in records])
            name = short_metric(metric)
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        aggregates.append(row)

    if not aggregates:
        raise RuntimeError("没有可汇总的验证集结果。")
    write_csv(output, aggregates)

    print(f"{'ID':<3} {'R':>1} {'S':>1} {'D':>1} {'mAP50 mean±std':>22} {'mAP50-95 mean±std':>22}")
    for row in aggregates:
        map50 = f"{row['mAP50_mean']:.4f}±{row['mAP50_std']:.4f}"
        map5095 = f"{row['mAP50-95_mean']:.4f}±{row['mAP50-95_std']:.4f}"
        print(f"{row['variant'].upper():<3} {row['RPSB']:>1} {row['STGF']:>1} {row['DAGS']:>1} {map50:>22} {map5095:>22}")
    print(f"验证集汇总已保存：{output}")

    effects = calculate_main_effects(aggregates)
    if effects:
        effect_output = output.with_name("ablation_main_effects.csv")
        write_csv(effect_output, effects)
        print(f"三模块平均主效应已保存：{effect_output}")
    else:
        print("未包含完整 A0 至 A7，因此不计算 2^3 因子主效应。")


if __name__ == "__main__":
    main()
