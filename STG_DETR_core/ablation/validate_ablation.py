"""Validate the A0-A7 matrix without starting training."""

import argparse
import py_compile
import sys
from pathlib import Path

from variants import ABLATION_DIR, DEFAULT_DATA, ROOT, VARIANTS, config_path, parse_variants


SCRIPTS = (
    "variants.py",
    "train_ablation.py",
    "test_ablation.py",
    "summarize_ablation.py",
    "complexity_ablation.py",
    "validate_ablation.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--build-models", action="store_true", help="导入本地 PyTorch 并实际构建每个模型。")
    parser.add_argument("--nc", type=int, default=1, help="配合 --build-models 使用的类别数。")
    return parser.parse_args()


def normalized_yaml_lines(path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            lines.append(value)
    return lines


def check_dataset_yaml(path):
    errors = []
    if not path.is_file():
        return [f"数据集 YAML 不存在：{path}"]
    text = path.read_text(encoding="utf-8")
    for key in ("train:", "val:", "names:"):
        if key not in text:
            errors.append(f"数据集 YAML 缺少 {key}")
    return errors


def main():
    args = parse_args()
    keys = parse_variants(args.variants)
    errors = []

    for script in SCRIPTS:
        path = ABLATION_DIR / script
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"脚本编译失败 {path}: {exc}")

    seen = set()
    print(f"{'ID':<3} {'RPSB':<5} {'STGF':<5} {'DAGS':<5} 配置")
    for key in keys:
        path = config_path(key)
        if not path.is_file():
            errors.append(f"缺少配置：{path}")
            continue
        text = path.read_text(encoding="utf-8")
        actual = (
            "BasicBlock_PConv" in text,
            "LiteCSAG" in text,
            "GGASHead" in text and "enabled: true" in text,
        )
        expected = VARIANTS[key][2:5]
        if actual != expected:
            errors.append(f"{key.upper()} 模块矩阵不一致，期望 {expected}，实际 {actual}")
        if text.count("RTDETRDecoder") != 1:
            errors.append(f"{key.upper()} 必须且只能包含一个 RTDETRDecoder。")
        normalized = tuple(normalized_yaml_lines(path))
        if normalized in seen:
            errors.append(f"{key.upper()} 与另一份配置结构完全重复。")
        seen.add(normalized)
        print(f"{key.upper():<3} {int(actual[0]):<5} {int(actual[1]):<5} {int(actual[2]):<5} {path.name}")

    baseline = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-r18.yaml"
    full = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-r1.yaml"
    if "a0" in keys and normalized_yaml_lines(config_path("a0")) != normalized_yaml_lines(baseline):
        errors.append("A0 与 rtdetr-r18.yaml 的有效结构不一致。")
    if "a7" in keys and normalized_yaml_lines(config_path("a7")) != normalized_yaml_lines(full):
        errors.append("A7 与 rtdetr-r1.yaml 的有效结构不一致。")

    errors.extend(check_dataset_yaml(Path(args.data).expanduser().resolve()))

    if args.build_models:
        if args.nc < 1:
            errors.append("--nc 必须大于 0。")
        else:
            sys.path.insert(0, str(ROOT))
            try:
                from ultralytics.nn.tasks import RTDETRDetectionModel

                for key in keys:
                    RTDETRDetectionModel(str(config_path(key)), nc=args.nc, verbose=False)
                    print(f"模型构建通过：{key.upper()}")
            except Exception as exc:
                errors.append(f"模型构建失败：{type(exc).__name__}: {exc}")

    if errors:
        print("\n检查失败：")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("\n检查通过：脚本可编译，A0 至 A7 模块矩阵正确，A0/A7 与主配置一致。")


if __name__ == "__main__":
    main()
