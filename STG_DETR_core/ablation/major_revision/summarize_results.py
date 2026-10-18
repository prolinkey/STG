"""Merge training, complexity, and latency evidence into the revision tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.major_revision.variants import (
    OVERALL_CONFIGS,
    RUNS_ROOT,
    SUMMARY_DIR,
    VARIANTS,
    completed_epochs,
    run_name,
)
from ablation.major_revision.train_major_revision import existing_run_mismatches


FIELDS = [
    "part",
    "setting",
    "seed",
    "config",
    "status",
    "AP",
    "AP50",
    "AP_S",
    "Params_M",
    "GFLOPs",
    "Latency_ms",
    "Latency_std_ms",
    "FPS",
    "Train_Params_M",
    "result_dir",
    "metric_source",
    "evidence_errors",
    "notes",
]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def as_percent(value):
    if value in (None, ""):
        return ""
    number = float(value)
    return number * 100.0 if abs(number) <= 1.0 else number


def select_metric(mapping: dict, names: tuple[str, ...]):
    normalized = {key.strip(): value for key, value in mapping.items()}
    for name in names:
        if name in normalized and normalized[name] not in (None, ""):
            return as_percent(normalized[name])
    return ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_lock_manifest() -> str:
    path = RUNS_ROOT / "metadata" / "dataset_lock.json"
    if not path.is_file():
        return ""
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("manifest_sha256", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def read_best_training_metrics(result_dir: Path, config: Path, dataset_manifest: str):
    """Read only metrics whose fixed-split provenance can be verified."""

    evidence_errors = []
    evaluation = result_dir / "evaluation_metrics.json"
    best_checkpoint = result_dir / "weights" / "best.pt"
    if evaluation.is_file() and best_checkpoint.is_file():
        try:
            payload = json.loads(evaluation.read_text(encoding="utf-8"))
            provenance = payload.get("provenance", {})
            metrics = payload.get("metrics", payload)
            expected = {
                "config_sha256": file_sha256(config),
                "dataset_manifest_sha256": dataset_manifest,
                "checkpoint_sha256": file_sha256(best_checkpoint),
            }
            mismatches = [
                f"evaluation {name}={provenance.get(name)!r} expected {value!r}"
                for name, value in expected.items()
                if not value or provenance.get(name) != value
            ]
            if provenance.get("split") not in {"val", "test"}:
                mismatches.append(f"evaluation split={provenance.get('split')!r} is not fixed val/test")
            if not mismatches:
                return {
                    "AP": select_metric(metrics, ("AP",)),
                    "AP50": select_metric(metrics, ("AP50",)),
                    "AP_S": select_metric(metrics, ("AP_S",)),
                    "source": str(evaluation),
                    "evidence_errors": [],
                }
            evidence_errors.extend(mismatches)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            evidence_errors.append(f"invalid evaluation_metrics.json: {exc}")
    results = read_csv(result_dir / "results.csv")
    if not results:
        return {"AP": "", "AP50": "", "AP_S": "", "source": "", "evidence_errors": evidence_errors}
    ap_key = next((key for key in results[0] if key.strip() == "metrics/mAP50-95(B)"), None)
    if ap_key is None:
        return {
            "AP": "",
            "AP50": "",
            "AP_S": "",
            "source": str(result_dir / "results.csv"),
            "evidence_errors": evidence_errors,
        }
    best = max(results, key=lambda row: float(row.get(ap_key) or "-inf"))
    return {
        "AP": select_metric(best, ("metrics/mAP50-95(B)",)),
        "AP50": select_metric(best, ("metrics/mAP50(B)",)),
        # Ultralytics' training CSV has no COCO area-stratified AP_S.
        "AP_S": "",
        "source": str(result_dir / "results.csv"),
        "evidence_errors": evidence_errors,
    }


def legacy_run_name(key: str, seed: int) -> str:
    stem = OVERALL_CONFIGS[key][1].stem
    suffix = stem.removeprefix(f"{key.lower()}_")
    return f"{key.lower()}_{suffix}_s{seed}"


def discovered_seeds(key: str, overall: bool) -> list[int]:
    seeds = {0}
    roots = [RUNS_ROOT / "overall", RUNS_ROOT / "key_3seeds_overall"] if overall else [RUNS_ROOT]
    prefix = key.lower() + "_"
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir() or not path.name.startswith(prefix) or "_s" not in path.name:
                continue
            try:
                seeds.add(int(path.name.rsplit("_s", 1)[1]))
            except ValueError:
                pass
    return sorted(seeds)


def result_directory(key: str, seed: int, overall: bool) -> Path:
    if not overall:
        return RUNS_ROOT / run_name(key, seed)
    name = legacy_run_name(key, seed)
    preferred = RUNS_ROOT / "overall" / name
    fallback = RUNS_ROOT / "key_3seeds_overall" / name
    candidates = [preferred, fallback]
    # Prefer a complete, metadata-bearing run; otherwise retain the run with
    # the most recorded progress.  This avoids an incomplete single-seed run
    # hiding a completed key-three-seed replicate.
    def score(path: Path):
        epochs = completed_epochs(path / "results.csv")
        complete = epochs >= 200 and (path / "weights" / "best.pt").is_file() and (path / "weights" / "last.pt").is_file()
        return (int(complete), int((path / "run_metadata.json").is_file()), epochs, int(path.exists()))

    return max(candidates, key=score)


def fmt(value, digits=3):
    if value in (None, ""):
        return "待完成"
    try:
        number = float(value)
        return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def status_for(
    result_dir: Path,
    *,
    key: str,
    seed: int,
    config: Path,
    dataset_manifest: str,
    blocked: bool = False,
):
    if blocked:
        return "blocked", []
    epochs = completed_epochs(result_dir / "results.csv")
    best = result_dir / "weights" / "best.pt"
    last = result_dir / "weights" / "last.pt"
    candidate_complete = epochs >= 200 and best.is_file() and last.is_file()
    candidate_incomplete = epochs > 0 or last.is_file()
    if not candidate_complete and not candidate_incomplete:
        return "pending", []
    if not dataset_manifest:
        return "invalid", ["missing shared dataset_lock.json"]
    errors = existing_run_mismatches(
        result_dir,
        variant=key,
        seed=seed,
        config_sha256=file_sha256(config),
        dataset_manifest_sha256=dataset_manifest,
        imgsz=640,
        epochs=200,
    )
    if errors:
        return "invalid", errors
    return ("complete" if candidate_complete else "incomplete"), []


def load_by_variant(path: Path):
    return {row.get("variant", "").upper(): row for row in read_csv(path)}


def verified_measurement(rows: dict, key: str, config: Path) -> tuple[dict, list[str]]:
    row = rows.get(key, {})
    if not row:
        return {}, []
    expected = file_sha256(config)
    if row.get("config_sha256") != expected:
        return {}, [f"stale measurement config_sha256={row.get('config_sha256')!r} expected {expected!r}"]
    return row, []


def build_rows():
    complexity = load_by_variant(SUMMARY_DIR / "complexity.csv")
    latency = load_by_variant(SUMMARY_DIR / "latency.csv")
    dataset_manifest = dataset_lock_manifest()
    rows = []
    for key, (setting, config) in OVERALL_CONFIGS.items():
        for seed in discovered_seeds(key, overall=True):
            result_dir = result_directory(key, seed, overall=True)
            status, evidence_errors = status_for(
                result_dir, key=key, seed=seed, config=config, dataset_manifest=dataset_manifest
            )
            metrics = (
                read_best_training_metrics(result_dir, config, dataset_manifest)
                if status == "complete"
                else {"AP": "", "AP50": "", "AP_S": "", "source": "", "evidence_errors": []}
            )
            comp, comp_errors = verified_measurement(complexity, key, config)
            lat, lat_errors = verified_measurement(latency, key, config)
            evidence_errors.extend(metrics.get("evidence_errors", []) + comp_errors + lat_errors)
            notes = ["A0-A7整体模块组合保持原配置。"]
            if status != "complete":
                notes.append("无满足同一CWPS划分/640/200轮/无预训练/同协议的可核查结果，必须重跑。")
            if metrics["AP_S"] == "":
                notes.append("AP_S缺少固定划分COCO面积分层评估证据。")
            if evidence_errors:
                notes.append("证据校验失败项已记录，未复用相应数值。")
            rows.append(
                {
                    "part": "Overall",
                    "setting": key,
                    "seed": seed,
                    "config": str(config),
                    "status": status,
                    "AP": metrics["AP"],
                    "AP50": metrics["AP50"],
                    "AP_S": metrics["AP_S"],
                    "Params_M": comp.get("Params_M", ""),
                    "GFLOPs": comp.get("GFLOPs", ""),
                    "Latency_ms": lat.get("Latency_ms", ""),
                    "Latency_std_ms": lat.get("Latency_std_ms", ""),
                    "FPS": lat.get("FPS", ""),
                    "Train_Params_M": comp.get("Train_Params_M", ""),
                    "result_dir": str(result_dir),
                    "metric_source": metrics.get("source", ""),
                    "evidence_errors": "; ".join(evidence_errors),
                    "notes": " ".join(notes),
                }
            )

    for key, variant in VARIANTS.items():
        for seed in ([0] if not variant.runnable else discovered_seeds(key, overall=False)):
            result_dir = result_directory(key, seed, overall=False)
            status, evidence_errors = status_for(
                result_dir,
                key=key,
                seed=seed,
                config=variant.config,
                dataset_manifest=dataset_manifest,
                blocked=not variant.runnable,
            )
            metrics = (
                read_best_training_metrics(result_dir, variant.config, dataset_manifest)
                if variant.runnable and status == "complete"
                else {"AP": "", "AP50": "", "AP_S": "", "source": "", "evidence_errors": []}
            )
            comp, comp_errors = verified_measurement(complexity, key, variant.config)
            lat, lat_errors = verified_measurement(latency, key, variant.config)
            evidence_errors.extend(metrics.get("evidence_errors", []) + comp_errors + lat_errors)
            notes = []
            if variant.notes:
                notes.append(variant.notes)
            if status in {"pending", "incomplete", "invalid"}:
                notes.append("待完成；不得据此形成性能结论。")
            if status == "complete" and metrics.get("AP_S", "") == "":
                notes.append("AP_S待固定划分COCO面积分层评估。")
            if evidence_errors:
                notes.append("证据校验失败项已记录，未复用相应数值。")
            rows.append(
                {
                    "part": variant.part,
                    "setting": key,
                    "seed": seed,
                    "config": str(variant.config),
                    "status": status,
                    "AP": metrics.get("AP", ""),
                    "AP50": metrics.get("AP50", ""),
                    "AP_S": metrics.get("AP_S", ""),
                    "Params_M": comp.get("Params_M", ""),
                    "GFLOPs": comp.get("GFLOPs", ""),
                    "Latency_ms": lat.get("Latency_ms", ""),
                    "Latency_std_ms": lat.get("Latency_std_ms", ""),
                    "FPS": lat.get("FPS", ""),
                    "Train_Params_M": comp.get("Train_Params_M", ""),
                    "result_dir": str(result_dir),
                    "metric_source": metrics.get("source", ""),
                    "evidence_errors": "; ".join(evidence_errors),
                    "notes": " ".join(notes),
                }
            )
    return rows


def markdown_table(title: str, rows: list[dict], columns: list[tuple[str, str]]):
    lines = [f"# {title}", "", "仅列入磁盘上可追溯的结果；`待完成` 不代表 0，禁止用于比较。", ""]
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("| " + " | ".join("---" if index < 2 else "---:" for index in range(len(columns))) + " |")
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if key in {"AP", "AP50", "AP_S", "Params_M", "GFLOPs", "Latency_ms", "FPS", "Train_Params_M"}:
                value = fmt(value)
            elif value in (None, ""):
                value = "待完成"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def mean_std(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.fmean(values):.3f} ± {statistics.stdev(values):.3f}"


def table_rows(rows: list[dict], parts: set[str]) -> list[dict]:
    """Use seed 0 for screening, then mean±std once all discovered seeds finish."""

    selected = [row for row in rows if row["part"] in parts]
    ordered_settings = list(dict.fromkeys(row["setting"] for row in selected))
    output = []
    for setting in ordered_settings:
        group = [row for row in selected if row["setting"] == setting]
        complete = [row for row in group if row["status"] == "complete" and row["AP"] != ""]
        if len(group) > 1 and len(complete) == len(group):
            row = dict(complete[0])
            row["status"] = f"complete (n={len(complete)})"
            for field in ("AP", "AP50", "AP_S"):
                values = [float(item[field]) for item in complete if item[field] not in (None, "")]
                row[field] = mean_std(values) if len(values) == len(complete) else ""
        else:
            row = next((item for item in group if int(item["seed"]) == 0), group[0])
            row = dict(row)
        if row.get("Latency_ms", "") not in (None, ""):
            mean = float(row["Latency_ms"])
            std = row.get("Latency_std_ms", "")
            row["Latency_display"] = f"{mean:.3f} ± {float(std):.3f}" if std not in (None, "") else f"{mean:.3f}"
        else:
            row["Latency_display"] = ""
        output.append(row)
    return output


def write_tables(rows):
    overall = table_rows(rows, {"Overall"})
    rpsb_stgf = table_rows(rows, {"RPSB", "STGF"})
    dags = table_rows(rows, {"DAGS", "Sensitivity"})
    common = [
        ("setting", "Setting"),
        ("status", "状态"),
        ("AP", "AP (%)"),
        ("AP50", "AP50 (%)"),
        ("AP_S", "AP_S (%)"),
        ("Params_M", "Params (M)"),
        ("GFLOPs", "GFLOPs"),
        ("Latency_display", "Latency, mean ± std (ms)"),
        ("FPS", "FPS"),
    ]
    (HERE / "TABLE_1_OVERALL_A0_A7.md").write_text(markdown_table("表1 A0–A7整体模块消融", overall, common), encoding="utf-8")
    (HERE / "TABLE_2_RPSB_STGF.md").write_text(markdown_table("表2 RPSB与STGF结构消融", rpsb_stgf, [("part", "Part"), *common]), encoding="utf-8")
    (HERE / "TABLE_3_DAGS_SENSITIVITY.md").write_text(markdown_table("表3 DAGS设计与参数敏感性", dags, [("part", "Part"), *common]), encoding="utf-8")


def complete_rows(rows, settings):
    selected = [row for row in rows if row["setting"] in settings and row["status"] == "complete" and row["AP"] != ""]
    return {row["setting"] for row in selected}


def write_drafts(rows):
    r_ready = complete_rows(rows, {f"R{i}" for i in range(6)})
    s_ready = complete_rows(rows, {f"S{i}" for i in range(9)})
    d_ready = complete_rows(rows, {f"D{i}" for i in range(5)})
    h_ready = complete_rows(rows, {f"H{i}" for i in range(11)})
    evidence_state = (
        f"当前可核查完成数：RPSB {len(r_ready)}/6，STGF {len(s_ready)}/9（S9另行阻塞），"
        f"DAGS {len(d_ready)}/5，敏感性 {len(h_ready)}/11。"
    )
    draft = f"""# 消融实验章节草稿

## 术语表

| 规范术语 | 定义 |
| --- | --- |
| RPSB | Residual Partial Spatial Block，残差部分空间块 |
| STGF | Semantic–Texture Gated Fusion，语义–纹理门控融合 |
| DAGS | Density-Aware Gaussian Supervision，密度感知高斯监督 |
| P3′ | STGF输出的128通道、步长为8的增强骨干侧向特征；不是neck最终256通道P3 |
| AP_S | COCO面积定义下的小目标AP；没有面积分层评估证据时留空 |

## 证据边界

{evidence_state} 当前工作区缺少可验证的CWPS训练结果，以下文本只陈述实验设计与可由代码直接核验的机制；性能数值和优劣判断保留为“待完成”。稿件中既有但没有日志、权重和固定划分评估文件支撑的数值未被复用。

## 整体模块消融

表1保留A0–A7作为整体模块组合实验。A0为RT-DETR-R18基线，A1–A3分别引入RPSB、STGF和DAGS，A4–A6考察两两组合，A7同时启用三者。由于当前目录中没有与统一CWPS划分、640×640输入、200轮、无预训练、相同优化器与增强设置相匹配的原始结果，八组实验均标记为待重新训练。在结果补齐前，不能据此声称统计意义上的“协同效应”；更稳妥的表述是三个模块分别承担结构压缩、定向补偿与显式监督功能，待组合结果验证其功能互补或递进作用。

## RPSB与STGF结构消融

表2首先固定骨干宽度、深度、残差路径和阶段转换，仅在stride=1、输入输出通道相同且采用恒等捷径的内部残差块中改变两次3×3空间卷积。R0–R3比较Full/Partial卷积位置，R4和R5在仅第二次卷积部分化的条件下将参与空间卷积的通道比例分别设为C/2和C/8。该控制消除了阶段降采样与通道投影差异，因此旧A0/A1结果不能替代R0/R3。待R0–R5完成后，应以AP、AP_S、参数量、GFLOPs与实测延迟共同判断卷积位置和r值，而不能仅凭理论计算量确定最终结构。

STGF对照均采用相同的P3/P4输入投影、64通道交互宽度、局部建模和残差输出框架。S1与S2分别使用直接相加和拼接后1×1卷积，S3与S4分别仅保留空间门控和通道门控，S5仅将局部PConv替换为Full 3×3卷积，S6–S8只改变可学习缩放系数γ的初值。S9因缺少可验证的官方FPF实现而未运行。只有在S1–S8的真实结果完整后，才能判断门控、PConv和γ初始化的贡献；若组合提升不足以支持强交互结论，正文应使用“功能互补”或“递进作用”。

## DAGS设计与参数敏感性

表3以R3和完整STGF（γ=0.1）为共同基础。D0移除高斯辅助监督；D1采用全图MAE；D2分别归一化非零高斯支持域与背景误差后等权求和；D3按单张图像在P3上的非零支持域占据率ρ进行逆密度加权，并监督P3′；D4保持相同GGASHead、高斯目标与损失，仅将监督位置改为原始P3。无小目标图像显式退化为全背景MAE，从而继续抑制背景伪激活。目标、掩码、ρ和权重均以FP32计算。DAGS只在训练阶段执行，部署前可将辅助头替换为无参数恒等节点，因此推理路径不包含辅助卷积。

H0–H10采用单因素控制，默认值为τ=32²、η=0.50、λg=0.25、κ=3、σmin=1.0和ε=1e-6。敏感性结果尚未完成，不能预先宣称默认参数最优。结果补齐后，应分别讨论面积阈值、高斯尺度、辅助损失权重、支持半径与最小标准差对AP和AP_S的影响，并明确区分稳定区间与单次随机波动。

## 待填证据

- 统一CWPS划分上的AP、AP50与COCO AP_S。
- CUDA batch=1、100次预热、1000次计时、5次独立重复的延迟均值与标准差。
- 单随机种子筛选后，对A0、A1、A4、A7及关键对照的3随机种子均值与标准差。
"""
    (HERE / "ABLATION_SECTION_DRAFT.md").write_text(draft, encoding="utf-8")

    response = f"""# 审稿意见回复草稿

尊敬的审稿人：

感谢您指出原消融证据不足的问题。我们已将补充实验重构为三层证据链：表1保留A0–A7整体模块组合；表2分别检验RPSB中卷积位置与通道比例，以及STGF中的融合方式、空间/通道门控、局部卷积和γ初始化；表3比较DAGS的损失形式、监督位置与五个关键超参数。{evidence_state}

在代码审计中，我们发现旧RPSB实现不仅将第二次Full 3×3卷积替换为r=4的PConv，还额外加入了残差相加前的ReLU，并在阶段入口块中同样使用PConv。为避免混杂，修订实验统一保留所有阶段入口/降采样块的标准结构，仅修改stride=1、同通道、恒等捷径的内部块。因此，旧A0/A1不能直接替代新的R0/R3，我们已将其列入重跑范围。

我们还核对了DAGS实现与论文公式。旧代码没有计算逐图占据率ρ，而是采用高斯幅值归一化的软前景/背景误差；σmin固定为0.5，且多个高斯采用最大值融合。修订实现按照文稿公式在P3上计算非零支持域占据率，采用求和后裁剪的高斯目标，显式配置τ、η、λg、κ、σmin和ε，并为无小目标图像保留背景约束。所有目标与权重运算均使用FP32。辅助头只在训练期执行，部署模型可物理移除该头。

关于FPF对照，当前工作区仅包含明确标注为“FPF-inspired”的轻量替代模块，其中窗口多头注意力已被PConv和ECA式通道交互替换。由于尚未获得带来源、版本和许可证的官方代码或足以无歧义复现的实现细节，我们没有将该模块冒充FPF，也未运行S9；相应缺失项已在FPF_BLOCKER.md中列明。

目前正式训练仍受两项客观条件阻塞：工作区数据为1类ship小样本而非4类CWPS，且现有RT-DETR环境安装的是CPU版PyTorch。为避免产生不可比较或虚假的数值，我们保留所有未完成单元为空，并将在取得正确CWPS划分和CUDA环境后由安全队列自动续训。修订稿最终只会写入带配置、日志、权重来源、数据指纹和固定划分评估证据的结果。

我们在表述上不把PConv、ECA式通道交互或高斯目标生成本身作为原创，而将贡献边界限定为面向轻量RT-DETR的“结构压缩—定向补偿—显式监督”递进设计。若补充实验不能支持强交互效应，修订稿将使用“功能互补”或“递进作用”，不作过度推断。
"""
    (HERE / "REVIEW_RESPONSE_DRAFT.md").write_text(response, encoding="utf-8")


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    output = SUMMARY_DIR / "all_ablation_results.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
    write_tables(rows)
    write_drafts(rows)
    print(f"Summary: {output} ({len(rows)} rows)")
    print(f"Tables/drafts: {HERE}")


if __name__ == "__main__":
    main()
