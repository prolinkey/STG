"""Smoke-test the untouched A0--A7 configs before formal reproduction."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.major_revision.variants import OVERALL_CONFIGS, RUNS_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", default=str(RUNS_ROOT / "overall_smoke_tests.csv"))
    return parser.parse_args()


def parse_keys(value: str) -> list[str]:
    requested = value.strip().upper()
    keys = list(OVERALL_CONFIGS) if requested == "ALL" else [item.strip().upper() for item in value.split(",") if item.strip()]
    keys = list(dict.fromkeys(keys))
    unknown = [key for key in keys if key not in OVERALL_CONFIGS]
    if unknown or not keys:
        raise ValueError(f"Choose A0--A7 or all; invalid selection: {unknown or value!r}.")
    return keys


def main():
    args = parse_args()
    if args.imgsz != 640:
        raise ValueError("The revision protocol requires imgsz=640.")

    import torch
    from ultralytics.nn.extra_modules.block import BasicBlock_PConv
    from ultralytics.nn.extra_modules.fpf_lite import GGASHead, LiteCSAG
    from ultralytics.nn.tasks import RTDETRDetectionModel

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested, but this PyTorch build has no CUDA support.")

    rows = []
    for key in parse_keys(args.variants):
        setting, config = OVERALL_CONFIGS[key]
        started = time.perf_counter()
        row = {
            "variant": key,
            "part": "Overall",
            "setting": setting,
            "config": str(config),
            "device": str(device),
            "imgsz": args.imgsz,
            "status": "FAILED",
            "forward_ok": False,
            "backward_ok": False,
            "loss": "",
            "grad_l1": "",
            "seconds": "",
            "error": "",
        }
        try:
            with config.open(encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            if not isinstance(cfg, dict) or not isinstance(cfg.get("backbone"), list) or not isinstance(cfg.get("head"), list):
                raise ValueError(f"Incomplete model YAML: {config}")
            lowered = config.read_text(encoding="utf-8").lower()
            if any(token in lowered for token in (".pt", "weights:", "checkpoint:")):
                raise ValueError(f"Pretrained/checkpoint reference found in {config}.")

            torch.manual_seed(0)
            model = RTDETRDetectionModel(str(config), nc=4, verbose=False).to(device)
            modules = list(model.modules())
            expected_rpsb = key in {"A1", "A4", "A5", "A7"}
            expected_stgf = key in {"A2", "A4", "A6", "A7"}
            expected_dags = key in {"A3", "A5", "A6", "A7"}
            if any(isinstance(module, BasicBlock_PConv) for module in modules) != expected_rpsb:
                raise AssertionError(f"{key}: legacy RPSB presence mismatch.")
            if any(isinstance(module, LiteCSAG) for module in modules) != expected_stgf:
                raise AssertionError(f"{key}: legacy STGF presence mismatch.")
            if any(isinstance(module, GGASHead) for module in modules) != expected_dags:
                raise AssertionError(f"{key}: legacy DAGS presence mismatch.")

            image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
            model.eval()
            with torch.no_grad():
                prediction = model(image)
            if prediction is None:
                raise AssertionError("Forward returned None.")
            row["forward_ok"] = True

            model.train()
            model.zero_grad(set_to_none=True)
            batch = {
                "img": image,
                "cls": torch.zeros((1, 1), device=device),
                "bboxes": torch.tensor([[0.5, 0.5, 0.025, 0.025]], device=device),
                "batch_idx": torch.zeros(1, device=device),
            }
            loss, _ = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {float(loss.detach())}")
            loss.backward()
            grad_l1 = sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            if not math.isfinite(grad_l1) or grad_l1 <= 0:
                raise AssertionError(f"No finite parameter gradient: {grad_l1}")
            row.update(status="PASSED", backward_ok=True, loss=float(loss.detach()), grad_l1=grad_l1)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            row["seconds"] = time.perf_counter() - started
            rows.append(row)
            print(f"{key:>3} {row['status']:<6} forward={row['forward_ok']} backward={row['backward_ok']} {row['seconds']:.2f}s")
            if "model" in locals():
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "all_passed": all(row["status"] == "PASSED" for row in rows),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Overall smoke-test report: {output}")
    failed = [row["variant"] for row in rows if row["status"] != "PASSED"]
    if failed:
        raise SystemExit(f"Overall smoke tests failed: {failed}")


if __name__ == "__main__":
    main()
