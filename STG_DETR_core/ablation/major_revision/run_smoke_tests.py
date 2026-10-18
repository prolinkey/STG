"""Build every revision model and run a 640 forward plus minimal backward."""

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

from ablation.major_revision.bootstrap import register_major_revision
from ablation.major_revision.variants import RUNS_ROOT, VARIANTS, parse_variants


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--device", default="cuda:0" if False else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps PyTorch default.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(RUNS_ROOT / "smoke_tests.csv"))
    return parser.parse_args()


def validate_yaml(path: Path, key: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("backbone"), list) or not isinstance(cfg.get("head"), list):
        raise ValueError(f"{path} is not a complete model YAML.")
    meta = cfg.get("major_revision", {})
    if meta.get("id") != key or meta.get("pretrained") is not False:
        raise ValueError(f"{path} has inconsistent id/pretrained metadata.")
    forbidden = [name for name in ("weights", "checkpoint", "pretrained_weights") if cfg.get(name)]
    if forbidden:
        raise ValueError(f"Pretrained-weight fields are forbidden: {forbidden}")
    return cfg


def structural_checks(model, key: str, cfg: dict) -> dict:
    import torch

    from ablation.major_revision.modules import (
        MajorRevisionGGASHead,
        MajorRevisionSTGF,
        PartialConvNormAct,
        strip_dags_heads,
    )

    stages = list(model.model[4:8])
    if key.startswith("R") or cfg["major_revision"].get("rpsb_transition_rule"):
        for stage in stages:
            first = stage.blocks[0]
            if getattr(first, "major_revision_applied", False):
                raise AssertionError(f"{key}: stage-entry block was modified.")
            for block in list(stage.blocks)[1:]:
                if not getattr(block, "major_revision_applied", False):
                    raise AssertionError(f"{key}: interior residual block was not selected.")
                for branch_name in ("branch2a", "branch2b"):
                    branch = getattr(block, branch_name)
                    if isinstance(branch, PartialConvNormAct) and branch.act.__class__.__name__ != (
                        "ReLU" if branch_name == "branch2a" else "Identity"
                    ):
                        raise AssertionError(f"{key}: activation mismatch in {branch_name}.")

    stgf_nodes = [module for module in model.modules() if isinstance(module, MajorRevisionSTGF)]
    dags_nodes = [module for module in model.modules() if isinstance(module, MajorRevisionGGASHead)]
    expects_stgf = any(layer[2] == "LiteCSAG" for layer in cfg["head"])
    expects_dags = bool(cfg.get("ggas", {}).get("enabled", False))
    if len(stgf_nodes) != int(expects_stgf):
        raise AssertionError(f"{key}: expected STGF={expects_stgf}, found {len(stgf_nodes)}.")
    if len(dags_nodes) != int(expects_dags):
        raise AssertionError(f"{key}: expected DAGS={expects_dags}, found {len(dags_nodes)}.")
    if stgf_nodes:
        stgf = stgf_nodes[0]
        expected_gamma = float(cfg["major_revision"]["stgf"]["gamma"])
        if not math.isclose(float(stgf.gamma.detach()), expected_gamma, abs_tol=1e-7):
            raise AssertionError(f"{key}: gamma initialization mismatch.")
        expects_spatial = stgf.mode in {"spatial", "full"}
        expects_channel = stgf.mode in {"channel", "full"}
        if (stgf.key_mixer is not None) != expects_spatial:
            raise AssertionError(f"{key}: spatial/key branch was not structurally added/removed as requested.")
        if (stgf.channel_gate is not None) != expects_channel:
            raise AssertionError(f"{key}: channel gate was not structurally added/removed as requested.")
    if expects_dags:
        criterion = model.ggas_criterion
        ggas = cfg["ggas"]
        expected = {
            "tau": ggas["tau"],
            "eta": ggas["eta"],
            "loss_weight": ggas["lambda_g"],
            "kappa": ggas["kappa"],
            "sigma_min": ggas["sigma_min"],
            "eps": ggas["eps"],
            "loss_mode": ggas["loss_mode"],
        }
        for attr, value in expected.items():
            actual = getattr(criterion, attr)
            if isinstance(value, str):
                if actual != value:
                    raise AssertionError(f"{key}: {attr}={actual!r}, expected {value!r}.")
            elif not math.isclose(float(actual), float(value), rel_tol=1e-7, abs_tol=1e-9):
                raise AssertionError(f"{key}: {attr}={actual}, expected {value}.")

        # Explicit background-only branch: positive finite loss and gradient.
        logits = torch.zeros(1, 1, 80, 80, device=next(model.parameters()).device, requires_grad=True)
        no_boxes = torch.empty(0, 4, device=logits.device)
        no_indices = torch.empty(0, device=logits.device, dtype=torch.long)
        background_loss = criterion(logits, no_boxes, no_indices, (640, 640))
        background_loss.backward()
        if not torch.isfinite(background_loss) or logits.grad is None or logits.grad.abs().sum() <= 0:
            raise AssertionError(f"{key}: background-only DAGS branch has no finite gradient.")

        # Lock the reviewer-facing DAGS contracts: per-image rho, explicit
        # no-positive behavior, FP32 math, summed/clipped targets, source node,
        # and physical deployment stripping through an outer wrapper.
        boxes = torch.tensor([[0.5, 0.5, 0.025, 0.025]], device=logits.device, dtype=torch.float64)
        indices = torch.tensor([0], device=logits.device, dtype=torch.long)
        targets = criterion.build_targets(boxes, indices, 2, (80, 80), (640, 640), logits.device)
        if targets.dtype != torch.float32:
            raise AssertionError(f"{key}: Gaussian target is not FP32.")
        probe = torch.zeros(2, 1, 80, 80, device=logits.device, dtype=torch.float16, requires_grad=True)
        probe_loss = criterion(probe, boxes, indices, (640, 640))
        expected_rho = (targets > 0).float().mean(dim=(1, 2, 3))
        if probe_loss.dtype != torch.float32 or not torch.equal(criterion.last_rho, expected_rho):
            raise AssertionError(f"{key}: FP32/per-image rho contract failed.")
        probe_loss.backward()
        if probe.grad is None or probe.grad.abs().sum() <= 0:
            raise AssertionError(f"{key}: FP32 cast severed the auxiliary gradient.")

        single = criterion.build_targets(boxes[:1], indices, 1, (80, 80), (640, 640), logits.device)
        doubled = criterion.build_targets(
            torch.cat((boxes[:1], boxes[:1])),
            torch.zeros(2, device=logits.device, dtype=torch.long),
            1,
            (80, 80),
            (640, 640),
            logits.device,
        )
        if not torch.equal(doubled, (2.0 * single).clamp(0.0, 1.0)):
            raise AssertionError(f"{key}: Gaussian targets are not sum+clip.")

        big_box = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=logits.device)
        big_target = criterion.build_targets(big_box, indices, 1, (80, 80), (640, 640), logits.device)
        if torch.count_nonzero(big_target):
            raise AssertionError(f"{key}: area>=tau object incorrectly entered the small-object target.")
        source = cfg["major_revision"].get("dags_source")
        expected_from = 8 if source == "enhanced_p3" else 5
        if dags_nodes[0].f != expected_from:
            raise AssertionError(f"{key}: GGAS source f={dags_nodes[0].f}, expected {expected_from}.")

        class _Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Sequential(MajorRevisionGGASHead(4))

        class _Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Inner()

        with torch.random.fork_rng(devices=[]):
            wrapped = _Outer().to(logits.device)
        if strip_dags_heads(wrapped) != 1 or any(isinstance(node, MajorRevisionGGASHead) for node in wrapped.modules()):
            raise AssertionError(f"{key}: deployment strip failed through the RTDETR-style outer wrapper.")
    return {"stgf_nodes": len(stgf_nodes), "dags_nodes": len(dags_nodes)}


def main():
    args = parse_args()
    if args.imgsz != 640:
        raise ValueError("The revision protocol requires imgsz=640.")
    keys = parse_variants(args.variants)

    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested, but this PyTorch build has no CUDA support.")

    rows = []
    for key in keys:
        variant = VARIANTS[key]
        started = time.perf_counter()
        row = {
            "variant": key,
            "config": str(variant.config),
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
            cfg = validate_yaml(variant.config, key)
            register_major_revision(cfg)
            model = RTDETRDetectionModel(str(variant.config), nc=int(cfg["nc"]), verbose=False).to(device)
            row.update(structural_checks(model, key, cfg))

            image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
            model.eval()
            with torch.no_grad():
                output = model(image)
            if output is None:
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
            row.update(
                status="PASSED",
                backward_ok=True,
                loss=float(loss.detach()),
                grad_l1=grad_l1,
            )
        except Exception as exc:  # keep the matrix running unless --fail-fast
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            if args.fail_fast:
                raise
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
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "all_passed": all(row["status"] == "PASSED" for row in rows),
        "results": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [row["variant"] for row in rows if row["status"] != "PASSED"]
    print(f"Smoke-test report: {output}")
    if failed:
        raise SystemExit(f"Smoke tests failed: {failed}")


if __name__ == "__main__":
    main()
