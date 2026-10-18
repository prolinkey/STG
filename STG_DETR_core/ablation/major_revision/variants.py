"""Registry for the isolated major-revision experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_DIR = HERE / "configs"
RUNS_ROOT = ROOT / "major_revision_runs"
SUMMARY_DIR = RUNS_ROOT / "summary"
DEFAULT_DATA = ROOT / "dataset" / "data.yaml"

OVERALL_CONFIGS = {
    "A0": ("Baseline", ROOT / "ablation" / "configs" / "a0_baseline.yaml"),
    "A1": ("RPSB", ROOT / "ablation" / "configs" / "a1_rpsb.yaml"),
    "A2": ("STGF", ROOT / "ablation" / "configs" / "a2_stgf.yaml"),
    "A3": ("DAGS", ROOT / "ablation" / "configs" / "a3_dags.yaml"),
    "A4": ("RPSB+STGF", ROOT / "ablation" / "configs" / "a4_rpsb_stgf.yaml"),
    "A5": ("RPSB+DAGS", ROOT / "ablation" / "configs" / "a5_rpsb_dags.yaml"),
    "A6": ("STGF+DAGS", ROOT / "ablation" / "configs" / "a6_stgf_dags.yaml"),
    "A7": ("RPSB+STGF+DAGS", ROOT / "ablation" / "configs" / "a7_full.yaml"),
}


@dataclass(frozen=True)
class Variant:
    key: str
    part: str
    setting: str
    description: str
    config_name: str
    runnable: bool = True
    notes: str = ""

    @property
    def config(self) -> Path:
        return CONFIG_DIR / self.config_name


def _v(key, part, setting, description, runnable=True, notes=""):
    suffix = "yaml" if runnable else "BLOCKED.yaml"
    return Variant(key, part, setting, description, f"{key.lower()}_{description.lower().replace(' ', '_')}.{suffix}", runnable, notes)


VARIANTS = {
    "R0": _v("R0", "RPSB", "Full+Full", "full_full"),
    "R1": _v("R1", "RPSB", "PConv+PConv, r=4", "pconv_pconv_r4"),
    "R2": _v("R2", "RPSB", "PConv+Full, r=4", "pconv_full_r4"),
    "R3": _v("R3", "RPSB", "Full+PConv, r=4", "full_pconv_r4"),
    "R4": _v("R4", "RPSB", "Full+PConv, r=2", "full_pconv_r2"),
    "R5": _v("R5", "RPSB", "Full+PConv, r=8", "full_pconv_r8"),
    "S0": _v("S0", "STGF", "No STGF", "no_stgf"),
    "S1": _v("S1", "STGF", "Direct addition", "direct_add"),
    "S2": _v("S2", "STGF", "Concat+1x1", "concat_1x1"),
    "S3": _v("S3", "STGF", "Spatial gate only", "spatial_only"),
    "S4": _v("S4", "STGF", "Channel gate only", "channel_only"),
    "S5": _v("S5", "STGF", "Full gates+FullConv", "fullconv"),
    "S6": _v("S6", "STGF", "Full STGF, gamma=0", "gamma_0"),
    "S7": _v("S7", "STGF", "Full STGF, gamma=0.1", "gamma_0p1"),
    "S8": _v("S8", "STGF", "Full STGF, gamma=1", "gamma_1"),
    "S9": _v(
        "S9",
        "STGF",
        "FPF reproduction",
        "fpf",
        runnable=False,
        notes="Blocked: no official or independently verifiable FPF implementation is present.",
    ),
    "D0": _v("D0", "DAGS", "No Gaussian supervision", "no_dags"),
    "D1": _v("D1", "DAGS", "Global MAE on P3'", "mae_p3prime"),
    "D2": _v("D2", "DAGS", "Balanced MAE on P3'", "balanced_p3prime"),
    "D3": _v("D3", "DAGS", "Occupancy MAE on P3'", "occupancy_p3prime"),
    "D4": _v("D4", "DAGS", "Occupancy MAE on P3", "occupancy_p3"),
    "H0": _v("H0", "Sensitivity", "Default", "default"),
    "H1": _v("H1", "Sensitivity", "tau=16^2", "tau_16sq"),
    "H2": _v("H2", "Sensitivity", "tau=48^2", "tau_48sq"),
    "H3": _v("H3", "Sensitivity", "eta=0.25", "eta_0p25"),
    "H4": _v("H4", "Sensitivity", "eta=0.75", "eta_0p75"),
    "H5": _v("H5", "Sensitivity", "lambda_g=0.10", "lambda_0p10"),
    "H6": _v("H6", "Sensitivity", "lambda_g=0.50", "lambda_0p50"),
    "H7": _v("H7", "Sensitivity", "kappa=2", "kappa_2"),
    "H8": _v("H8", "Sensitivity", "kappa=4", "kappa_4"),
    "H9": _v("H9", "Sensitivity", "sigma_min=0.5", "sigma_min_0p5"),
    "H10": _v("H10", "Sensitivity", "sigma_min=1.5", "sigma_min_1p5"),
}


def parse_variants(value: str, include_blocked: bool = False) -> list[str]:
    requested = value.strip().upper()
    keys = list(VARIANTS) if requested == "ALL" else [item.strip().upper() for item in value.split(",") if item.strip()]
    keys = list(dict.fromkeys(keys))
    unknown = [key for key in keys if key not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)} or all.")
    if not include_blocked:
        keys = [key for key in keys if VARIANTS[key].runnable]
    if not keys:
        raise ValueError("No runnable variants selected.")
    return keys


def run_name(key: str, seed: int) -> str:
    return f"{key.lower()}_s{seed}"


def completed_epochs(results_csv: Path) -> int:
    import csv

    if not results_csv.is_file():
        return 0
    try:
        with results_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return 0
        epoch_key = next((name for name in rows[-1] if name.strip() == "epoch"), None)
        return int(float(rows[-1][epoch_key])) + 1 if epoch_key else 0
    except (OSError, TypeError, ValueError, KeyError):
        return 0


__all__ = (
    "CONFIG_DIR",
    "DEFAULT_DATA",
    "HERE",
    "OVERALL_CONFIGS",
    "ROOT",
    "RUNS_ROOT",
    "SUMMARY_DIR",
    "VARIANTS",
    "Variant",
    "completed_epochs",
    "parse_variants",
    "run_name",
)
