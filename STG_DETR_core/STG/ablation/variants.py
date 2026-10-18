

from pathlib import Path


ABLATION_DIR = Path(__file__).resolve().parent
ROOT = ABLATION_DIR.parent
WORKSPACE = ROOT.parent
CONFIG_DIR = ABLATION_DIR / "configs"
DEFAULT_DATA = ROOT / "dataset" / "data.yaml"
DEFAULT_PROJECT = WORKSPACE / "runs" / "ablation"

# key: (display name, config filename, RPSB, STGF, DAGS)
VARIANTS = {
    "a0": ("A0 Baseline", "a0_baseline.yaml", False, False, False),
    "a1": ("A1 RPSB", "a1_rpsb.yaml", True, False, False),
    "a2": ("A2 STGF", "a2_stgf.yaml", False, True, False),
    "a3": ("A3 DAGS", "a3_dags.yaml", False, False, True),
    "a4": ("A4 RPSB+STGF", "a4_rpsb_stgf.yaml", True, True, False),
    "a5": ("A5 RPSB+DAGS", "a5_rpsb_dags.yaml", True, False, True),
    "a6": ("A6 STGF+DAGS", "a6_stgf_dags.yaml", False, True, True),
    "a7": ("A7 Full", "a7_full.yaml", True, True, True),
}


def parse_variants(value):
    """Parse ``all`` or a comma-separated list such as ``a0,a1,a7``."""
    requested = value.strip().lower()
    keys = list(VARIANTS) if requested == "all" else [item.strip().lower() for item in value.split(",")]
    keys = list(dict.fromkeys(key for key in keys if key))
    unknown = [key for key in keys if key not in VARIANTS]
    if unknown:
        raise ValueError(f"未知消融编号 {unknown}。可选值为 {list(VARIANTS)} 或 all。")
    if not keys:
        raise ValueError("至少需要选择一个消融编号。")
    return keys


def parse_seeds(value):
    """Parse a comma-separated integer seed list."""
    seeds = list(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not seeds:
        raise ValueError("至少需要一个随机种子。")
    if any(seed < 0 for seed in seeds):
        raise ValueError("随机种子必须为非负整数。")
    return seeds


def config_path(key):
    """Return the absolute YAML path for one variant."""
    return CONFIG_DIR / VARIANTS[key][1]


def run_name(key, seed):
    """Return a stable output folder name for one variant and seed."""
    suffix = Path(VARIANTS[key][1]).stem.removeprefix(f"{key}_")
    return f"{key}_{suffix}_s{seed}"


def variant_flags(key):
    """Return the three module flags as integers for CSV output."""
    return {
        "RPSB": int(VARIANTS[key][2]),
        "STGF": int(VARIANTS[key][3]),
        "DAGS": int(VARIANTS[key][4]),
    }
