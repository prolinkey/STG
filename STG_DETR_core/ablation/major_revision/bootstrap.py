"""Register isolated major-revision modules with Ultralytics' YAML parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from . import modules


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Model config must contain a YAML mapping: {path}")
    return loaded


def register_major_revision(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Expose revision classes only in the current Python process.

    The legacy source files and their defaults are not edited.  Calling this
    function is therefore an explicit opt-in required by every revision tool.
    """

    config_dict = _load_config(config)
    modules.set_active_config(config_dict)

    import ultralytics.nn.tasks as tasks

    for name, cls in modules.RPSB_CLASSES.items():
        setattr(tasks, name, cls)
    # Rebinding these parser symbols makes the existing identity-based special
    # cases inject channel arguments for the isolated extended classes.
    tasks.LiteCSAG = modules.MajorRevisionSTGF
    tasks.GGASHead = modules.MajorRevisionGGASHead
    tasks.GGASLoss = modules.DAGSLoss
    return config_dict


__all__ = ("register_major_revision",)
