"""Loading of the project configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML configuration.

    Parameters
    ----------
    path:
        Path to a YAML file. Defaults to ``configs/config.yaml`` in the project root.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(relative: str | Path) -> Path:
    """Return an absolute path inside the project for a path given relative to its root."""
    return PROJECT_ROOT / relative
