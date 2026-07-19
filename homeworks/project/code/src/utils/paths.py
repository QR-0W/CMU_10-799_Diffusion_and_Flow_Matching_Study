"""Canonical project and runtime artifact paths."""

from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATASETS_ROOT = ARTIFACTS_ROOT / "datasets"
CELEBA_ROOT = DATASETS_ROOT / "celeba-subset"
CHECKPOINTS_ROOT = ARTIFACTS_ROOT / "checkpoints"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"
SAMPLES_ROOT = ARTIFACTS_ROOT / "samples"
METRICS_ROOT = ARTIFACTS_ROOT / "metrics"


def resolve_code_path(path: str | Path) -> Path:
    """Resolve relative runtime paths from the code directory."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = CODE_ROOT / resolved
    return resolved.resolve()
