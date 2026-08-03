"""Online training support for external speculative draft models."""

from .config import external_draft_enabled, should_run_draft_interval, validate_external_draft_args
from .feature_schema import DraftFeatureSample, VersionedFeatureQueue

__all__ = [
    "DraftFeatureSample",
    "VersionedFeatureQueue",
    "external_draft_enabled",
    "should_run_draft_interval",
    "validate_external_draft_args",
]
