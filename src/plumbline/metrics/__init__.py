"""Metrics. Calibration reads prob_selected, discrimination reads confidence."""

from plumbline.metrics.calibration import (
    Bin,
    Binning,
    FloorBand,
    brier,
    calibration_floor,
    ece,
    is_distinguishable,
    mce,
    multiclass_brier,
    reliability,
    synthetic_floor,
    verdict,
)
from plumbline.metrics.cascade import (
    CascadeRow,
    candidate_thresholds,
    cascade_sweep,
    optimal_threshold,
)
from plumbline.metrics.discrimination import (
    DEFAULT_THRESHOLDS,
    SweepRow,
    auroc,
    threshold_sweep,
)
from plumbline.metrics.recalibration import (
    MetricSet,
    RecalibrationResult,
    Split,
    apply_temperature_binary,
    fit_temperature_binary,
    fit_temperature_multiclass,
    make_split,
    recalibrate,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "Bin",
    "Binning",
    "CascadeRow",
    "FloorBand",
    "MetricSet",
    "RecalibrationResult",
    "Split",
    "SweepRow",
    "apply_temperature_binary",
    "auroc",
    "brier",
    "calibration_floor",
    "candidate_thresholds",
    "cascade_sweep",
    "ece",
    "fit_temperature_binary",
    "fit_temperature_multiclass",
    "is_distinguishable",
    "make_split",
    "mce",
    "multiclass_brier",
    "optimal_threshold",
    "recalibrate",
    "reliability",
    "synthetic_floor",
    "threshold_sweep",
    "verdict",
]
