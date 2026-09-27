"""Persistence-Constrained Break Filter for standard COLD records.

PCBF reclassifies confirmed breaks without refitting COLD or changing the
fitted segment models. It preserves the existing category rule and then
evaluates spectral change persistence using the post-break CCDC model and the
prediction band of the pre-break CCDC model.
"""

from __future__ import annotations

import numpy as np


__all__ = ["cold_pcbf"]

CATEGORY_BANDS = (2, 3, 4)  # Red, NIR, SWIR1
PCBF_BANDS = (1, 2, 3, 4, 5)  # Green, Red, NIR, SWIR1, SWIR2
CATEGORY_THRESHOLD = -200.0
REQUIRED_FIELDS = {
    "t_start",
    "t_end",
    "t_break",
    "pos",
    "change_prob",
    "coefs",
    "rmse",
    "magnitude",
}


def _validate_records(records: np.ndarray) -> None:
    if not isinstance(records, np.ndarray) or records.ndim != 1:
        raise ValueError("records must be a one-dimensional NumPy structured array")
    names = set(records.dtype.names or ())
    missing = REQUIRED_FIELDS - names
    if missing:
        raise ValueError(f"records are missing required fields: {sorted(missing)}")
    if records.dtype["coefs"].shape[0] <= max(PCBF_BANDS):
        raise ValueError("coefs does not contain the six reflective bands")
    if records.dtype["coefs"].shape[1] < 8:
        raise ValueError("coefs must contain the eight-term COLD model")
    for field in ("rmse", "magnitude"):
        if records.dtype[field].shape[0] <= max(PCBF_BANDS):
            raise ValueError(f"{field} does not contain the six reflective bands")


def _require_finite(values, context: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite {context}")


def _harmonic_basis(start: int, end: int) -> np.ndarray:
    dates = np.arange(start, end + 1, dtype=np.float64)
    phase = 2.0 * np.pi * dates / 365.25
    return np.vstack(
        (
            np.ones(dates.size, dtype=np.float64),
            dates / 10000.0,
            np.cos(phase),
            np.sin(phase),
            np.cos(2.0 * phase),
            np.sin(2.0 * phase),
            np.cos(3.0 * phase),
            np.sin(3.0 * phase),
        )
    )


def _is_minus200_category2(record, post_break_record, threshold: float) -> bool:
    magnitude = np.asarray(record["magnitude"])[list(CATEGORY_BANDS)]
    _require_finite(magnitude, "category magnitude")
    greener = bool(
        magnitude[1] > threshold
        and magnitude[0] < -threshold
        and magnitude[2] < -threshold
    )
    if not greener:
        return False

    pre_break_slopes = np.asarray(record["coefs"])[list(CATEGORY_BANDS), 1]
    post_break_slopes = np.asarray(post_break_record["coefs"])[list(CATEGORY_BANDS), 1]
    _require_finite(pre_break_slopes, "category pre-break slopes")
    _require_finite(post_break_slopes, "category post-break slopes")
    afforestation = bool(
        post_break_slopes[1] > abs(pre_break_slopes[1])
        and post_break_slopes[0] < -abs(pre_break_slopes[0])
        and post_break_slopes[2] < -abs(pre_break_slopes[2])
    )
    return not afforestation


def _spectral_recovery_within_duration_threshold(
    record,
    post_break_record,
    *,
    duration_threshold_days: int,
    z_value: float,
    required_bands: int,
) -> bool:
    bands = list(PCBF_BANDS)
    pre_break_coefs = np.asarray(record["coefs"], dtype=np.float64)[bands]
    post_break_coefs = np.asarray(post_break_record["coefs"], dtype=np.float64)[bands]
    magnitudes = np.asarray(record["magnitude"], dtype=np.float64)[bands]
    rmses = np.asarray(record["rmse"], dtype=np.float64)[bands]
    _require_finite(pre_break_coefs, "PCBF pre-break coefficients")
    _require_finite(post_break_coefs, "PCBF post-break coefficients")
    _require_finite(magnitudes, "PCBF magnitude")
    _require_finite(rmses, "PCBF RMSE")
    if np.any(rmses < 0):
        raise ValueError("negative PCBF RMSE")

    break_date = int(record["t_end"])
    start = max(int(post_break_record["t_start"]), break_date)
    end = min(
        int(post_break_record["t_end"]),
        break_date + duration_threshold_days - 1,
    )
    if end < start:
        return False

    basis = _harmonic_basis(start, end)
    pre_break_prediction = pre_break_coefs @ basis
    post_break_prediction = post_break_coefs @ basis
    satisfies_recovery_condition = np.zeros(
        pre_break_prediction.shape,
        dtype=bool,
    )
    negative = magnitudes < 0
    positive = magnitudes > 0
    zero = ~(negative | positive)
    satisfies_recovery_condition[negative] = post_break_prediction[negative] >= (
        pre_break_prediction[negative] - z_value * rmses[negative, None]
    )
    satisfies_recovery_condition[positive] = post_break_prediction[positive] <= (
        pre_break_prediction[positive] + z_value * rmses[positive, None]
    )
    satisfies_recovery_condition[zero] = np.abs(
        post_break_prediction[zero] - pre_break_prediction[zero]
    ) <= (z_value * rmses[zero, None])
    return bool(
        np.any(
            np.count_nonzero(satisfies_recovery_condition, axis=0)
            >= required_bands
        )
    )


def _pcbf_retention_mask(
    cold_results: np.ndarray,
    *,
    duration_threshold_days: int = 192,
    z_value: float = 2.326,
    required_bands: int = 4,
) -> np.ndarray:
    """Return a record-aligned mask for breaks retained by PCBF.

    Non-break records are always true. Confirmed breaks are records with
    ``t_break > 0`` and ``change_prob == 100``. A confirmed break without a
    post-break CCDC segment for the same pixel is retained conservatively.
    """

    _validate_records(cold_results)
    if (
        not isinstance(duration_threshold_days, (int, np.integer))
        or duration_threshold_days <= 0
    ):
        raise ValueError("duration_threshold_days must be a positive integer")
    if not np.isfinite(z_value) or z_value <= 0:
        raise ValueError("z_value must be finite and positive")
    if (
        not isinstance(required_bands, (int, np.integer))
        or required_bands < 1
        or required_bands > len(PCBF_BANDS)
    ):
        raise ValueError("required_bands must be between 1 and 5")

    retained = np.ones(cold_results.size, dtype=bool)
    confirmed = (cold_results["t_break"] > 0) & (
        cold_results["change_prob"] == 100
    )
    bands = list(PCBF_BANDS)
    for index in np.flatnonzero(confirmed):
        record = cold_results[index]
        coefs = np.asarray(record["coefs"], dtype=np.float64)[bands]
        magnitudes = np.asarray(record["magnitude"], dtype=np.float64)[bands]
        rmses = np.asarray(record["rmse"], dtype=np.float64)[bands]
        _require_finite(coefs, "PCBF pre-break coefficients")
        _require_finite(magnitudes, "PCBF magnitude")
        _require_finite(rmses, "PCBF RMSE")
        if np.any(rmses < 0):
            raise ValueError("negative PCBF RMSE")

        next_index = int(index) + 1
        if (
            next_index >= cold_results.size
            or cold_results[next_index]["pos"] != record["pos"]
        ):
            continue
        post_break_record = cold_results[next_index]
        if _is_minus200_category2(
            record,
            post_break_record,
            CATEGORY_THRESHOLD,
        ):
            retained[index] = False
            continue
        if _spectral_recovery_within_duration_threshold(
            record,
            post_break_record,
            duration_threshold_days=int(duration_threshold_days),
            z_value=float(z_value),
            required_bands=int(required_bands),
        ):
            retained[index] = False
    return retained


def cold_pcbf(
    cold_results,
    duration_threshold_days=192,
    z_value=2.326,
    required_bands=4,
) -> np.ndarray:
    """Apply PCBF to standard COLD records and return a processed copy.

    Parameters
    ----------
    cold_results : numpy.ndarray
        One-dimensional standard COLD structured record array.
    duration_threshold_days : int, default=192
        Maximum number of days used to evaluate post-break recovery.
    z_value : float, default=2.326
        Multiplier applied to the pre-break model RMSE when constructing the
        magnitude-directed prediction boundary.
    required_bands : int, default=4
        Minimum number of the five PCBF bands that must satisfy the recovery
        condition on the same date.

    Returns
    -------
    numpy.ndarray
        Copy of ``cold_results`` with the same dtype, shape, order and model
        fields. Filtered confirmed breaks receive ``change_prob=0`` while the
        original candidate ``t_break`` is preserved.

    Notes
    -----
    The existing minus-200 category rule is fixed internally and is not a
    configurable PCBF parameter. PCBF does not refit COLD or mutate the input.
    """

    retained = _pcbf_retention_mask(
        cold_results,
        duration_threshold_days=duration_threshold_days,
        z_value=z_value,
        required_bands=required_bands,
    )
    processed = cold_results.copy()
    filtered = (
        ~retained
        & (processed["t_break"] > 0)
        & (processed["change_prob"] == 100)
    )
    processed["change_prob"][filtered] = 0
    return processed
