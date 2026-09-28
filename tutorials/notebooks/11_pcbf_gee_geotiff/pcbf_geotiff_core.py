"""Decode the 392-band GEE CCDC export and apply the original PCBF rules.

The module works directly in the GEE CCDC date and coefficient convention. It
does not require :func:`pyxccd.cold_pcbf` and does not refit the CCDC models.
"""

from __future__ import annotations

import numpy as np


COMMON_BANDS = ("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2")
PCBF_BAND_INDICES = np.asarray((1, 2, 3, 4, 5))
FIXED_CATEGORY_THRESHOLD = 200.0


def expected_band_count(depth: int = 6) -> int:
    """Return the flattened GEE CCDC band count for ``depth`` segments."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    return 5 * depth + 2 + len(COMMON_BANDS) * 10 * depth


def decode_392_window(values: np.ndarray, *, depth: int = 6) -> dict[str, np.ndarray]:
    """Restore segment-major arrays from one flattened raster window."""
    values = np.asarray(values)
    expected = expected_band_count(depth)
    if values.ndim != 3 or values.shape[0] != expected:
        raise ValueError(
            f"values must have shape ({expected}, height, width), got {values.shape}"
        )

    height, width = values.shape[1:]
    spectral_offset = 5 * depth + 2
    coefficient_groups = []
    rmse_groups = []
    magnitude_groups = []
    for band_index in range(len(COMMON_BANDS)):
        start = spectral_offset + band_index * 10 * depth
        group = values[start : start + 10 * depth]
        coefficients = group[: 8 * depth].reshape(depth, 8, height, width)
        coefficient_groups.append(np.transpose(coefficients, (0, 2, 3, 1)))
        rmse_groups.append(group[8 * depth : 9 * depth])
        magnitude_groups.append(group[9 * depth : 10 * depth])

    return {
        "t_start": np.rint(values[0:depth]).astype(np.int64),
        "t_end": np.rint(values[depth : 2 * depth]).astype(np.int64),
        "t_break": np.rint(values[2 * depth : 3 * depth]).astype(np.int64),
        "change_probability": values[3 * depth : 4 * depth].astype(float),
        "num_observations": np.rint(
            values[4 * depth : 5 * depth]
        ).astype(np.int32),
        "coefficients": np.stack(coefficient_groups, axis=3).astype(float),
        "rmse": np.stack(rmse_groups, axis=3).astype(float),
        "magnitude": np.stack(magnitude_groups, axis=3).astype(float),
        "segment_count": np.rint(values[5 * depth]).astype(np.int16),
        "segment_overflow": values[5 * depth + 1] != 0,
    }


def _validate_parameters(
    duration_threshold_days: int,
    z_value: float,
    required_bands: int,
) -> None:
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
        or required_bands > len(PCBF_BAND_INDICES)
    ):
        raise ValueError("required_bands must be between 1 and 5")


def _classify_candidates(
    *,
    previous_t_end: np.ndarray,
    previous_coefficients: np.ndarray,
    previous_rmse: np.ndarray,
    previous_magnitude: np.ndarray,
    following_t_start: np.ndarray,
    following_t_end: np.ndarray,
    following_coefficients: np.ndarray,
    has_following: np.ndarray,
    duration_threshold_days: int,
    z_value: float,
    required_bands: int,
) -> np.ndarray:
    """Return a candidate-aligned Boolean mask of breaks removed by PCBF."""
    t_end = np.asarray(previous_t_end, dtype=np.int64)
    before_coefs = np.asarray(previous_coefficients, dtype=float)
    rmses = np.asarray(previous_rmse, dtype=float)
    magnitudes = np.asarray(previous_magnitude, dtype=float)
    next_start = np.asarray(following_t_start, dtype=np.int64)
    next_end = np.asarray(following_t_end, dtype=np.int64)
    after_coefs = np.asarray(following_coefficients, dtype=float)
    has_next = np.asarray(has_following, dtype=bool)

    n_candidates = t_end.size
    if before_coefs.shape != (n_candidates, 6, 8):
        raise ValueError("previous_coefficients must have shape (n, 6, 8)")
    if after_coefs.shape != (n_candidates, 6, 8):
        raise ValueError("following_coefficients must have shape (n, 6, 8)")
    if rmses.shape != (n_candidates, 6) or magnitudes.shape != (n_candidates, 6):
        raise ValueError("RMSE and magnitude arrays must have shape (n, 6)")

    greener_direction = (
        (magnitudes[:, 3] > -FIXED_CATEGORY_THRESHOLD)
        & (magnitudes[:, 2] < FIXED_CATEGORY_THRESHOLD)
        & (magnitudes[:, 4] < FIXED_CATEGORY_THRESHOLD)
    )
    afforestation = (
        has_next
        & greener_direction
        & (after_coefs[:, 3, 1] > np.abs(before_coefs[:, 3, 1]))
        & (after_coefs[:, 2, 1] < -np.abs(before_coefs[:, 2, 1]))
        & (after_coefs[:, 4, 1] < -np.abs(before_coefs[:, 4, 1]))
    )
    category_two = greener_direction & has_next & ~afforestation

    removed = np.zeros(n_candidates, dtype=bool)
    start = np.maximum(t_end, next_start)
    stop = np.minimum(next_end, t_end + duration_threshold_days - 1)

    before_band_coefs = before_coefs[:, PCBF_BAND_INDICES, :]
    after_band_coefs = after_coefs[:, PCBF_BAND_INDICES, :]
    band_rmse = rmses[:, PCBF_BAND_INDICES]
    band_magnitude = magnitudes[:, PCBF_BAND_INDICES]
    valid_band = (
        np.isfinite(band_rmse)
        & (band_rmse >= 0)
        & np.isfinite(band_magnitude)
    )

    for offset in range(duration_threshold_days):
        jday = t_end + offset
        active = has_next & ~removed & (jday >= start) & (jday <= stop)
        if not np.any(active):
            continue
        active_dates = jday[active].astype(float)
        phase = 2.0 * np.pi * active_dates / 365.25
        basis = np.column_stack(
            (
                np.ones(active_dates.size),
                active_dates,
                np.cos(phase),
                np.sin(phase),
                np.cos(2.0 * phase),
                np.sin(2.0 * phase),
                np.cos(3.0 * phase),
                np.sin(3.0 * phase),
            )
        )
        predicted_before = np.einsum(
            "nbi,ni->nb", before_band_coefs[active], basis
        )
        predicted_after = np.einsum(
            "nbi,ni->nb", after_band_coefs[active], basis
        )
        active_rmse = band_rmse[active]
        active_magnitude = band_magnitude[active]
        passed = np.where(
            active_magnitude < 0,
            predicted_after >= predicted_before - z_value * active_rmse,
            np.where(
                active_magnitude > 0,
                predicted_after <= predicted_before + z_value * active_rmse,
                np.abs(predicted_after - predicted_before)
                <= z_value * active_rmse,
            ),
        )
        counts = np.sum(passed & valid_band[active], axis=1)
        recovered = counts >= required_bands
        if np.any(recovered):
            active_indices = np.flatnonzero(active)
            removed[active_indices[recovered]] = True

    removed |= ~removed & category_two
    return removed


def apply_pcbf_to_392_window(
    values: np.ndarray,
    *,
    duration_threshold_days: int = 192,
    z_value: float = 2.326,
    required_bands: int = 4,
    depth: int = 6,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply PCBF and return a copy with filtered ``changeProb`` values set to 0.

    All dates, coefficients, RMSE values, magnitudes, diagnostics, and spatial
    placement remain unchanged.
    """
    _validate_parameters(duration_threshold_days, z_value, required_bands)
    decoded = decode_392_window(values, depth=depth)
    result = np.asarray(values).copy()

    segment_indices = np.arange(depth)[:, None, None]
    existing = segment_indices < decoded["segment_count"][None, :, :]
    confirmed = decoded["change_probability"] == 1.0
    candidate_mask = (
        existing
        & (decoded["t_break"] > 0)
        & confirmed
        & ~decoded["segment_overflow"][None, :, :]
    )
    segment, row, col = np.nonzero(candidate_mask)

    if segment.size:
        next_segment = segment + 1
        has_following = next_segment < decoded["segment_count"][row, col]
        safe_next = np.minimum(next_segment, depth - 1)
        removed = _classify_candidates(
            previous_t_end=decoded["t_end"][segment, row, col],
            previous_coefficients=decoded["coefficients"][segment, row, col],
            previous_rmse=decoded["rmse"][segment, row, col],
            previous_magnitude=decoded["magnitude"][segment, row, col],
            following_t_start=decoded["t_start"][safe_next, row, col],
            following_t_end=decoded["t_end"][safe_next, row, col],
            following_coefficients=decoded["coefficients"][safe_next, row, col],
            has_following=has_following,
            duration_threshold_days=int(duration_threshold_days),
            z_value=float(z_value),
            required_bands=int(required_bands),
        )
        result[
            3 * depth + segment[removed],
            row[removed],
            col[removed],
        ] = 0
        removed_count = int(np.count_nonzero(removed))
    else:
        removed_count = 0

    confirmed_before = int(segment.size)
    summary = {
        "confirmed_before": confirmed_before,
        "removed": removed_count,
        "retained_after": confirmed_before - removed_count,
        "overflow_pixels": int(np.count_nonzero(decoded["segment_overflow"])),
    }
    return result, summary
