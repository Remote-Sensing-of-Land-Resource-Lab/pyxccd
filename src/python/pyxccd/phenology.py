"""Phenology extraction for COLD and SCCD outputs.

The public API consists of :func:`cold_detect_phenology` and
:func:`sccd_detect_phenology`. 

Both routines use the same public phenology parameters while preserving the
method-specific peak-window logic of the original COLD and SCCD
implementations.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .utils import coefficient_matrix


__all__ = [
    "cold_detect_phenology",
    "sccd_detect_phenology",
]

PHENOLOGY_COLUMNS = [
    "position",
    "t_start",
    "t_end",
    "break_date",
    "fitted_peak_date",
    "fitted_peak",
    "greenup",
    "maturity",
    "senescence",
    "dormancy",
]

# SCCD state dates are Python ordinals. The NRT start date is stored as days
# since the Landsat reference epoch used by pyxccd.
_LANDSAT_REFERENCE_ORDINAL = 723742

# Preserve the original SCCD behavior of ending the active NRT segment six
# valid observations before the final valid state date when possible.
_NRT_TAIL_BUFFER_OBS = 6

def _empty_output() -> pd.DataFrame:
    """Return an empty phenology table with the public output schema."""
    return pd.DataFrame(columns=PHENOLOGY_COLUMNS)


def _field(record, name: str, default=None):
    """Safely extract a field from a dict or structured-array record."""
    if isinstance(record, dict):
        return record.get(name, default)

    try:
        dtype = getattr(record, "dtype", None)
        names = getattr(dtype, "names", None)

        if names is not None and name in names:
            return record[name]

        return record[name]

    except (KeyError, TypeError, ValueError, IndexError):
        return default


def _first_ge(values: np.ndarray, threshold: float) -> int:
    """Return the first finite index at or above a threshold."""
    values = np.asarray(values)

    indices = np.flatnonzero(
        np.isfinite(values)
        & (values >= float(threshold))
    )

    return int(indices[0]) if indices.size else -1


def _first_le(values: np.ndarray, threshold: float) -> int:
    """Return the first finite index at or below a threshold."""
    values = np.asarray(values)

    indices = np.flatnonzero(
        np.isfinite(values)
        & (values <= float(threshold))
    )

    return int(indices[0]) if indices.size else -1


def _first_downcrossing(values: np.ndarray, threshold: float) -> int:
    """Return the first strict-above to at-or-below threshold crossing."""
    values = np.asarray(values)

    if values.size < 2:
        return -1

    finite_pair = (
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
    )

    crossings = np.flatnonzero(
        finite_pair
        & (values[:-1] > float(threshold))
        & (values[1:] <= float(threshold))
    )

    return int(crossings[0] + 1) if crossings.size else -1


def _years_months_from_ordinal(dates_ord: np.ndarray):
    """Convert Python ordinal dates to year and month arrays."""
    d64 = (
        np.datetime64("1970-01-01")
        + (dates_ord.astype(np.int64) - 719163).astype("timedelta64[D]")
    )

    y64 = d64.astype("datetime64[Y]")
    m64 = d64.astype("datetime64[M]")

    years = (y64.astype(int) + 1970).astype(np.int32)
    months = (m64.astype(int) % 12 + 1).astype(np.int16)

    return years, months


def _state_col_or_zeros(
    state_output: pd.DataFrame,
    col: str,
    order: np.ndarray,
    size: int,
) -> np.ndarray:
    """Read a fitted SCCD state component, or return zeros if absent."""
    if col in state_output.columns:
        return np.asarray(
            state_output[col].to_numpy(),
            dtype=np.float32,
        )[order]

    return np.zeros(int(size), dtype=np.float32)


def _select_band_coefficients(coefs, band_index: int):
    """Select one band's COLD coefficient vector.

    The normal pyxccd layout is bands x coefficients. The alternative
    coefficients x bands layout and one-dimensional inputs are retained for
    compatibility with the original phenology implementation.
    """
    array = np.asarray(coefs, dtype=np.float32)
    array = np.squeeze(array)

    if array.ndim == 1:
        return array.ravel()

    if array.ndim != 2:
        return None

    if array.shape[1] == 8 and 0 <= band_index < array.shape[0]:
        return np.asarray(
            array[band_index, :],
            dtype=np.float32,
        ).ravel()

    if array.shape[0] == 8 and 0 <= band_index < array.shape[1]:
        return np.asarray(
            array[:, band_index],
            dtype=np.float32,
        ).ravel()

    if 0 <= band_index < array.shape[0]:
        return np.asarray(
            array[band_index, :],
            dtype=np.float32,
        ).ravel()

    if 0 <= band_index < array.shape[1]:
        return np.asarray(
            array[:, band_index],
            dtype=np.float32,
        ).ravel()

    return None


def _clip_peak_window_to_floor(
    values: np.ndarray,
    left: int,
    peak: int,
    right: int,
    floor_value: float,
):
    """Clip a peak window to the continuous valid interval around the peak."""
    left = int(left)
    peak = int(peak)
    right = int(right)
    floor_value = float(floor_value)

    if not (0 <= left < peak < right < values.size):
        return None

    peak_value = float(values[peak])

    if not np.isfinite(peak_value) or peak_value <= floor_value:
        return None

    left_values = values[left : peak + 1]

    invalid_left = np.flatnonzero(
        (~np.isfinite(left_values))
        | (left_values < floor_value)
    )

    if invalid_left.size > 0:
        left = left + int(invalid_left[-1]) + 1

    right_values = values[peak : right + 1]

    invalid_right = np.flatnonzero(
        (~np.isfinite(right_values))
        | (right_values < floor_value)
    )

    if invalid_right.size > 0:
        right = peak + int(invalid_right[0]) - 1

    if not (left < peak < right):
        return None

    return int(left), int(right)


def _validate_parameters(
    band: int,
    threshold1: float,
    threshold2: float,
    min_peak_gap_days: int,
    peak_threshold: float,
    peak_ratio: float,
    peak_month_filter_mode: str,
):
    """Validate public phenology parameters."""
    if int(band) < 1:
        raise ValueError("band must be a one-based positive integer.")

    if not (0.0 <= float(threshold1) <= 1.0):
        raise ValueError("threshold1 must be between 0 and 1.")

    if not (0.0 <= float(threshold2) <= 1.0):
        raise ValueError("threshold2 must be between 0 and 1.")

    if float(threshold1) > float(threshold2):
        raise ValueError("threshold1 cannot be greater than threshold2.")

    if int(min_peak_gap_days) < 0:
        raise ValueError("min_peak_gap_days cannot be negative.")

    if not np.isfinite(float(peak_threshold)) or float(peak_threshold) <= 0:
        raise ValueError("peak_threshold must be a positive finite number.")

    if not (0.0 <= float(peak_ratio) <= 1.0):
        raise ValueError("peak_ratio must be between 0 and 1.")

    if peak_month_filter_mode not in ("keep", "drop"):
        raise ValueError(
            'peak_month_filter_mode must be either "keep" or "drop".'
        )


def _postprocess_output(
    out_df: pd.DataFrame,
    peak_month_filter: Optional[List[int]],
    peak_month_filter_mode: str,
    min_peak_gap_days: int,
) -> pd.DataFrame:
    """Apply optional month filtering, chronological sorting, and peak spacing."""
    if out_df.empty:
        return out_df.reindex(columns=PHENOLOGY_COLUMNS).fillna(0)

    if peak_month_filter:
        fitted_peak_dates = out_df[
            "fitted_peak_date"
        ].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        months = _years_months_from_ordinal(
            fitted_peak_dates
        )[1]

        month_set = {
            int(month)
            for month in peak_month_filter
            if 1 <= int(month) <= 12
        }

        if month_set:
            month_values = np.fromiter(
                month_set,
                dtype=np.int16,
            )

            in_set = np.isin(
                months,
                month_values,
            )

            if peak_month_filter_mode == "keep":
                keep_month = in_set
            else:
                keep_month = ~in_set

            out_df = out_df.loc[
                keep_month
            ].reset_index(drop=True)

    if out_df.empty:
        return out_df.reindex(columns=PHENOLOGY_COLUMNS).fillna(0)

    out_df = out_df.sort_values(
        "fitted_peak_date",
        kind="mergesort",
    ).reset_index(drop=True)

    fitted_peak_dates = out_df[
        "fitted_peak_date"
    ].to_numpy(
        dtype=np.int64,
        copy=False,
    )

    fitted_peak_values = out_df[
        "fitted_peak"
    ].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    keep = np.zeros(
        out_df.shape[0],
        dtype=bool,
    )

    last_index = -1
    minimum_gap = max(
        0,
        int(min_peak_gap_days),
    )

    for current_index in range(out_df.shape[0]):
        if last_index < 0:
            keep[current_index] = True
            last_index = current_index
            continue

        peak_gap = (
            fitted_peak_dates[current_index]
            - fitted_peak_dates[last_index]
        )

        if peak_gap >= minimum_gap:
            keep[current_index] = True
            last_index = current_index

        elif (
            float(fitted_peak_values[current_index])
            > float(fitted_peak_values[last_index])
        ):
            keep[last_index] = False
            keep[current_index] = True
            last_index = current_index

    return (
        out_df.loc[keep]
        .reset_index(drop=True)
        .reindex(columns=PHENOLOGY_COLUMNS)
        .fillna(0)
    )


def sccd_detect_phenology(
    state_output: pd.DataFrame,
    sccd_result,
    band: int,
    *,
    threshold1: float = 0.15,
    threshold2: float = 0.90,
    min_peak_gap_days: int = 75,
    peak_threshold: float = 800.0,
    peak_month_filter: Optional[List[int]] = None,
    peak_month_filter_mode: str = "drop",
    peak_ratio: float = 0.35,
    state_floor: float = 0.0,
) -> pd.DataFrame:
    """Extract phenology from SCCD fitted state output.

    Parameters
    ----------
    state_output : pandas.DataFrame
        State output returned by ``sccd_detect(..., state_intervaldays > 0)``.
        The table must contain ``dates`` and may contain band-specific fitted
        components such as ``b5_trend``, ``b5_annual``,
        ``b5_semiannual``, and ``b5_trimodal``.
    sccd_result
        SCCD output containing ``rec_cg``, ``position``, ``nrt_mode`` and,
        when applicable, ``nrt_model``.
    band : int
        One-based band index.
    threshold1 : float, default=0.15
        Relative threshold used for greenup and dormancy.
    threshold2 : float, default=0.90
        Relative threshold used for maturity and senescence.
    min_peak_gap_days : int, default=75
        Minimum allowed date gap between retained peaks.
    peak_threshold : float, default=800.0
        Shared absolute peak threshold. It is used both as the scipy peak
        prominence threshold and as the minimum effective phenological peak
        amplitude.
    peak_month_filter : list[int] or None
        Optional months to retain or remove based on fitted peak date.
    peak_month_filter_mode : {"keep", "drop"}, default="drop"
        Determines how ``peak_month_filter`` is applied.
    peak_ratio : float, default=0.35
        Shared relative peak threshold. It is used both for weak-peak
        filtering and for detecting/repairing a high rising-phase boundary.
    state_floor : float, default=0.0
        Values below this fitted-state level cannot form phenological window
        boundaries.

    Returns
    -------
    pandas.DataFrame
        Phenology metrics with columns given by ``PHENOLOGY_COLUMNS``.

        For historical SCCD segments, ``t_end`` and ``break_date`` both equal
        the SCCD ``t_break`` because historical SCCD records do not have an
        independent ``t_end`` field. For the active NRT segment,
        ``break_date`` is zero because its fitted endpoint is not a detected
        structural break.
    """
    _validate_parameters(
        band=band,
        threshold1=threshold1,
        threshold2=threshold2,
        min_peak_gap_days=min_peak_gap_days,
        peak_threshold=peak_threshold,
        peak_ratio=peak_ratio,
        peak_month_filter_mode=peak_month_filter_mode,
    )

    if (
        state_output is None
        or not isinstance(state_output, pd.DataFrame)
        or "dates" not in state_output.columns
        or len(state_output) == 0
    ):
        return _empty_output()

    dates = np.asarray(
        state_output["dates"].to_numpy(),
        dtype=np.int64,
    )

    order = np.argsort(
        dates,
        kind="mergesort",
    )

    dates = dates[order]

    if dates.size == 0:
        return _empty_output()

    band0 = int(band) - 1

    tcol = f"b{band0}_trend"
    acol = f"b{band0}_annual"
    scol = f"b{band0}_semiannual"
    fcol = f"b{band0}_trimodal"

    vals = (
        _state_col_or_zeros(
            state_output,
            tcol,
            order,
            dates.size,
        )
        + _state_col_or_zeros(
            state_output,
            acol,
            order,
            dates.size,
        )
        + _state_col_or_zeros(
            state_output,
            scol,
            order,
            dates.size,
        )
        + _state_col_or_zeros(
            state_output,
            fcol,
            order,
            dates.size,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    position = int(
        getattr(
            sccd_result,
            "position",
            0,
        )
    )

    # Each tuple is:
    # (segment_start, segment_end, real_break_date)
    segments = []
    actual_break_dates = []

    rcg = getattr(
        sccd_result,
        "rec_cg",
        None,
    )

    if rcg is not None:
        for seg in rcg:
            t0_value = _field(
                seg,
                "t_start",
                None,
            )

            t_break_value = _field(
                seg,
                "t_break",
                None,
            )

            if (
                t0_value is None
                or t_break_value is None
            ):
                continue

            try:
                t0 = int(t0_value)
                t_break = int(t_break_value)
            except (TypeError, ValueError):
                continue

            if t_break >= t0:
                segments.append(
                    (
                        int(t0),
                        int(t_break),
                        int(t_break),
                    )
                )

            actual_break_dates.append(
                int(t_break)
            )

    # Add the active SCCD NRT model as the final segment.
    nrt_mode = getattr(
        sccd_result,
        "nrt_mode",
        None,
    )

    if nrt_mode in (1, 3):
        nrt = getattr(
            sccd_result,
            "nrt_model",
            None,
        )

        t_start_since1982 = None

        if nrt is not None:
            try:
                if (
                    hasattr(nrt, "dtype")
                    and getattr(nrt, "dtype").names
                    and "t_start_since1982" in nrt.dtype.names
                ):
                    t_start_since1982 = int(
                        np.atleast_1d(
                            nrt["t_start_since1982"]
                        ).astype(np.int64)[0]
                    )

                elif isinstance(nrt, dict):
                    value = nrt.get(
                        "t_start_since1982",
                        None,
                    )

                    if value is not None:
                        t_start_since1982 = int(value)

            except Exception:
                t_start_since1982 = None

        if t_start_since1982 is not None:
            t_start = (
                int(t_start_since1982)
                + _LANDSAT_REFERENCE_ORDINAL
            )

            if "qa" in state_output.columns:
                qa = np.asarray(
                    state_output["qa"].to_numpy(),
                    dtype=np.int8,
                )[order]

                valid_dates = dates[
                    (qa == 0)
                    | (qa == 1)
                ]

            else:
                valid_dates = dates

            if valid_dates.size >= _NRT_TAIL_BUFFER_OBS:
                t_end = int(
                    valid_dates[-_NRT_TAIL_BUFFER_OBS]
                )

            elif valid_dates.size > 0:
                t_end = int(valid_dates[0])

            else:
                t_end = int(t_start)

            if t_end < t_start:
                t_end = int(t_start)

            segments.append(
                (
                    int(t_start),
                    int(t_end),
                    0,
                )
            )

    if not segments:
        return _empty_output()

    segments = [
        (
            int(t0),
            int(t1),
            int(break_date),
        )
        for t0, t1, break_date in segments
        if int(t1) >= int(t0)
    ]

    if not segments:
        return _empty_output()

    segments.sort(
        key=lambda item: item[0]
    )

    # Sort/deduplicate state dates after component reconstruction.
    dates = np.asarray(
        dates,
        dtype=np.int64,
    )

    vals = np.asarray(
        vals,
        dtype=np.float32,
    )

    _, unique_idx = np.unique(
        dates,
        return_index=True,
    )

    unique_idx = np.sort(
        unique_idx
    )

    dates = dates[unique_idx]
    vals = vals[unique_idx]

    if dates.size == 0:
        return _empty_output()

    actual_break_dates = np.asarray(
        actual_break_dates,
        dtype=np.int64,
    )

    if actual_break_dates.size > 0:
        actual_break_dates = actual_break_dates[
            (actual_break_dates > dates[0])
            & (actual_break_dates < dates[-1])
        ]

        actual_break_dates = np.asarray(
            sorted(
                set(
                    actual_break_dates.tolist()
                )
            ),
            dtype=np.int64,
        )

    n = int(dates.size)
    search_sorted = np.searchsorted

    # SCCD behavior is intentionally preserved: every global peak window is
    # bounded by the nearest sufficiently prominent troughs.
    troughs_all, _ = find_peaks(
        -vals,
        prominence=float(peak_threshold),
    )

    troughs_all = np.asarray(
        troughs_all,
        dtype=np.int32,
    )

    peaks_all, _ = find_peaks(
        vals,
        prominence=float(peak_threshold),
    )

    peaks_all = np.asarray(
        peaks_all,
        dtype=np.int32,
    )

    if peaks_all.size == 0:
        return _empty_output()

    global_windows = []

    for peak in peaks_all:
        peak = int(peak)

        trough_insert = int(
            np.searchsorted(
                troughs_all,
                peak,
                side="left",
            )
        )

        if trough_insert > 0:
            left = int(
                troughs_all[
                    trough_insert - 1
                ]
            )
            edge_head_global = 0
        else:
            left = 0
            edge_head_global = 1

        if trough_insert < troughs_all.size:
            right = int(
                troughs_all[
                    trough_insert
                ]
            )
            edge_tail_global = 0
        else:
            right = n - 1
            edge_tail_global = 1

        if not (left < peak < right):
            continue

        global_windows.append(
            (
                int(left),
                int(right),
                int(peak),
                int(dates[peak]),
                int(edge_head_global),
                int(edge_tail_global),
            )
        )

    if not global_windows:
        return _empty_output()

    output_rows = []

    for (
        t0,
        t1,
        segment_break_date,
    ) in segments:

        left_segment = int(
            search_sorted(
                dates,
                t0,
                side="left",
            )
        )

        right_segment = int(
            search_sorted(
                dates,
                t1,
                side="right",
            )
            - 1
        )

        if left_segment > right_segment:
            continue

        candidate_windows = []

        for (
            global_left,
            global_right,
            peak,
            peak_date,
            edge_head_global,
            edge_tail_global,
        ) in global_windows:

            if not (
                t0
                <= peak_date
                <= t1
            ):
                continue

            raw_left = int(
                global_left
            )

            raw_right = int(
                global_right
            )

            left_clip = max(
                raw_left,
                left_segment,
            )

            right_clip = min(
                raw_right,
                right_segment,
            )

            if not (
                left_clip
                < peak
                < right_clip
            ):
                continue

            # Preserve the original SCCD treatment of a high first rising
            # boundary after a segment break.
            forced_edge_head = False
            found_complete_rising_trough = False

            peak_above_floor = (
                float(vals[peak])
                - float(state_floor)
            )

            if peak_above_floor > 0:
                start_above_floor = (
                    float(vals[left_clip])
                    - float(state_floor)
                )

                start_peak_ratio = (
                    start_above_floor
                    / peak_above_floor
                )

                days_from_segment_start = (
                    int(dates[peak])
                    - int(t0)
                )

                near_segment_head = (
                    days_from_segment_start
                    <= 200
                )

                if (
                    start_peak_ratio
                    > float(peak_ratio)
                ):
                    earliest_date = (
                        int(dates[peak])
                        - 200
                    )

                    lookback_left = max(
                        int(left_segment),
                        int(
                            np.searchsorted(
                                dates,
                                earliest_date,
                                side="left",
                            )
                        ),
                    )

                    previous_troughs = troughs_all[
                        (troughs_all >= lookback_left)
                        & (troughs_all < peak)
                    ].astype(
                        np.int64,
                        copy=False,
                    )

                    if previous_troughs.size > 0:
                        trough_values = vals[
                            previous_troughs
                        ]

                        maximum_start_value = (
                            float(state_floor)
                            + float(peak_ratio)
                            * peak_above_floor
                        )

                        acceptable = (
                            np.isfinite(
                                trough_values
                            )
                            & (
                                trough_values
                                <= maximum_start_value
                            )
                        )

                        acceptable_troughs = (
                            previous_troughs[
                                acceptable
                            ]
                        )

                        if (
                            acceptable_troughs.size
                            > 0
                        ):
                            repaired_left = int(
                                acceptable_troughs[-1]
                            )

                            if repaired_left < peak:
                                left_clip = (
                                    repaired_left
                                )

                                found_complete_rising_trough = (
                                    True
                                )

                    if (
                        near_segment_head
                        and not found_complete_rising_trough
                    ):
                        forced_edge_head = True

            clipped_by_segment_head = (
                raw_left < left_segment
            )

            clipped_by_segment_tail = (
                raw_right > right_segment
            )

            edge_head = int(
                edge_head_global
                or clipped_by_segment_head
                or forced_edge_head
            )

            edge_tail = int(
                edge_tail_global
                or clipped_by_segment_tail
            )

            floor_clipped = _clip_peak_window_to_floor(
                values=vals,
                left=left_clip,
                peak=peak,
                right=right_clip,
                floor_value=state_floor,
            )

            if floor_clipped is None:
                continue

            left_clip, right_clip = (
                floor_clipped
            )

            candidate_windows.append(
                (
                    int(left_clip),
                    int(right_clip),
                    int(peak),
                    int(edge_head),
                    int(edge_tail),
                )
            )

        if not candidate_windows:
            continue

        segment_peak_indices = []
        segment_peak_metadata = []
        segment_amplitudes = []

        for (
            left,
            right,
            peak,
            is_segment_head,
            is_segment_tail,
        ) in candidate_windows:

            peak_value = float(
                vals[peak]
            )

            start_value = float(
                vals[left]
            )

            end_value = float(
                vals[right]
            )

            if not (
                np.isfinite(peak_value)
                and np.isfinite(start_value)
                and np.isfinite(end_value)
            ):
                continue

            rising_amplitude = max(
                0.0,
                peak_value - start_value,
            )

            falling_amplitude = max(
                0.0,
                peak_value - end_value,
            )

            if (
                not is_segment_head
                and not is_segment_tail
            ):
                amplitude = min(
                    rising_amplitude,
                    falling_amplitude,
                )

            elif (
                is_segment_head
                and not is_segment_tail
            ):
                amplitude = (
                    falling_amplitude
                )

            elif (
                is_segment_tail
                and not is_segment_head
            ):
                amplitude = (
                    rising_amplitude
                )

            else:
                amplitude = max(
                    rising_amplitude,
                    falling_amplitude,
                )

            if (
                not np.isfinite(amplitude)
                or amplitude <= 0
            ):
                continue

            greenup_threshold = (
                start_value
                + float(threshold1)
                * rising_amplitude
            )

            maturity_threshold = (
                start_value
                + float(threshold2)
                * rising_amplitude
            )

            senescence_threshold = (
                peak_value
                - (
                    1.0
                    - float(threshold2)
                )
                * falling_amplitude
            )

            dormancy_threshold = (
                peak_value
                - (
                    1.0
                    - float(threshold1)
                )
                * falling_amplitude
            )

            rising_values = vals[
                left : peak + 1
            ]

            falling_values = vals[
                peak : right + 1
            ]

            greenup_relative = _first_ge(
                rising_values,
                greenup_threshold,
            )

            maturity_relative = _first_ge(
                rising_values,
                maturity_threshold,
            )

            # Preserve the SCCD implementation's explicit crossing rule.
            senescence_relative = _first_downcrossing(
                falling_values,
                senescence_threshold,
            )

            dormancy_relative = _first_downcrossing(
                falling_values,
                dormancy_threshold,
            )

            greenup = (
                int(
                    dates[
                        left
                        + greenup_relative
                    ]
                )
                if greenup_relative >= 0
                else 0
            )

            maturity = (
                int(
                    dates[
                        left
                        + maturity_relative
                    ]
                )
                if maturity_relative >= 0
                else 0
            )

            senescence = (
                int(
                    dates[
                        peak
                        + senescence_relative
                    ]
                )
                if senescence_relative >= 0
                else 0
            )

            dormancy = (
                int(
                    dates[
                        peak
                        + dormancy_relative
                    ]
                )
                if dormancy_relative >= 0
                else 0
            )

            if is_segment_head:
                greenup = 0
                maturity = 0

            if is_segment_tail:
                senescence = 0
                dormancy = 0

            if actual_break_dates.size > 0:
                left_date = int(
                    dates[left]
                )

                current_peak_date = int(
                    dates[peak]
                )

                right_date = int(
                    dates[right]
                )

                rising_breaks = actual_break_dates[
                    (
                        actual_break_dates
                        > left_date
                    )
                    & (
                        actual_break_dates
                        < current_peak_date
                    )
                ]

                falling_breaks = actual_break_dates[
                    (
                        actual_break_dates
                        > current_peak_date
                    )
                    & (
                        actual_break_dates
                        < right_date
                    )
                ]

                if rising_breaks.size > 0:
                    first_rising_break = int(
                        np.min(
                            rising_breaks
                        )
                    )

                    if (
                        greenup != 0
                        and greenup
                        >= first_rising_break
                    ):
                        greenup = 0

                    if (
                        maturity != 0
                        and maturity
                        >= first_rising_break
                    ):
                        maturity = 0

                if falling_breaks.size > 0:
                    first_falling_break = int(
                        np.min(
                            falling_breaks
                        )
                    )

                    if (
                        senescence != 0
                        and senescence
                        >= first_falling_break
                    ):
                        senescence = 0

                    if (
                        dormancy != 0
                        and dormancy
                        >= first_falling_break
                    ):
                        dormancy = 0

            segment_peak_indices.append(
                int(peak)
            )

            segment_peak_metadata.append(
                (
                    int(greenup),
                    int(maturity),
                    int(senescence),
                    int(dormancy),
                )
            )

            segment_amplitudes.append(
                float(amplitude)
            )

        if not segment_peak_indices:
            continue

        amplitudes = np.asarray(
            segment_amplitudes,
            dtype=np.float32,
        )

        valid_amplitude = (
            np.isfinite(amplitudes)
            & (amplitudes > 0)
        )

        if not np.any(
            valid_amplitude
        ):
            continue

        reference_amplitude = float(
            np.max(
                amplitudes[
                    valid_amplitude
                ]
            )
        )

        keep_mask = (
            valid_amplitude
            & (
                amplitudes
                >= float(peak_threshold)
            )
            & (
                amplitudes
                >= float(peak_ratio)
                * reference_amplitude
            )
        )

        for (
            retained,
            peak,
            metadata,
        ) in zip(
            keep_mask,
            segment_peak_indices,
            segment_peak_metadata,
        ):
            if not bool(retained):
                continue

            (
                greenup,
                maturity,
                senescence,
                dormancy,
            ) = metadata

            output_rows.append(
                {
                    "position": int(position),
                    "t_start": int(t0),
                    "t_end": int(t1),
                    "break_date": int(
                        segment_break_date
                    ),
                    "fitted_peak_date": int(
                        dates[peak]
                    ),
                    "fitted_peak": float(
                        vals[peak]
                    ),
                    "greenup": int(
                        greenup
                    ),
                    "maturity": int(
                        maturity
                    ),
                    "senescence": int(
                        senescence
                    ),
                    "dormancy": int(
                        dormancy
                    ),
                }
            )

    if not output_rows:
        return _empty_output()

    return _postprocess_output(
        pd.DataFrame(
            output_rows,
            columns=PHENOLOGY_COLUMNS,
        ),
        peak_month_filter=peak_month_filter,
        peak_month_filter_mode=peak_month_filter_mode,
        min_peak_gap_days=min_peak_gap_days,
    )


def cold_detect_phenology(
    cold_rec_cg,
    band: int,
    *,
    threshold1: float = 0.15,
    threshold2: float = 0.90,
    min_peak_gap_days: int = 75,
    peak_threshold: float = 800.0,
    peak_month_filter: Optional[List[int]] = None,
    peak_month_filter_mode: str = "drop",
    peak_ratio: float = 0.35,
    state_floor: float = 0.0,
) -> pd.DataFrame:
    """Extract phenology directly from COLD temporal segments.

    Parameters
    ----------
    cold_rec_cg
        COLD record collection containing ``t_start``, ``t_end`` (or
        ``t_break`` as a fallback), ``t_break``, and ``coefs``.
    band : int
        One-based band index.
    threshold1 : float, default=0.15
        Relative threshold used for greenup and dormancy.
    threshold2 : float, default=0.90
        Relative threshold used for maturity and senescence.
    min_peak_gap_days : int, default=75
        Minimum allowed date gap between retained peaks.
    peak_threshold : float, default=800.0
        Shared absolute peak threshold. It is used both as the scipy peak
        prominence threshold and as the minimum effective phenological peak
        amplitude.
    peak_month_filter : list[int] or None
        Optional months to retain or remove based on fitted peak date.
    peak_month_filter_mode : {"keep", "drop"}, default="drop"
        Determines how ``peak_month_filter`` is applied.
    peak_ratio : float, default=0.35
        Shared relative peak threshold. It is used both for weak-peak
        filtering and for repairing a high rising-phase boundary.
    state_floor : float, default=0.0
        Values below this fitted-state level cannot form phenological window
        boundaries.

    Returns
    -------
    pandas.DataFrame
        Phenology metrics with columns given by ``PHENOLOGY_COLUMNS``.
        ``t_end`` is the fitted COLD segment endpoint and ``break_date`` is
        the COLD structural break date (zero when no real break is recorded).
    """
    _validate_parameters(
        band=band,
        threshold1=threshold1,
        threshold2=threshold2,
        min_peak_gap_days=min_peak_gap_days,
        peak_threshold=peak_threshold,
        peak_ratio=peak_ratio,
        peak_month_filter_mode=peak_month_filter_mode,
    )

    if (
        cold_rec_cg is None
        or len(cold_rec_cg) == 0
    ):
        return _empty_output()

    band0 = int(band) - 1

    segment_records = []
    actual_break_dates = []
    default_position = 0

    for (
        original_seg_id,
        segment,
    ) in enumerate(
        cold_rec_cg
    ):
        t_start_value = _field(
            segment,
            "t_start",
            None,
        )

        t_end_value = _field(
            segment,
            "t_end",
            None,
        )

        t_break_value = _field(
            segment,
            "t_break",
            None,
        )

        if t_end_value is None:
            t_end_value = (
                t_break_value
            )

        coefs_value = _field(
            segment,
            "coefs",
            None,
        )

        if (
            t_start_value is None
            or t_end_value is None
            or coefs_value is None
        ):
            continue

        try:
            t0 = int(
                t_start_value
            )

            t1 = int(
                t_end_value
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if t1 < t0:
            continue

        break_date = 0

        if t_break_value is not None:
            try:
                candidate_break = int(
                    t_break_value
                )

                if candidate_break > 0:
                    break_date = (
                        candidate_break
                    )

                    actual_break_dates.append(
                        candidate_break
                    )

            except (
                TypeError,
                ValueError,
            ):
                pass

        position_value = _field(
            segment,
            "pos",
            None,
        )

        if position_value is None:
            position_value = _field(
                segment,
                "position",
                None,
            )

        if position_value is not None:
            try:
                default_position = int(
                    position_value
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        coefficients = (
            _select_band_coefficients(
                coefs=coefs_value,
                band_index=band0,
            )
        )

        if (
            coefficients is None
            or coefficients.size == 0
        ):
            continue

        coefficients = np.asarray(
            coefficients,
            dtype=np.float32,
        ).ravel()

        ncoef = min(
            int(
                coefficients.size
            ),
            8,
        )

        if ncoef <= 0:
            continue

        segment_dates = np.arange(
            t0,
            t1 + 1,
            dtype=np.int64,
        )

        if segment_dates.size == 0:
            continue

        segment_values = np.full(
            segment_dates.size,
            np.nan,
            dtype=np.float32,
        )

        for (
            index,
            current_date,
        ) in enumerate(
            segment_dates
        ):
            design_vector = np.asarray(
                coefficient_matrix(
                    int(current_date),
                    int(ncoef),
                ),
                dtype=np.float32,
            ).ravel()

            usable_count = min(
                int(ncoef),
                int(
                    design_vector.size
                ),
                int(
                    coefficients.size
                ),
            )

            if usable_count <= 0:
                continue

            segment_values[
                index
            ] = float(
                np.dot(
                    design_vector[
                        :usable_count
                    ],
                    coefficients[
                        :usable_count
                    ],
                )
            )

        if not np.any(
            np.isfinite(
                segment_values
            )
        ):
            continue

        segment_records.append(
            {
                "original_seg_id": int(
                    original_seg_id
                ),
                "t_start": int(t0),
                "t_end": int(t1),
                "break_date": int(
                    break_date
                ),
                "position": int(
                    default_position
                ),
                "dates": (
                    segment_dates
                ),
                "values": (
                    segment_values
                ),
            }
        )

    if not segment_records:
        return _empty_output()

    segment_records.sort(
        key=lambda record: (
            record["t_start"],
            record["t_end"],
            record[
                "original_seg_id"
            ],
        )
    )

    all_dates = []
    all_values = []
    all_segment_ranks = []

    for (
        segment_rank,
        record,
    ) in enumerate(
        segment_records
    ):
        segment_dates = (
            record["dates"]
        )

        segment_values = (
            record["values"]
        )

        all_dates.append(
            segment_dates
        )

        all_values.append(
            segment_values
        )

        all_segment_ranks.append(
            np.full(
                segment_dates.size,
                int(segment_rank),
                dtype=np.int32,
            )
        )

    dates = np.concatenate(
        all_dates
    ).astype(
        np.int64,
        copy=False,
    )

    vals = np.concatenate(
        all_values
    ).astype(
        np.float32,
        copy=False,
    )

    segment_ranks = np.concatenate(
        all_segment_ranks
    ).astype(
        np.int32,
        copy=False,
    )

    sort_index = np.lexsort(
        (
            segment_ranks,
            dates,
        )
    )

    dates = dates[
        sort_index
    ]

    vals = vals[
        sort_index
    ]

    segment_ranks = (
        segment_ranks[
            sort_index
        ]
    )

    if dates.size > 1:
        keep_last_duplicate = np.r_[
            dates[1:]
            != dates[:-1],
            True,
        ]

        dates = dates[
            keep_last_duplicate
        ]

        vals = vals[
            keep_last_duplicate
        ]

        segment_ranks = (
            segment_ranks[
                keep_last_duplicate
            ]
        )

    if dates.size == 0:
        return _empty_output()

    actual_break_dates = np.asarray(
        actual_break_dates,
        dtype=np.int64,
    )

    if actual_break_dates.size > 0:
        actual_break_dates = (
            actual_break_dates[
                (
                    actual_break_dates
                    > dates[0]
                )
                & (
                    actual_break_dates
                    < dates[-1]
                )
            ]
        )

        actual_break_dates = np.asarray(
            sorted(
                set(
                    actual_break_dates.tolist()
                )
            ),
            dtype=np.int64,
        )

    n = int(
        dates.size
    )

    search_sorted = (
        np.searchsorted
    )

    troughs_all, _ = find_peaks(
        -vals,
        prominence=float(
            peak_threshold
        ),
    )

    troughs_all = np.asarray(
        troughs_all,
        dtype=np.int32,
    )

    peaks_all, _ = find_peaks(
        vals,
        prominence=float(
            peak_threshold
        ),
    )

    peaks_all = np.asarray(
        peaks_all,
        dtype=np.int32,
    )

    if peaks_all.size == 0:
        return _empty_output()

    # Preserve the original COLD behavior: minima between adjacent detected
    # peaks form shared cycle boundaries even when those minima do not meet
    # the trough prominence threshold.
    global_windows = []

    peaks_sorted = np.asarray(
        np.sort(
            peaks_all
        ),
        dtype=np.int32,
    )

    for (
        peak_order,
        peak_index,
    ) in enumerate(
        peaks_sorted
    ):
        peak_index = int(
            peak_index
        )

        if peak_order > 0:
            previous_peak = int(
                peaks_sorted[
                    peak_order - 1
                ]
            )

            valley_start = (
                previous_peak + 1
            )

            valley_stop = (
                peak_index
            )

            valley_values = vals[
                valley_start:
                valley_stop
            ]

            finite_mask = (
                np.isfinite(
                    valley_values
                )
            )

            if (
                valley_values.size > 0
                and np.any(
                    finite_mask
                )
            ):
                valley_search = (
                    np.where(
                        finite_mask,
                        valley_values,
                        np.inf,
                    )
                )

                left_index = (
                    valley_start
                    + int(
                        np.argmin(
                            valley_search
                        )
                    )
                )

            else:
                left_index = (
                    previous_peak
                    + 1
                )

            left_is_interpeak = 1
            edge_head_global = 0

        else:
            preceding_troughs = (
                troughs_all[
                    troughs_all
                    < peak_index
                ]
            )

            if (
                preceding_troughs.size
                > 0
            ):
                left_index = int(
                    preceding_troughs[
                        -1
                    ]
                )

                edge_head_global = 0

            else:
                left_index = 0
                edge_head_global = 1

            left_is_interpeak = 0

        if (
            peak_order
            < peaks_sorted.size - 1
        ):
            next_peak = int(
                peaks_sorted[
                    peak_order + 1
                ]
            )

            valley_start = (
                peak_index + 1
            )

            valley_stop = (
                next_peak
            )

            valley_values = vals[
                valley_start:
                valley_stop
            ]

            finite_mask = (
                np.isfinite(
                    valley_values
                )
            )

            if (
                valley_values.size > 0
                and np.any(
                    finite_mask
                )
            ):
                valley_search = (
                    np.where(
                        finite_mask,
                        valley_values,
                        np.inf,
                    )
                )

                right_index = (
                    valley_start
                    + int(
                        np.argmin(
                            valley_search
                        )
                    )
                )

            else:
                right_index = (
                    next_peak - 1
                )

            right_is_interpeak = 1
            edge_tail_global = 0

        else:
            following_troughs = (
                troughs_all[
                    troughs_all
                    > peak_index
                ]
            )

            if (
                following_troughs.size
                > 0
            ):
                right_index = int(
                    following_troughs[
                        0
                    ]
                )

                edge_tail_global = 0

            else:
                right_index = (
                    n - 1
                )

                edge_tail_global = 1

            right_is_interpeak = 0

        if not (
            left_index
            < peak_index
            < right_index
        ):
            continue

        global_windows.append(
            (
                int(
                    left_index
                ),
                int(
                    right_index
                ),
                int(
                    peak_index
                ),
                int(
                    dates[
                        peak_index
                    ]
                ),
                int(
                    edge_head_global
                ),
                int(
                    edge_tail_global
                ),
                int(
                    left_is_interpeak
                ),
                int(
                    right_is_interpeak
                ),
            )
        )

    if not global_windows:
        return _empty_output()

    output_rows = []

    for record in segment_records:
        t0 = int(
            record["t_start"]
        )

        t1 = int(
            record["t_end"]
        )

        segment_break_date = int(
            record["break_date"]
        )

        segment_position = int(
            record["position"]
        )

        left_segment = int(
            search_sorted(
                dates,
                t0,
                side="left",
            )
        )

        right_segment = int(
            search_sorted(
                dates,
                t1,
                side="right",
            )
            - 1
        )

        if (
            left_segment
            > right_segment
        ):
            continue

        candidate_windows = []

        for (
            global_left,
            global_right,
            peak,
            peak_date,
            edge_head_global,
            edge_tail_global,
            left_is_interpeak,
            right_is_interpeak,
        ) in global_windows:

            if not (
                t0
                <= peak_date
                <= t1
            ):
                continue

            raw_left = int(
                global_left
            )

            raw_right = int(
                global_right
            )

            left_clip = max(
                raw_left,
                left_segment,
            )

            right_clip = min(
                raw_right,
                right_segment,
            )

            if not (
                left_clip
                < peak
                < right_clip
            ):
                continue

            peak_above_floor = (
                float(vals[peak])
                - float(state_floor)
            )

            if (
                not bool(
                    left_is_interpeak
                )
                and peak_above_floor > 0
            ):
                start_above_floor = (
                    float(
                        vals[
                            left_clip
                        ]
                    )
                    - float(
                        state_floor
                    )
                )

                start_peak_ratio = (
                    start_above_floor
                    / peak_above_floor
                )

                if (
                    np.isfinite(
                        start_peak_ratio
                    )
                    and start_peak_ratio
                    > float(
                        peak_ratio
                    )
                ):
                    earliest_date = (
                        int(
                            dates[
                                peak
                            ]
                        )
                        - 200
                    )

                    lookback_left = max(
                        int(
                            left_segment
                        ),
                        int(
                            np.searchsorted(
                                dates,
                                earliest_date,
                                side="left",
                            )
                        ),
                    )

                    previous_troughs = troughs_all[
                        (
                            troughs_all
                            >= lookback_left
                        )
                        & (
                            troughs_all
                            < peak
                        )
                    ].astype(
                        np.int64,
                        copy=False,
                    )

                    if (
                        previous_troughs.size
                        > 0
                    ):
                        trough_values = vals[
                            previous_troughs
                        ]

                        maximum_start_value = (
                            float(
                                state_floor
                            )
                            + float(
                                peak_ratio
                            )
                            * peak_above_floor
                        )

                        acceptable = (
                            np.isfinite(
                                trough_values
                            )
                            & (
                                trough_values
                                <= maximum_start_value
                            )
                        )

                        acceptable_troughs = (
                            previous_troughs[
                                acceptable
                            ]
                        )

                        if (
                            acceptable_troughs.size
                            > 0
                        ):
                            repaired_left = int(
                                acceptable_troughs[
                                    -1
                                ]
                            )

                            if (
                                repaired_left
                                < peak
                            ):
                                left_clip = (
                                    repaired_left
                                )

            clipped_by_segment_head = (
                raw_left
                < left_segment
            )

            clipped_by_segment_tail = (
                raw_right
                > right_segment
            )

            edge_head = int(
                edge_head_global
                or clipped_by_segment_head
            )

            edge_tail = int(
                edge_tail_global
                or clipped_by_segment_tail
            )

            floor_clipped = (
                _clip_peak_window_to_floor(
                    values=vals,
                    left=left_clip,
                    peak=peak,
                    right=right_clip,
                    floor_value=state_floor,
                )
            )

            if (
                floor_clipped
                is None
            ):
                continue

            (
                left_clip,
                right_clip,
            ) = floor_clipped

            candidate_windows.append(
                (
                    int(
                        left_clip
                    ),
                    int(
                        right_clip
                    ),
                    int(peak),
                    int(
                        edge_head
                    ),
                    int(
                        edge_tail
                    ),
                )
            )

        if not candidate_windows:
            continue

        segment_peak_indices = []
        segment_peak_metadata = []
        segment_amplitudes = []

        for (
            left,
            right,
            peak,
            is_segment_head,
            is_segment_tail,
        ) in candidate_windows:

            peak_value = float(
                vals[peak]
            )

            start_value = float(
                vals[left]
            )

            end_value = float(
                vals[right]
            )

            if not (
                np.isfinite(
                    peak_value
                )
                and np.isfinite(
                    start_value
                )
                and np.isfinite(
                    end_value
                )
            ):
                continue

            rising_amplitude = max(
                0.0,
                peak_value
                - start_value,
            )

            falling_amplitude = max(
                0.0,
                peak_value
                - end_value,
            )

            if (
                not is_segment_head
                and not is_segment_tail
            ):
                amplitude = min(
                    rising_amplitude,
                    falling_amplitude,
                )

            elif (
                is_segment_head
                and not is_segment_tail
            ):
                amplitude = (
                    falling_amplitude
                )

            elif (
                is_segment_tail
                and not is_segment_head
            ):
                amplitude = (
                    rising_amplitude
                )

            else:
                amplitude = max(
                    rising_amplitude,
                    falling_amplitude,
                )

            if (
                not np.isfinite(
                    amplitude
                )
                or amplitude <= 0
            ):
                continue

            greenup_threshold = (
                start_value
                + float(
                    threshold1
                )
                * rising_amplitude
            )

            maturity_threshold = (
                start_value
                + float(
                    threshold2
                )
                * rising_amplitude
            )

            senescence_threshold = (
                peak_value
                - (
                    1.0
                    - float(
                        threshold2
                    )
                )
                * falling_amplitude
            )

            dormancy_threshold = (
                peak_value
                - (
                    1.0
                    - float(
                        threshold1
                    )
                )
                * falling_amplitude
            )

            rising_values = vals[
                left : peak + 1
            ]

            falling_values = vals[
                peak : right + 1
            ]

            greenup_relative = (
                _first_ge(
                    rising_values,
                    greenup_threshold,
                )
            )

            maturity_relative = (
                _first_ge(
                    rising_values,
                    maturity_threshold,
                )
            )

            # Preserve the COLD implementation's first-at-or-below rule.
            senescence_relative = (
                _first_le(
                    falling_values,
                    senescence_threshold,
                )
            )

            dormancy_relative = (
                _first_le(
                    falling_values,
                    dormancy_threshold,
                )
            )

            greenup = (
                int(
                    dates[
                        left
                        + greenup_relative
                    ]
                )
                if (
                    greenup_relative
                    >= 0
                )
                else 0
            )

            maturity = (
                int(
                    dates[
                        left
                        + maturity_relative
                    ]
                )
                if (
                    maturity_relative
                    >= 0
                )
                else 0
            )

            senescence = (
                int(
                    dates[
                        peak
                        + senescence_relative
                    ]
                )
                if (
                    senescence_relative
                    >= 0
                )
                else 0
            )

            dormancy = (
                int(
                    dates[
                        peak
                        + dormancy_relative
                    ]
                )
                if (
                    dormancy_relative
                    >= 0
                )
                else 0
            )

            if is_segment_head:
                greenup = 0
                maturity = 0

            if is_segment_tail:
                senescence = 0
                dormancy = 0

            if (
                actual_break_dates.size
                > 0
            ):
                left_date = int(
                    dates[left]
                )

                current_peak_date = int(
                    dates[peak]
                )

                right_date = int(
                    dates[right]
                )

                rising_breaks = (
                    actual_break_dates[
                        (
                            actual_break_dates
                            > left_date
                        )
                        & (
                            actual_break_dates
                            < current_peak_date
                        )
                    ]
                )

                falling_breaks = (
                    actual_break_dates[
                        (
                            actual_break_dates
                            > current_peak_date
                        )
                        & (
                            actual_break_dates
                            < right_date
                        )
                    ]
                )

                if (
                    rising_breaks.size
                    > 0
                ):
                    first_rising_break = int(
                        np.min(
                            rising_breaks
                        )
                    )

                    if (
                        greenup != 0
                        and greenup
                        >= first_rising_break
                    ):
                        greenup = 0

                    if (
                        maturity != 0
                        and maturity
                        >= first_rising_break
                    ):
                        maturity = 0

                if (
                    falling_breaks.size
                    > 0
                ):
                    first_falling_break = int(
                        np.min(
                            falling_breaks
                        )
                    )

                    if (
                        senescence != 0
                        and senescence
                        >= first_falling_break
                    ):
                        senescence = 0

                    if (
                        dormancy != 0
                        and dormancy
                        >= first_falling_break
                    ):
                        dormancy = 0

            segment_peak_indices.append(
                int(peak)
            )

            segment_peak_metadata.append(
                (
                    int(
                        greenup
                    ),
                    int(
                        maturity
                    ),
                    int(
                        senescence
                    ),
                    int(
                        dormancy
                    ),
                )
            )

            segment_amplitudes.append(
                float(
                    amplitude
                )
            )

        if not segment_peak_indices:
            continue

        amplitudes = np.asarray(
            segment_amplitudes,
            dtype=np.float32,
        )

        valid_amplitude = (
            np.isfinite(
                amplitudes
            )
            & (
                amplitudes
                > 0
            )
        )

        if not np.any(
            valid_amplitude
        ):
            continue

        reference_amplitude = float(
            np.max(
                amplitudes[
                    valid_amplitude
                ]
            )
        )

        keep_mask = (
            valid_amplitude
            & (
                amplitudes
                >= float(
                    peak_threshold
                )
            )
            & (
                amplitudes
                >= float(
                    peak_ratio
                )
                * reference_amplitude
            )
        )

        for (
            retained,
            peak,
            metadata,
        ) in zip(
            keep_mask,
            segment_peak_indices,
            segment_peak_metadata,
        ):
            if not bool(
                retained
            ):
                continue

            (
                greenup,
                maturity,
                senescence,
                dormancy,
            ) = metadata

            output_rows.append(
                {
                    "position": int(
                        segment_position
                    ),
                    "t_start": int(
                        t0
                    ),
                    "t_end": int(
                        t1
                    ),
                    "break_date": int(
                        segment_break_date
                    ),
                    "fitted_peak_date": int(
                        dates[
                            peak
                        ]
                    ),
                    "fitted_peak": float(
                        vals[
                            peak
                        ]
                    ),
                    "greenup": int(
                        greenup
                    ),
                    "maturity": int(
                        maturity
                    ),
                    "senescence": int(
                        senescence
                    ),
                    "dormancy": int(
                        dormancy
                    ),
                }
            )

    if not output_rows:
        return _empty_output()

    return _postprocess_output(
        pd.DataFrame(
            output_rows,
            columns=PHENOLOGY_COLUMNS,
        ),
        peak_month_filter=peak_month_filter,
        peak_month_filter_mode=peak_month_filter_mode,
        min_peak_gap_days=min_peak_gap_days,
    )
