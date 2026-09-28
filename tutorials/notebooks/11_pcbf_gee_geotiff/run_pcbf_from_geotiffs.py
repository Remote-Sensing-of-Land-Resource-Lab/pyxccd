#!/usr/bin/env python3
"""Apply PCBF to spatial shards from the 392-band GEE CCDC export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import rasterio
from rasterio.windows import Window

from pcbf_geotiff_core import apply_pcbf_to_392_window, expected_band_count


SUMMARY_FIELDS = (
    "input_file",
    "output_file",
    "confirmed_before",
    "removed",
    "retained_after",
    "overflow_pixels",
)


def _windows(width: int, height: int, size: int):
    if size <= 0:
        raise ValueError("window_size must be positive")
    for row_off in range(0, height, size):
        for col_off in range(0, width, size):
            yield Window(
                col_off,
                row_off,
                min(size, width - col_off),
                min(size, height - row_off),
            )


def _discover_shards(
    input_dir: Path,
    pattern: str,
    expected_shards: int,
) -> list[Path]:
    paths = sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file()
        and not path.name.startswith(".")
        and not path.stem.endswith("_pcbf")
    )
    if expected_shards > 0 and len(paths) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} input GeoTIFF shards, found {len(paths)}"
        )
    if not paths:
        raise FileNotFoundError(f"no input GeoTIFFs matched {input_dir / pattern}")
    return paths


def _validate_shards(paths: list[Path], *, depth: int) -> None:
    expected_count = expected_band_count(depth)
    signatures = []
    bounds = []
    for path in paths:
        with rasterio.open(path) as dataset:
            if dataset.count != expected_count:
                raise ValueError(
                    f"{path.name} has {dataset.count} bands; expected {expected_count}"
                )
            signatures.append(
                (
                    str(dataset.crs),
                    float(dataset.transform.a),
                    float(dataset.transform.b),
                    float(dataset.transform.d),
                    float(dataset.transform.e),
                )
            )
            bounds.append(dataset.bounds)
    if len(set(signatures)) != 1:
        raise ValueError("input shards do not share one CRS, resolution, and orientation")

    keys = [(round(item.left, 6), round(item.top, 6)) for item in bounds]
    if len(keys) != len(set(keys)):
        raise ValueError("input shards contain duplicate spatial bounds")
    for first_index, first in enumerate(bounds):
        for second in bounds[first_index + 1 :]:
            overlap_width = max(
                0.0,
                min(first.right, second.right) - max(first.left, second.left),
            )
            overlap_height = max(
                0.0,
                min(first.top, second.top) - max(first.bottom, second.bottom),
            )
            if overlap_width * overlap_height > 0:
                raise ValueError("input shard bounds overlap")


def _process_shard(
    source_path: Path,
    output_path: Path,
    *,
    depth: int,
    window_size: int,
    duration_threshold_days: int,
    z_value: float,
    required_bands: int,
    overwrite: bool,
) -> dict[str, int | str]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}")
    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    totals = {
        "confirmed_before": 0,
        "removed": 0,
        "retained_after": 0,
        "overflow_pixels": 0,
    }
    try:
        with rasterio.open(source_path) as source:
            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                BIGTIFF="IF_SAFER",
                compress="deflate",
            )
            with rasterio.open(partial_path, "w", **profile) as destination:
                destination.update_tags(**source.tags())
                destination.descriptions = source.descriptions
                for band_index in range(1, source.count + 1):
                    tags = source.tags(band_index)
                    if tags:
                        destination.update_tags(band_index, **tags)

                for window in _windows(source.width, source.height, window_size):
                    values = source.read(window=window)
                    processed, summary = apply_pcbf_to_392_window(
                        values,
                        duration_threshold_days=duration_threshold_days,
                        z_value=z_value,
                        required_bands=required_bands,
                        depth=depth,
                    )
                    destination.write(processed, window=window)
                    destination.write_mask(source.dataset_mask(window=window), window=window)
                    for name in totals:
                        totals[name] += summary[name]
        partial_path.replace(output_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    return {
        "input_file": source_path.name,
        "output_file": output_path.name,
        **totals,
    }


def run(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.tif",
    expected_shards: int = 4,
    depth: int = 6,
    window_size: int = 128,
    duration_threshold_days: int = 192,
    z_value: float = 2.326,
    required_bands: int = 4,
    overwrite: bool = False,
) -> dict[str, int]:
    """Process all spatial shards and return the combined count summary."""
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _discover_shards(input_dir, pattern, expected_shards)
    _validate_shards(paths, depth=depth)
    rows = []
    for source_path in paths:
        output_path = output_dir / f"{source_path.stem}_pcbf.tif"
        rows.append(
            _process_shard(
                source_path,
                output_path,
                depth=depth,
                window_size=window_size,
                duration_threshold_days=duration_threshold_days,
                z_value=z_value,
                required_bands=required_bands,
                overwrite=overwrite,
            )
        )

    totals = {
        name: sum(int(row[name]) for row in rows)
        for name in (
            "confirmed_before",
            "removed",
            "retained_after",
            "overflow_pixels",
        )
    }
    rows.append(
        {
            "input_file": "TOTAL",
            "output_file": "",
            **totals,
        }
    )
    summary_path = output_dir / "pcbf_summary.csv"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return totals


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply PCBF to four 392-band GEE CCDC GeoTIFF shards."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--expected-shards", default=4, type=int)
    parser.add_argument("--window-size", default=128, type=int)
    parser.add_argument("--duration-threshold-days", default=192, type=int)
    parser.add_argument("--z-value", default=2.326, type=float)
    parser.add_argument("--required-bands", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    totals = run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        expected_shards=args.expected_shards,
        window_size=args.window_size,
        duration_threshold_days=args.duration_threshold_days,
        z_value=args.z_value,
        required_bands=args.required_bands,
        overwrite=args.overwrite,
    )
    print(f"Confirmed CCDC breaks: {totals['confirmed_before']}")
    print(f"Breaks removed by PCBF: {totals['removed']}")
    print(f"Breaks retained by PCBF: {totals['retained_after']}")
    print(f"Overflow pixels excluded: {totals['overflow_pixels']}")


if __name__ == "__main__":
    main()
