"""Command-line interface for the CFDAD workflow."""

import argparse
from pathlib import Path

from .mosaic import mosaic
from .tile import predict, train


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('expected a positive integer')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train', help='fit one target tile RF from ordered source CSVs')
    fit.add_argument('--tile', required=True)
    fit.add_argument('--routes', type=Path, required=True)
    fit.add_argument('--training-dir', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--jobs', type=positive_int, default=1)
    fit.add_argument('--no-local', action='store_true', help='omit a local CSV outside the nine-source list')

    classify = commands.add_parser('predict', help='classify one prepared Parquet and write an ADDDS TIFF')
    classify.add_argument('--tile', required=True)
    classify.add_argument('--year', type=int, required=True)
    classify.add_argument('--model', type=Path, required=True)
    classify.add_argument('--database', type=Path, required=True)
    classify.add_argument('--forest', type=Path, required=True)
    classify.add_argument('--output', type=Path, required=True)
    classify.add_argument('--dem', type=Path, help='optional native-grid DEM override')
    classify.add_argument('--slope', type=Path, help='optional native-grid slope override')
    classify.add_argument('--jobs', type=positive_int, default=1)

    merge = commands.add_parser('mosaic', help='mosaic native ADDDS TIFFs and export final ASDDD')
    merge.add_argument('--year', type=int, required=True)
    merge.add_argument('--tiles', type=Path, required=True, help='ordered JSON manifest with tile, disturbance and forest paths')
    merge.add_argument('--grid', type=Path, required=True, help='north-up EPSG:4326 reference raster')
    merge.add_argument('--output', type=Path, required=True)
    merge.add_argument('--scratch-dir', type=Path, help='temporary disk space for mosaic and patch labels')

    args = parser.parse_args()
    if args.command == 'train':
        train(args.tile, args.routes, args.training_dir, args.output, args.jobs, not args.no_local)
    elif args.command == 'predict':
        predict(args.tile, args.year, args.model, args.database, args.forest,
                args.output, args.jobs, args.dem, args.slope)
    else:
        mosaic(args.year, args.tiles, args.grid, args.output, args.scratch_dir)


if __name__ == '__main__':
    main()
