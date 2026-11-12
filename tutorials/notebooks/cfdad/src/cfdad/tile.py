"""Train an RF and encode one tile-year using the CFDAD production rules."""

import calendar
import json
from pathlib import Path
import pickle
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from sklearn.ensemble import RandomForestClassifier
from skimage import measure
import yaml

from .schema import (
    CHANGE_MAGNITUDE, DATABASE_ATTRIBUTES, FEATURES, LEGACY_FEATURES, resolve_columns,
)

CLASSES = [1, 2, 3, 4]
RF_SETTINGS = {'n_estimators': 150, 'max_depth': 15, 'random_state': 42}
NATIVE_ENCODING = 'agent*10000+doy*10+severity'
TILE_ID = re.compile(r'\d{2}[A-Z]{3}')
BATCH_SIZE = 262144


def read_routes(path):
    with path.open(encoding='utf-8') as stream:
        routes = yaml.safe_load(stream)
    if not isinstance(routes, dict):
        raise ValueError('Expected a mapping from target tile IDs to ordered source IDs')
    return routes


def train(tile, routes, training_dir, output, jobs=1, include_local=True):
    neighbors = read_routes(routes).get(tile)
    if (not TILE_ID.fullmatch(tile) or not isinstance(neighbors, list) or len(neighbors) != 9
            or not all(isinstance(t, str) and TILE_ID.fullmatch(t) for t in neighbors)
            or len(set(neighbors)) != 9):
        raise ValueError(f'Expected nine distinct ordered training source tiles for {tile}')
    sources = list(neighbors)
    local_file = training_dir / f'{tile}_combined.csv'
    if include_local and tile not in sources and local_file.is_file():
        sources.append(tile)
    frames = []
    for source in sources:
        path = training_dir / f'{source}_combined.csv'
        header = pd.read_csv(path, nrows=0).columns
        columns = resolve_columns(header, FEATURES + ['label'], path)
        frame = pd.read_csv(path, usecols=list(columns.values()))
        frame = frame.rename(columns={actual: name for name, actual in columns.items()})
        if frame.empty or not np.isin(frame['label'], CLASSES).all():
            raise ValueError(f'Empty training file or invalid class labels in {path}')
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    x = data[FEATURES]
    y = data['label'].astype(int)
    if sorted(y.unique()) != CLASSES or np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError('Combined training data must contain classes 1–4 and no infinite values')
    model = RandomForestClassifier(**RF_SETTINGS, n_jobs=jobs)
    model.fit(x, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        temporary = Path(folder) / output.name
        with temporary.open('wb') as stream:
            pickle.dump(model, stream)
        temporary.replace(output)
    print(json.dumps({'tile': tile, 'training_sources': sources, 'samples': len(data),
                      'classes': model.classes_.tolist(), 'model': str(output)}))


def pixel_indices(values, limit, name):
    if (values.dtype.kind not in 'iuf' or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values >= limit)
            or np.any(values != np.floor(values))):
        raise ValueError(f'{name} must contain zero-based integer indices in [0, {limit})')
    return values.astype(np.intp, copy=False)


def filter_patches(classes):
    """Keep eight-connected disturbance patches of at least four pixels."""
    disturbed = (classes >= 2) & (classes <= 4)
    regions = measure.label(disturbed, connectivity=2).ravel()
    counts = np.bincount(regions)
    votes = np.zeros((len(counts), 3), dtype=np.int64)
    valid = disturbed.ravel()
    np.add.at(votes, (regions[valid], classes.ravel()[valid] - 2), 1)
    winners = votes.argmax(axis=1) + 2
    keep = counts >= 4
    keep[0] = False
    return np.where(keep[regions], winners[regions], 0).astype(np.int16).reshape(classes.shape)


def predict_pixels(parquet, columns, model, forest, terrain):
    classes = np.zeros(forest.shape, dtype=np.int16)
    required = FEATURES + ['row', 'col']
    input_names = [columns[name] for name in required]
    rename = dict(zip(input_names, required))
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=input_names, use_threads=False):
        frame = batch.to_pandas().rename(columns=rename)
        rows = pixel_indices(frame['row'].to_numpy(), forest.shape[0], 'row')
        cols = pixel_indices(frame['col'].to_numpy(), forest.shape[1], 'col')
        keep = forest[rows, cols]
        if not np.any(keep):
            continue
        features = frame.loc[keep, FEATURES].copy()
        rows, cols = rows[keep], cols[keep]
        for name, values in terrain.items():
            selected = values[rows, cols]
            lower, upper = (-1000, 9000) if name == 'elevation_m' else (0, 90)
            if np.any(np.ma.getmaskarray(selected) | ~np.isfinite(selected.data)
                      | (selected.data < lower) | (selected.data > upper)):
                raise ValueError(f'Invalid {name} at forest candidate pixels')
            features[name] = selected.data
        features.columns = model.feature_names_in_
        classes[rows, cols] = model.predict(features.fillna(-9999))
    return filter_patches(classes)


def encode_native(parquet, columns, classes, year):
    packed = np.zeros(classes.shape, dtype=np.uint16)
    required = DATABASE_ATTRIBUTES + [CHANGE_MAGNITUDE]
    input_names = [columns[name] for name in required]
    rename = dict(zip(input_names, required))
    max_day = 366 if calendar.isleap(year) else 365
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=input_names, use_threads=False):
        frame = batch.to_pandas().rename(columns=rename)
        rows = pixel_indices(frame['row'].to_numpy(), classes.shape[0], 'row')
        cols = pixel_indices(frame['col'].to_numpy(), classes.shape[1], 'col')
        keep = classes[rows, cols] != 0
        if not keep.any():
            continue
        frame = frame.loc[keep]
        rows, cols = rows[keep], cols[keep]
        flat = rows * classes.shape[1] + cols
        if np.any(packed[rows, cols]) or len(np.unique(flat)) != len(flat):
            raise ValueError('More than one database row for a retained disturbance pixel')
        dates = pd.to_datetime(frame['detection_date'].astype(str), format='%Y%m%d', errors='raise', exact=True)
        days = frame['detection_doy'].to_numpy()
        if (not (dates.dt.year == year).all()
                or not np.array_equal(dates.dt.dayofyear.to_numpy(), days)
                or np.any((days < 1) | (days > max_day))):
            raise ValueError('Detection time and DOY must agree with the map year')
        cm = frame[CHANGE_MAGNITUDE].to_numpy()
        if np.any(~np.isfinite(cm) | (cm != np.floor(cm)) | (cm < 0) | (cm > 32767)):
            raise ValueError(f'{CHANGE_MAGNITUDE} must contain nonnegative stored Int16 values')
        severity = 1 + (cm >= 1509).astype(np.uint16) + (cm >= 2052).astype(np.uint16)
        packed[rows, cols] = ((classes[rows, cols] - 1).astype(np.uint16) * 10000
                              + days.astype(np.uint16) * 10 + severity)
    if not np.array_equal(packed != 0, classes != 0):
        raise ValueError('A retained disturbance pixel has no matching database attributes')
    return packed


def predict(tile, year, model_path, database, forest_path, output, jobs=1, dem=None, slope=None):
    if not TILE_ID.fullmatch(tile):
        raise ValueError(f'Invalid tile ID: {tile}')
    # Pickled models must come from a trusted local training run.
    with model_path.open('rb') as stream:
        model = pickle.load(stream)
    if list(model.feature_names_in_) not in (FEATURES, LEGACY_FEATURES) or list(model.classes_) != CLASSES:
        raise ValueError('Model features or classes do not match the CFDAD input contract')
    model.n_jobs = jobs
    with rasterio.open(forest_path) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError('Expected a single-band CLCD raster on the native HLS grid')
        forest = src.read(1) == 2
        profile = src.profile.copy()
    terrain = {}
    for name, path in (('elevation_m', dem), ('slope_deg', slope)):
        if path is None:
            continue
        with rasterio.open(path) as src:
            if (src.crs, src.transform, src.shape, src.count) != (
                    profile['crs'], profile['transform'], forest.shape, 1):
                raise ValueError(f'{name} must match the CLCD raster grid')
            terrain[name] = src.read(1, masked=True)
    with pq.ParquetFile(database) as table:
        columns = resolve_columns(table.schema_arrow.names, FEATURES + DATABASE_ATTRIBUTES, database)
        classes = predict_pixels(table, columns, model, forest, terrain)
        pixels = encode_native(table, columns, classes, year)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        temporary = Path(folder) / output.name
        profile.update(driver='GTiff', dtype='uint16', count=1, nodata=0,
                       compress='lzw', predictor=2, tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(temporary, 'w', **profile) as dst:
            dst.write(pixels, 1)
            dst.update_tags(tile=tile, year=str(year), encoding=NATIVE_ENCODING, layout='ADDDS')
        temporary.replace(output)
    print(json.dumps({'tile': tile, 'year': year, 'disturbed_pixels': int(np.count_nonzero(pixels)),
                      'native_raster': str(output)}))
