"""Make a geographic ASDDD mosaic from ordered native tile predictions."""

import calendar
from collections import defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform, transform_bounds
from rasterio.windows import Window
from scipy.ndimage import label

from .tile import NATIVE_ENCODING, TILE_ID


WGS84 = rasterio.crs.CRS.from_epsg(4326)
BLOCK = 1024
NODATA = 65535


def windows(width, height):
    for row in range(0, height, BLOCK):
        for col in range(0, width, BLOCK):
            yield Window(col, row, min(BLOCK, width - col), min(BLOCK, height - row))


def slices(window):
    col, row, width, height = map(int, window.flatten())
    return slice(row, row + height), slice(col, col + width)


def read_sources(manifest):
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('Expected a nonempty, ordered list of tile entries')
    seen = set()
    for entry in entries:
        tile = entry['tile']
        if not TILE_ID.fullmatch(tile) or tile in seen:
            raise ValueError(f'Invalid or repeated source tile: {tile}')
        seen.add(tile)
        yield tile, manifest.parent / entry['disturbance'], manifest.parent / entry['forest']


def build_index(manifest, profile):
    index = defaultdict(list)
    grid = profile['transform']
    width, height = profile['width'], profile['height']
    for tile, raster, mask in read_sources(manifest):
        with rasterio.open(raster) as src, rasterio.open(mask) as forest:
            if (src.count != 1 or src.crs is None or src.dtypes != ('uint16',) or src.nodata != 0
                    or src.tags().get('encoding') != NATIVE_ENCODING):
                raise ValueError(f'{tile}: expected a native ADDDS disturbance TIFF with NoData 0')
            if (src.crs, src.transform, src.shape, src.count) != (forest.crs, forest.transform, forest.shape, forest.count):
                raise ValueError(f'{tile}: disturbance and forest rasters must share a grid')
            x, y = src.transform * (src.width / 2, src.height / 2)
            lon, lat = transform(src.crs, WGS84, [x], [y])
            center = (math.radians(lon[0]), math.radians(lat[0]))
            west, south, east, north = transform_bounds(src.crs, WGS84, *src.bounds, densify_pts=21)
        c0, r0 = ~grid * (west, north)
        c1, r1 = ~grid * (east, south)
        c0, r0 = max(0, math.floor(c0)), max(0, math.floor(r0))
        c1, r1 = min(width, math.ceil(c1)), min(height, math.ceil(r1))
        if c1 <= c0 or r1 <= r0:
            continue
        for row in range(r0 // BLOCK, (r1 - 1) // BLOCK + 1):
            for col in range(c0 // BLOCK, (c1 - 1) // BLOCK + 1):
                index[row, col].append((center, raster, mask))
    return index


def open_sources(stack, entries, grid):
    warp = dict(crs=WGS84, resampling=Resampling.nearest, nodata=0,
                **{key: grid[key] for key in ('transform', 'width', 'height')})
    for center, raster, mask in entries:
        src = stack.enter_context(rasterio.open(raster))
        forest = stack.enter_context(rasterio.open(mask))
        values = stack.enter_context(WarpedVRT(src, **warp, src_nodata=0, dtype='uint16'))
        forests = stack.enter_context(WarpedVRT(forest, **warp))
        yield center, values, forests


def combine_window(entries, window, grid):
    col, row, width, height = map(int, window.flatten())
    shape = (height, width)
    selected = np.zeros(shape, dtype=np.uint16)
    forest = np.zeros(shape, dtype=bool)
    best_distance = np.full(shape, np.inf)
    lon = np.deg2rad(grid.c + (col + np.arange(width) + 0.5) * grid.a)
    lat = np.deg2rad(grid.f + (row + np.arange(height) + 0.5) * grid.e)
    for (center_lon, center_lat), source, mask in entries:
        forest |= mask.read(1, window=window) == 2
        values = source.read(1, window=window)
        rr, cc = np.nonzero(values)
        # The haversine score preserves the ordering of great-circle distances.
        distance = (np.sin((lat[rr] - center_lat) / 2)**2 + np.cos(lat[rr])
                    * math.cos(center_lat) * np.sin((lon[cc] - center_lon) / 2)**2)
        use = distance < best_distance[rr, cc]
        r, c = rr[use], cc[use]
        selected[r, c] = values[r, c]
        best_distance[r, c] = distance[use]
    return selected, forest


def raster_profile(grid, dtype, nodata, compression):
    return dict(driver='GTiff', count=1, dtype=dtype, nodata=nodata, compress=compression,
                tiled=True, blockxsize=256, blockysize=256, SPARSE_OK='TRUE', BIGTIFF='IF_SAFER',
                **{key: grid[key] for key in ('width', 'height', 'transform', 'crs')})


def prepare_mosaic(grid, manifest, year, scratch):
    shape = grid['height'], grid['width']
    disturbed = np.memmap(scratch / 'disturbed.bin', dtype='bool', mode='w+', shape=shape)
    raw_path, forest_path = scratch / 'selected.tif', scratch / 'forest.tif'
    maximum_day = 366 if calendar.isleap(year) else 365
    entries = build_index(manifest, grid)
    with rasterio.open(raw_path, 'w', **raster_profile(grid, 'uint16', 0, 'lzw')) as raw, \
            rasterio.open(forest_path, 'w', **raster_profile(grid, 'uint8', 0, 'lzw')) as forests:
        for window in windows(raw.width, raw.height):
            candidates = entries.get((int(window.row_off) // BLOCK, int(window.col_off) // BLOCK), [])
            if not candidates:
                continue
            with ExitStack() as stack:
                selected, forest = combine_window(open_sources(stack, candidates, grid), window, grid['transform'])
            live = selected != 0
            if np.any(live & ~forest):
                raise ValueError('A selected disturbance is outside the union of the forest masks')
            if live.any():
                values = selected[live]
                days = values % 10000 // 10
                if (not np.isin(values // 10000, [1, 2, 3]).all()
                        or not np.isin(values % 10, [1, 2, 3]).all()
                        or np.any((days < 1) | (days > maximum_day))):
                    raise ValueError('Invalid ADDDS values in the selected disturbance raster')
                raw.write(selected, 1, window=window)
                disturbed[slices(window)] = live
            if forest.any():
                forests.write(forest.astype(np.uint8), 1, window=window)
    disturbed.flush()
    return disturbed, raw_path, forest_path


def vote_patch_agents(raw, identifiers, count):
    votes = np.zeros((count + 1, 3), dtype=np.int64)
    for window in windows(raw.width, raw.height):
        labels = identifiers[slices(window)]
        live = labels != 0
        if live.any():
            agent = raw.read(1, window=window)[live] // 10000
            np.add.at(votes, (labels[live], agent - 1), 1)
    return (votes.argmax(axis=1) + 1).astype(np.uint16)


def write_output(path, grid, year, raw, forests, identifiers, agents):
    total = 0
    profile = raster_profile(grid, 'uint16', NODATA, 'zstd')
    with rasterio.open(path, 'w', **profile, ZSTD_LEVEL=15) as dst:
        for window in windows(raw.width, raw.height):
            forest = forests.read(1, window=window) == 1
            if not forest.any():
                continue
            result = np.where(forest, 0, NODATA).astype(np.uint16)
            labels = identifiers[slices(window)]
            live = labels != 0
            values = raw.read(1, window=window)[live]
            result[live] = agents[labels[live]] * 10000 + values % 10 * 1000 + values % 10000 // 10
            dst.write(result, 1, window=window)
            total += len(values)
        dst.update_tags(layout='ASDDD', year=str(year), encoding='agent*10000+severity*1000+DOY',
                        overlap_rule='nearest HLS tile center among disturbance detections',
                        patch_rule='eight-connected class majority, smallest class wins ties',
                        forest_rule='2016 CLCD class 2, union of source tiles')
    return total


def mosaic(year, manifest, reference, output, scratch_dir=None):
    with rasterio.open(reference) as src:
        grid = src.profile.copy()
    transform_ = grid['transform']
    if grid['crs'] != WGS84 or transform_.b or transform_.d or transform_.a <= 0 or transform_.e >= 0:
        raise ValueError('Expected a north-up, unrotated EPSG:4326 reference grid')
    output.parent.mkdir(parents=True, exist_ok=True)
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch_dir) as folder, TemporaryDirectory(dir=output.parent) as staging:
        scratch = Path(folder)
        disturbed, raw_path, forest_path = prepare_mosaic(grid, manifest, year, scratch)
        identifiers = np.memmap(scratch / 'patches.bin', dtype='uint32', mode='w+', shape=disturbed.shape)
        count = label(disturbed, structure=np.ones((3, 3), dtype=np.uint8), output=identifiers)
        temporary = Path(staging) / output.name
        with rasterio.open(raw_path) as raw, rasterio.open(forest_path) as forests:
            agents = vote_patch_agents(raw, identifiers, count)
            total = write_output(temporary, grid, year, raw, forests, identifiers, agents)
        temporary.replace(output)
        del disturbed, identifiers
    print(json.dumps({'year': year, 'patches': int(count), 'disturbed_pixels': total,
                      'annual_raster': str(output)}))
