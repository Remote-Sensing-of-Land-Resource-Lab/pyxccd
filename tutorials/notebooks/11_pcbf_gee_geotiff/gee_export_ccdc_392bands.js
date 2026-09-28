// Whole-tile historical CCDC export with a six-segment final schema.
//
// Copy this single script into the Earth Engine Code Editor. Change only the
// four settings below. Each enabled method creates one Drive export task. The
// task is one logical image; Earth Engine may split it into GeoTIFF shards.

var PROJECT_ID = 'YOUR_EE_PROJECT_ID';
var TILE = 'YOUR_MGRS_TILE';
var DRIVE_FOLDER = 'YOUR_DRIVE_FOLDER';
// Collaborator default: run only the 32-day MAX-RNB method. The managed
// all-observation task has already been submitted separately.
var RUN_ALLOBS = false;
var RUN_RNB32 = true;

var SUPPORT_START = '1982-01-01';
var SUPPORT_END_EXCLUSIVE = '2019-01-01';
var SEGMENT_DEPTH = 6;
var WINDOW_DAYS = 32;
var COMMON_BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'];
var COEFFICIENT_NAMES =
    ['INTP', 'SLP', 'COS', 'SIN', 'COS2', 'SIN2', 'COS3', 'SIN3'];

function validateConfiguration() {
  if (PROJECT_ID === 'YOUR_EE_PROJECT_ID' || PROJECT_ID.trim() === '') {
    throw new Error('Replace PROJECT_ID with your Earth Engine Cloud Project ID.');
  }
  if (TILE === 'YOUR_MGRS_TILE' || !/^[0-9]{2}[A-Z]{3}$/.test(TILE)) {
    throw new Error('Replace TILE with a five-character HLS MGRS tile ID.');
  }
  if (DRIVE_FOLDER === 'YOUR_DRIVE_FOLDER' || DRIVE_FOLDER.trim() === '') {
    throw new Error('Replace DRIVE_FOLDER with a Google Drive folder name.');
  }
  if (!RUN_ALLOBS && !RUN_RNB32) {
    throw new Error('Enable at least one export method.');
  }
}

function findHlsReference() {
  var candidates = [
    {dataset: 'NASA/HLS/HLSS30/v002', band: 'B2'},
    {dataset: 'NASA/HLS/HLSL30/v002', band: 'B2'}
  ];
  for (var i = 0; i < candidates.length; i++) {
    var candidate = candidates[i];
    var collection = ee.ImageCollection(candidate.dataset)
        .filter(ee.Filter.eq('MGRS_TILE_ID', TILE))
        .filterDate('2017-01-01', '2019-01-01');
    if (collection.limit(1).size().getInfo() > 0) {
      return collection.first().select(candidate.band);
    }
  }
  throw new Error('No official HLS reference granule was found for ' + TILE + '.');
}

function wholeTileGeometry(dimensions, transform, crs) {
  var xmin = transform[2];
  var xmax = transform[2] + dimensions[0] * transform[0];
  var ymax = transform[5];
  var ymin = transform[5] + dimensions[1] * transform[4];
  return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], crs, false);
}

function landsatMask(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 0).eq(0);
  [1, 2, 3, 4, 5].forEach(function(bit) {
    clear = clear.and(qa.bitwiseAnd(1 << bit).eq(0));
  });
  return clear.and(image.select('QA_RADSAT').eq(0));
}

function prepareLandsat(image, sourceBands, sensor, targetProjection) {
  var optical = image.select(sourceBands, COMMON_BANDS)
      .multiply(0.0000275).add(-0.2).multiply(10000).toFloat();
  return optical.updateMask(landsatMask(image))
      .resample('bilinear')
      .setDefaultProjection(targetProjection)
      .copyProperties(image, ['system:time_start', 'system:index'])
      .set('sensor', sensor);
}

function sentinelMask(image) {
  var scl = image.select('SCL');
  var valid = scl.neq(0).and(scl.neq(1)).and(scl.neq(3));
  [8, 9, 10, 11].forEach(function(value) {
    valid = valid.and(scl.neq(value));
  });
  var qa60 = image.select('QA60');
  return valid.and(qa60.bitwiseAnd(1 << 10).eq(0))
      .and(qa60.bitwiseAnd(1 << 11).eq(0));
}

function prepareSentinel(image, sensor, targetProjection) {
  var sourceBands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
  var optical = image.select(sourceBands, COMMON_BANDS)
      .multiply(0.0001).multiply(10000).toFloat();
  return optical.updateMask(sentinelMask(image))
      .resample('bilinear')
      .setDefaultProjection(targetProjection)
      .copyProperties(image, ['system:time_start', 'system:index'])
      .set('sensor', sensor);
}

function buildMergedCollection(roi, targetProjection) {
  var tmEtm = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'];
  var oli = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
  var lt4 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
      .filterBounds(roi).filterDate(SUPPORT_START, SUPPORT_END_EXCLUSIVE)
      .map(function(image) {
        return prepareLandsat(image, tmEtm, 'LT04', targetProjection);
      });
  var lt5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
      .filterBounds(roi).filterDate(SUPPORT_START, SUPPORT_END_EXCLUSIVE)
      .map(function(image) {
        return prepareLandsat(image, tmEtm, 'LT05', targetProjection);
      });
  var le7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
      .filterBounds(roi).filterDate(SUPPORT_START, SUPPORT_END_EXCLUSIVE)
      .map(function(image) {
        return prepareLandsat(image, tmEtm, 'LE07', targetProjection);
      });
  var lc8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterBounds(roi).filterDate(SUPPORT_START, SUPPORT_END_EXCLUSIVE)
      .map(function(image) {
        return prepareLandsat(image, oli, 'LC08', targetProjection);
      });
  var sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi).filterDate(SUPPORT_START, SUPPORT_END_EXCLUSIVE);
  var s2a = sentinel.filter(ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2A'))
      .map(function(image) {
        return prepareSentinel(image, 'S2A', targetProjection);
      });
  var s2b = sentinel.filter(ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2B'))
      .map(function(image) {
        return prepareSentinel(image, 'S2B', targetProjection);
      });
  return lt4.merge(lt5).merge(le7).merge(lc8).merge(s2a).merge(s2b)
      .select(COMMON_BANDS).sort('system:time_start');
}

// Select the maximum NIR/BLUE observation in each 32-day window. Exact ties
// are resolved by the earliest acquisition, and the source date is retained.
function selectMaxRnbWindow(collection, start, stop) {
  var scored = collection.filterDate(start, stop).map(function(image) {
    var blue = image.select('BLUE');
    var rnb = image.select('NIR').divide(blue).updateMask(blue.gt(0))
        .rename('RNB').toDouble();
    var sourceTime = ee.Image.constant(ee.Number(image.get('system:time_start')))
        .rename('SOURCE_TIME').toDouble();
    return image.addBands([rnb, sourceTime]);
  });
  var maxRnb = scored.select('RNB').max();
  var winnerTime = scored.map(function(image) {
    return image.select('SOURCE_TIME')
        .updateMask(image.select('RNB').eq(maxRnb)).rename('WINNER_TIME');
  }).select('WINNER_TIME').min();
  return scored.map(function(image) {
    return image.select(COMMON_BANDS)
        .updateMask(image.select('SOURCE_TIME').eq(winnerTime))
        .copyProperties(image, ['system:time_start', 'system:index']);
  });
}

function buildMaxRnb32(collection) {
  var start = ee.Date(SUPPORT_START);
  var stop = ee.Date(SUPPORT_END_EXCLUSIVE);
  var totalDays = stop.difference(start, 'day').getInfo();
  var windowCount = Math.ceil(totalDays / WINDOW_DAYS);
  var output = ee.ImageCollection([]);
  for (var i = 0; i < windowCount; i++) {
    var windowStart = start.advance(i * WINDOW_DAYS, 'day');
    var windowStop = start.advance((i + 1) * WINDOW_DAYS, 'day');
    var subset = collection.filterDate(windowStart, windowStop);
    var selected = ee.ImageCollection(ee.Algorithms.If(
        subset.size().gt(0),
        selectMaxRnbWindow(collection, windowStart, windowStop),
        ee.ImageCollection([])
    ));
    output = output.merge(selected);
  }
  return output.filterDate(SUPPORT_START, SUPPORT_END_EXCLUSIVE)
      .sort('system:time_start');
}

function runCcdc(collection) {
  return ee.Algorithms.TemporalSegmentation.Ccdc({
    collection: collection,
    breakpointBands: COMMON_BANDS,
    tmaskBands: ['GREEN', 'SWIR1'],
    minObservations: 6,
    chiSquareProbability: 0.99,
    minNumOfYearsScaler: 1.33,
    dateFormat: 0,
    lambda: 20,
    maxIterations: 25000
  });
}

function segmentNames(depth) {
  var names = [];
  for (var i = 1; i <= depth; i++) names.push('S0' + i);
  return names;
}

function zeros1D(length) {
  var values = [];
  for (var i = 0; i < length; i++) values.push(0);
  return values;
}

function zeros2D(rows, columns) {
  var values = [];
  for (var i = 0; i < rows; i++) values.push(zeros1D(columns));
  return values;
}

function flatten1D(result, source, depth, suffix) {
  var names = segmentNames(depth).map(function(segment) {
    return segment + '_' + suffix;
  });
  return result.select(source)
      .arrayCat(ee.Image(ee.Array(zeros1D(depth))), 0)
      .arraySlice(0, 0, depth).arrayFlatten([names]).toFloat();
}

function flattenCoefficients(result, band, depth) {
  var segments = segmentNames(depth);
  var names = [];
  segments.forEach(function(segment) {
    COEFFICIENT_NAMES.forEach(function(coefficient) {
      names.push(segment + '_' + band + '_' + coefficient);
    });
  });
  return result.select(band + '_coefs')
      .arrayCat(ee.Image(ee.Array(zeros2D(depth, COEFFICIENT_NAMES.length))), 0)
      .arraySlice(0, 0, depth)
      .arrayFlatten([segments, COEFFICIENT_NAMES]).rename(names).toFloat();
}

function buildFinalImage(result, depth) {
  var segmentCount = result.select('tStart').arrayLength(0)
      .rename('segment_count').toFloat();
  var segmentOverflow = segmentCount.gt(depth)
      .rename('segment_overflow').toFloat();
  var images = [
    flatten1D(result, 'tStart', depth, 'tStart'),
    flatten1D(result, 'tEnd', depth, 'tEnd'),
    flatten1D(result, 'tBreak', depth, 'tBreak'),
    flatten1D(result, 'changeProb', depth, 'changeProb'),
    flatten1D(result, 'numObs', depth, 'numObs'),
    segmentCount,
    segmentOverflow
  ];
  COMMON_BANDS.forEach(function(band) {
    images.push(flattenCoefficients(result, band, depth));
    images.push(flatten1D(result, band + '_rmse', depth, band + '_RMSE'));
    images.push(flatten1D(result, band + '_magnitude', depth, band + '_MAG'));
  });
  return ee.Image.cat(images).toFloat();
}

function createFinalExport(method, collection, region, crs, transform) {
  var description = TILE + '_1982_2018_' + method + '_depth6_final';
  var finalImage = buildFinalImage(runCcdc(collection), SEGMENT_DEPTH);
  Export.image.toDrive({
    image: finalImage,
    description: description,
    folder: DRIVE_FOLDER,
    fileNamePrefix: description,
    region: region,
    crs: crs,
    crsTransform: transform,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    shardSize: 256,
    fileDimensions: 2048,
    skipEmptyTiles: true,
    formatOptions: {cloudOptimized: true}
  });
  print(method + ' final band count:', finalImage.bandNames().size());
  print(method + ' export prefix:', description);
}

validateConfiguration();
var reference = findHlsReference();
var referenceInfo = reference.getInfo();
var bandInfo = referenceInfo.bands[0];
var gridCrs = bandInfo.crs;
var gridTransform = bandInfo.crs_transform;
var gridDimensions = bandInfo.dimensions;
var tileRegion = wholeTileGeometry(gridDimensions, gridTransform, gridCrs);
var merged = buildMergedCollection(tileRegion, reference.projection());

if (RUN_ALLOBS) {
  createFinalExport('allobs', merged, tileRegion, gridCrs, gridTransform);
}
if (RUN_RNB32) {
  createFinalExport('rnb32', buildMaxRnb32(merged), tileRegion,
      gridCrs, gridTransform);
}

print('Tile:', TILE);
print('HLS reference:', referenceInfo.id);
print('Grid CRS:', gridCrs);
print('Grid transform:', gridTransform);
print('Grid dimensions:', gridDimensions);
print('Merged source-image count:', merged.size());
print('Each enabled method creates one final Drive task. Start tasks in Tasks.');
