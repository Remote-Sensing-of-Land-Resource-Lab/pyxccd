// CCDC + PCBF example for the Earth Engine Code Editor.
// Edit USER SETTINGS, then click Run.


// ============================================================================
// 1. PCBF
// ============================================================================

var PCBF_BANDS = ['GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'];
var CATEGORY_THRESHOLD = -200;
var PCBF_DEFAULTS = {
  durationThresholdDays: 192,
  zValue: 2.326,
  requiredBands: 4
};


function withDefaults(options) {
  options = options || {};
  Object.keys(PCBF_DEFAULTS).forEach(function(key) {
    if (options[key] === undefined) {
      options[key] = PCBF_DEFAULTS[key];
    }
  });
  if (options.durationThresholdDays <= 0 ||
      Math.floor(options.durationThresholdDays) !== options.durationThresholdDays) {
    throw new Error('durationThresholdDays must be a positive integer.');
  }
  if (options.zValue <= 0) {
    throw new Error('zValue must be positive.');
  }
  if (options.requiredBands < 1 || options.requiredBands > PCBF_BANDS.length ||
      Math.floor(options.requiredBands) !== options.requiredBands) {
    throw new Error('requiredBands must be an integer between 1 and 5.');
  }
  return options;
}


function allButLast(arrayImage) {
  return arrayImage.arraySlice(0, 0, -1);
}


function allButFirst(arrayImage) {
  return arrayImage.arraySlice(0, 1);
}


function lastElement(arrayImage) {
  return arrayImage.arraySlice(0, -1);
}


function appendTerminal(pairArray, terminalArray) {
  return pairArray.arrayCat(terminalArray, 0);
}


function coefficient(coefficientArray, index) {
  return coefficientArray
    .arraySlice(1, index, index + 1)
    .arrayProject([0]);
}


function harmonicPrediction(coefficientArray, jDay) {
  var phase = jDay.multiply(2.0 * Math.PI / 365.25);
  return coefficient(coefficientArray, 0)
    .add(coefficient(coefficientArray, 1).multiply(jDay))
    .add(coefficient(coefficientArray, 2).multiply(phase.cos()))
    .add(coefficient(coefficientArray, 3).multiply(phase.sin()))
    .add(coefficient(coefficientArray, 4).multiply(phase.multiply(2).cos()))
    .add(coefficient(coefficientArray, 5).multiply(phase.multiply(2).sin()))
    .add(coefficient(coefficientArray, 6).multiply(phase.multiply(3).cos()))
    .add(coefficient(coefficientArray, 7).multiply(phase.multiply(3).sin()));
}


function directionalBandPass(previousPrediction, followingPrediction,
                             previousRmse, magnitude, zValue) {
  var lower = previousPrediction.subtract(previousRmse.multiply(zValue));
  var upper = previousPrediction.add(previousRmse.multiply(zValue));

  var negativePass = magnitude.lt(0).and(followingPrediction.gte(lower));
  var positivePass = magnitude.gt(0).and(followingPrediction.lte(upper));
  var zeroPass = magnitude.eq(0).and(
    followingPrediction.subtract(previousPrediction).abs()
      .lte(previousRmse.multiply(zValue))
  );
  return negativePass.or(positivePass).or(zeroPass);
}


function applyPCBF(ccdcImage, userOptions) {
  var options = withDefaults(userOptions);
  ccdcImage = ee.Image(ccdcImage);

  var tStart = ccdcImage.select('tStart');
  var tEnd = ccdcImage.select('tEnd');
  var tBreak = ccdcImage.select('tBreak');
  var changeProb = ccdcImage.select('changeProb');

  var previousEnd = allButLast(tEnd);
  var previousBreak = allButLast(tBreak);
  var previousProbability = allButLast(changeProb);
  var followingStart = allButFirst(tStart);
  var followingEnd = allButFirst(tEnd);

  var hasFollowing = previousBreak.multiply(0).add(1).eq(1);
  var confirmedPair = previousBreak.gt(0).and(previousProbability.gt(0));

  var previous = {};
  var following = {};
  PCBF_BANDS.forEach(function(band) {
    previous[band] = {
      coefs: allButLast(ccdcImage.select(band + '_coefs')),
      rmse: allButLast(ccdcImage.select(band + '_rmse')),
      magnitude: allButLast(ccdcImage.select(band + '_magnitude'))
    };
    following[band] = {
      coefs: allButFirst(ccdcImage.select(band + '_coefs'))
    };
  });

  var redMagnitude = previous.RED.magnitude;
  var nirMagnitude = previous.NIR.magnitude;
  var swir1Magnitude = previous.SWIR1.magnitude;
  var positiveThreshold = -CATEGORY_THRESHOLD;
  var greenerDirection = nirMagnitude.gt(CATEGORY_THRESHOLD)
    .and(redMagnitude.lt(positiveThreshold))
    .and(swir1Magnitude.lt(positiveThreshold));

  var previousRedSlope = coefficient(previous.RED.coefs, 1);
  var previousNirSlope = coefficient(previous.NIR.coefs, 1);
  var previousSwir1Slope = coefficient(previous.SWIR1.coefs, 1);
  var followingRedSlope = coefficient(following.RED.coefs, 1);
  var followingNirSlope = coefficient(following.NIR.coefs, 1);
  var followingSwir1Slope = coefficient(following.SWIR1.coefs, 1);
  var afforestation = followingNirSlope.gt(previousNirSlope.abs())
    .and(followingRedSlope.lt(previousRedSlope.abs().multiply(-1)))
    .and(followingSwir1Slope.lt(previousSwir1Slope.abs().multiply(-1)));
  var category2 = greenerDirection.and(afforestation.not()).and(hasFollowing);

  var zeroPairs = previousBreak.multiply(0);
  var initialState = ee.Dictionary({
    recovered: zeroPairs.eq(1),
    firstDate: zeroPairs,
    firstPassingCount: zeroPairs
  });

  var finalState = ee.Dictionary(
    ee.List.sequence(0, options.durationThresholdDays - 1).iterate(function(offset, state) {
      state = ee.Dictionary(state);
      offset = ee.Number(offset);
      var recoveredAlready = ee.Image(state.get('recovered'));
      var firstDate = ee.Image(state.get('firstDate'));
      var firstPassingCount = ee.Image(state.get('firstPassingCount'));

      var date = previousEnd.add(offset);
      var active = hasFollowing
        .and(date.gte(followingStart))
        .and(date.lte(followingEnd))
        .and(date.lte(previousEnd.add(options.durationThresholdDays - 1)));

      var passingBandCount = zeroPairs;
      PCBF_BANDS.forEach(function(band) {
        var before = harmonicPrediction(previous[band].coefs, date);
        var after = harmonicPrediction(following[band].coefs, date);
        var passed = directionalBandPass(
          before,
          after,
          previous[band].rmse,
          previous[band].magnitude,
          options.zValue
        ).and(active);
        passingBandCount = passingBandCount.add(passed);
      });

      var recoveredToday = passingBandCount.gte(options.requiredBands)
        .and(active)
        .and(recoveredAlready.not());
      return ee.Dictionary({
        recovered: recoveredAlready.or(recoveredToday),
        firstDate: firstDate.add(date.multiply(recoveredToday)),
        firstPassingCount: firstPassingCount.add(
          passingBandCount.multiply(recoveredToday)
        )
      });
    }, initialState)
  );

  var recoveredPair = ee.Image(finalState.get('recovered'));
  var firstRecoveryDate = ee.Image(finalState.get('firstDate'));
  var firstPassingCount = ee.Image(finalState.get('firstPassingCount'));

  var category2RejectedPair = confirmedPair.and(category2);
  var recoveryRejectedPair = confirmedPair
    .and(category2.not())
    .and(recoveredPair);
  var rejectedPair = category2RejectedPair.or(recoveryRejectedPair);
  var retainedPair = confirmedPair.and(rejectedPair.not());

  // A confirmed terminal break is retained because no following segment exists.
  var terminalBreak = lastElement(tBreak);
  var terminalProbability = lastElement(changeProb);
  var terminalConfirmed = terminalBreak.gt(0).and(terminalProbability.gt(0));
  var terminalZero = terminalBreak.multiply(0);
  var terminalMinusOne = terminalZero.add(-1);

  var filteredChangeProb = appendTerminal(
    previousProbability.multiply(rejectedPair.not()),
    terminalProbability
  ).rename('filteredChangeProb');
  var retainedBreakMask = appendTerminal(
    retainedPair,
    terminalConfirmed
  ).rename('retainedBreakMask');
  var retainedTBreak = tBreak.multiply(retainedBreakMask)
    .rename('retainedTBreak');
  var recoveryDate = appendTerminal(
    firstRecoveryDate,
    terminalZero
  ).rename('recoveryDate');
  var pairPersistence = firstRecoveryDate.subtract(previousEnd)
    .multiply(recoveredPair)
    .add(recoveredPair.not().multiply(-1));
  var persistenceDays = appendTerminal(
    pairPersistence,
    terminalMinusOne
  ).rename('persistenceDays');

  var category2Rejected = appendTerminal(
    category2RejectedPair,
    terminalZero
  ).rename('category2Rejected');
  var recoveryRejected = appendTerminal(
    recoveryRejectedPair,
    terminalZero
  ).rename('recoveryRejected');
  var persistentRetained = appendTerminal(
    retainedPair,
    terminalZero
  ).rename('persistentRetained');
  var terminalRetained = appendTerminal(
    zeroPairs,
    terminalConfirmed
  ).rename('terminalRetained');
  var hasFollowingFull = appendTerminal(
    hasFollowing,
    terminalZero
  ).rename('hasFollowing');
  var passingBandCountAtRecovery = appendTerminal(
    firstPassingCount,
    terminalZero
  ).rename('passingBandCountAtRecovery');

  return {
    source: ccdcImage,
    filteredChangeProb: filteredChangeProb,
    retainedBreakMask: retainedBreakMask,
    retainedTBreak: retainedTBreak,
    recoveryDate: recoveryDate,
    persistenceDays: persistenceDays,
    category2Rejected: category2Rejected,
    recoveryRejected: recoveryRejected,
    persistentRetained: persistentRetained,
    terminalRetained: terminalRetained,
    hasFollowing: hasFollowingFull,
    passingBandCountAtRecovery: passingBandCountAtRecovery
  };
}


// ============================================================================
// 2. USER SETTINGS
// ============================================================================

// Validated single-pixel example also used by the Jupyter tutorial.
var point = ee.Geometry.Point([109.006790, 37.251270]);
var roi = point.buffer(1500);
var startDate = '1995-01-01';
var endDate = '2007-01-01';
var analysisBands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'];
var PLOT_BAND = 'SWIR2';
var MAX_RNB_WINDOW_DAYS = 32;
var COMPOSITING_ANCHOR = '1982-01-01';
var MODEL_STEP_DAYS = 4;
var CHART_Y_MIN = 500;
var CHART_Y_MAX = 4500;
var targetProjection = ee.Projection('EPSG:32649')
  .atScale(30);


function landsatValidMask(image) {
  // Mask fill, dilated cloud, cirrus, cloud, shadow, snow, and saturation.
  return image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0));
}


function prepareLandsat57(image) {
  // Landsat Collection 2 Level 2 surface reflectance on the x10,000 scale.
  var reflectance = image
    .select(
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
      analysisBands
    )
    .multiply(0.0000275)
    .add(-0.2)
    .multiply(10000)
    .toFloat();

  return reflectance
    .updateMask(landsatValidMask(image))
    .resample('bilinear')
    .setDefaultProjection(targetProjection)
    .copyProperties(
      image, ['system:time_start', 'system:index', 'SPACECRAFT_ID']
    );
}


var landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(point.buffer(60))
  .filterDate(startDate, endDate)
  .map(prepareLandsat57);

var landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
  .filterBounds(point.buffer(60))
  .filterDate(startDate, endDate)
  .map(prepareLandsat57);

var inputCollection = landsat5
  .merge(landsat7)
  .sort('system:time_start');


// Select the point-scale maximum-NIR/Blue observation in every complete
// fixed 32-day window, retaining the acquisition date of the selected image.
function pointWinnerForWindow(collection, pointGeometry, start, stop) {
  var scored = collection.filterDate(start, stop).map(function(image) {
    var blue = image.select('BLUE');
    var rnb = image.select('NIR').divide(blue)
      .updateMask(blue.gt(0))
      .rename('RNB');
    var pointRnb = rnb.reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: pointGeometry,
      scale: 30,
      maxPixels: 16
    }).get('RNB');
    return image.set('POINT_RNB', pointRnb);
  });

  var validCandidates = scored.filter(ee.Filter.notNull(['POINT_RNB']));
  var hasData = validCandidates.size().gt(0);
  var sortedCandidates = validCandidates.sort('POINT_RNB', false);
  var winner = ee.Image(ee.Algorithms.If(
    hasData,
    sortedCandidates.first(),
    emptyWindowImage(start)
  ));
  var winnerTime = ee.Number(ee.Algorithms.If(
    hasData,
    ee.Image(sortedCandidates.first()).get('system:time_start'),
    start.millis()
  ));

  return winner.select(analysisBands)
    .clip(pointGeometry.buffer(30))
    .set('system:time_start', winnerTime)
    .set('HAS_DATA', ee.Number(ee.Algorithms.If(hasData, 1, 0)));
}


function emptyWindowImage(windowStart) {
  return ee.Image.constant([0, 0, 0, 0, 0, 0])
    .rename(analysisBands)
    .toFloat()
    .updateMask(ee.Image.constant(0))
    .setDefaultProjection(targetProjection)
    .set('system:time_start', windowStart.millis())
    .set('HAS_DATA', 0);
}


function buildPointMaxRnb32(collection, pointGeometry) {
  var millisPerDay = 24 * 60 * 60 * 1000;
  var anchorMillis = Date.parse(COMPOSITING_ANCHOR);
  var firstWindowIndex = Math.ceil(
    (Date.parse(startDate) - anchorMillis) /
      (MAX_RNB_WINDOW_DAYS * millisPerDay)
  );
  var stopWindowIndex = Math.floor(
    (Date.parse(endDate) - anchorMillis) /
      (MAX_RNB_WINDOW_DAYS * millisPerDay)
  );
  var firstWindow = ee.Date(COMPOSITING_ANCHOR);
  var indices = ee.List.sequence(firstWindowIndex, stopWindowIndex - 1);

  var winners = indices.map(function(value) {
    var index = ee.Number(value);
    var windowStart = firstWindow.advance(
      index.multiply(MAX_RNB_WINDOW_DAYS), 'day'
    );
    var windowStop = firstWindow.advance(
      index.add(1).multiply(MAX_RNB_WINDOW_DAYS), 'day'
    );
    return pointWinnerForWindow(
      collection, pointGeometry, windowStart, windowStop
    );
  });

  return ee.ImageCollection.fromImages(winners)
    .filter(ee.Filter.eq('HAS_DATA', 1))
    .select(analysisBands)
    .sort('system:time_start');
}


var compositingCollection = buildPointMaxRnb32(inputCollection, point);


function runCcdc(collection) {
  return ee.Algorithms.TemporalSegmentation.Ccdc({
    collection: collection,
    breakpointBands: analysisBands,
    tmaskBands: ['GREEN', 'SWIR1'],
    minObservations: 6,
    chiSquareProbability: 0.99,
    minNumOfYearsScaler: 1.33,
    dateFormat: 0,
    lambda: 20,
    maxIterations: 25000
  });
}


// PCBF requires CCDC dates in Julian days (dateFormat 0).
var allObservationCcdc = runCcdc(inputCollection);
var compositingCcdc = runCcdc(compositingCollection);


var pcbf = applyPCBF(allObservationCcdc, {
  durationThresholdDays: 192,
  zValue: 2.326,
  requiredBands: 4
});


// Reduce record-aligned arrays for map display only.
var originalBreakMask = allObservationCcdc.select('tBreak')
  .gt(0)
  .arrayReduce(ee.Reducer.max(), [0])
  .arrayGet([0])
  .rename('original_break_mask');

var retainedBreak = pcbf.retainedBreakMask
  .arrayReduce(ee.Reducer.max(), [0])
  .arrayGet([0])
  .rename('retained_break');

var recoveryRejected = pcbf.recoveryRejected
  .arrayReduce(ee.Reducer.max(), [0])
  .arrayGet([0])
  .rename('recovery_rejected');

var category2Rejected = pcbf.category2Rejected
  .arrayReduce(ee.Reducer.max(), [0])
  .arrayGet([0])
  .rename('category2_rejected');

Map.centerObject(point, 13);
Map.addLayer(roi, {color: 'FFFFFF'}, 'Example area', false);
Map.addLayer(originalBreakMask.selfMask().clip(roi),
             {palette: ['BDBDBD']}, 'CCDC break pixels', false);
Map.addLayer(retainedBreak.selfMask().clip(roi),
             {palette: ['D73027']}, 'CCDC+PCBF retained breaks', true);
Map.addLayer(recoveryRejected.selfMask().clip(roi),
             {palette: ['6B6B6B']}, 'PCBF-removed candidates', false);
Map.addLayer(category2Rejected.selfMask().clip(roi),
             {palette: ['31A354']}, 'Category-2 exclusions', false);
Map.addLayer(point, {color: '00FFFF'}, 'Sample point', true);


// ============================================================================
// 3. TIME-SERIES DISPLAY
// ============================================================================

var MILLIS_PER_DAY = 24 * 60 * 60 * 1000;
var CCDC_EPOCH_DAYS = 719163;


function valueAtPoint(image) {
  return image.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    crs: targetProjection,
    scale: 30,
    maxPixels: 1e6
  });
}


function scalarAtPoint(image, bandName) {
  return image.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    crs: targetProjection,
    scale: 30,
    maxPixels: 1e6
  }).get(bandName);
}


function arrayBandAtPoint(image, bandName) {
  return ee.Array(valueAtPoint(image.select(bandName)).get(bandName)).toList();
}


function ccdcPredictionAtDate(ccdcImage, bandName, jDay) {
  jDay = ee.Number(jDay);
  var starts = ccdcImage.select('tStart');
  var ends = ccdcImage.select('tEnd');
  var active = starts.lte(jDay).and(ends.gte(jDay));
  var prediction = harmonicPrediction(
    ccdcImage.select(bandName + '_coefs'),
    jDay
  );
  var activeCount = active
    .arrayReduce(ee.Reducer.sum(), [0])
    .arrayGet([0]);
  return prediction
    .multiply(active)
    .arrayReduce(ee.Reducer.sum(), [0])
    .arrayGet([0])
    .updateMask(activeCount.gt(0))
    .rename('prediction');
}


function ccdcSegmentPredictionAtDate(ccdcImage, bandName, jDay, segmentIndex) {
  var predictionArray = harmonicPrediction(
    ccdcImage.select(bandName + '_coefs'),
    ee.Number(jDay)
  );
  var values = ee.Array(
    valueAtPoint(predictionArray).get(bandName + '_coefs')
  ).toList();
  return values.get(ee.Number(segmentIndex));
}


function buildObservationFeatures(collection, seriesName) {
  var images = collection.toList(collection.size());
  return ee.FeatureCollection(images.map(function(element) {
    var image = ee.Image(element);
    var value = scalarAtPoint(image.select(PLOT_BAND), PLOT_BAND);
    return ee.Feature(null)
      .set('system:time_start', image.get('system:time_start'))
      .set(seriesName, value);
  })).filter(ee.Filter.notNull([seriesName]));
}


function buildModelGridFeatures() {
  var firstMillis = ee.Date(startDate).millis();
  var lastMillis = ee.Date(endDate).advance(-1, 'day').millis();
  var displayDates = ee.List.sequence(
    firstMillis,
    lastMillis,
    MODEL_STEP_DAYS * MILLIS_PER_DAY
  );
  return ee.FeatureCollection(displayDates.map(function(value) {
    return ee.Feature(null, {
      'system:time_start': ee.Number(value)
    });
  }));
}


function addModelPrediction(features, ccdcImage) {
  return features.map(function(feature) {
    feature = ee.Feature(feature);
    var millis = ee.Number(feature.get('system:time_start'));
    var jDay = millis.divide(MILLIS_PER_DAY).add(CCDC_EPOCH_DAYS);
    var prediction = scalarAtPoint(
      ccdcPredictionAtDate(ccdcImage, PLOT_BAND, jDay),
      'prediction'
    );
    return feature.set('Harmonic model', prediction);
  });
}


function buildBreakCandidates(ccdcImage, flagImage, flagBand) {
  var breakDays = arrayBandAtPoint(ccdcImage, 'tBreak');
  var flags = arrayBandAtPoint(flagImage, flagBand);
  var indices = ee.List.sequence(0, breakDays.length().subtract(1));
  return ee.FeatureCollection(indices.map(function(value) {
    var index = ee.Number(value);
    var breakDay = ee.Number(breakDays.get(index));
    var flag = ee.Number(flags.get(index));
    var fittedValue = ccdcSegmentPredictionAtDate(
      ccdcImage, PLOT_BAND, breakDay, index
    );
    var date = ee.Date(
      breakDay.subtract(CCDC_EPOCH_DAYS).multiply(MILLIS_PER_DAY)
    );
    return ee.Feature(null, {
      breakDay: breakDay,
      dateLabel: date.format('YYYY-MM-dd'),
      fittedValue: fittedValue,
      flag: flag,
      valid: breakDay.gt(0).and(flag.gt(0))
    });
  }))
    .filter(ee.Filter.eq('valid', 1))
    .sort('breakDay');
}


function buildVerticalBreakFeatures(ccdcImage, flagImage, flagBand,
                                    seriesName) {
  var candidates = buildBreakCandidates(ccdcImage, flagImage, flagBand);
  var fallback = ee.Feature(null, {
    breakDay: 0,
    dateLabel: 'none',
    fittedValue: 0,
    valid: 0
  });
  var first = ee.Feature(candidates.toList(1).cat([fallback]).get(0));
  var breakDay = ee.Number(first.get('breakDay'));
  var breakMillis = breakDay.subtract(CCDC_EPOCH_DAYS)
    .multiply(MILLIS_PER_DAY);
  var valid = ee.Number(first.get('valid'));

  var lower = ee.Feature(null)
    .set('system:time_start', breakMillis.add(1))
    .set(seriesName, CHART_Y_MIN)
    .set('valid', valid);
  var upper = ee.Feature(null)
    .set('system:time_start', breakMillis.add(2))
    .set(seriesName, CHART_Y_MAX)
    .set('valid', valid);

  return ee.Dictionary({
    features: ee.FeatureCollection([lower, upper])
      .filter(ee.Filter.eq('valid', 1)),
    dateLabel: first.get('dateLabel'),
    fittedValue: first.get('fittedValue')
  });
}


var removedBreakMask = pcbf.recoveryRejected
  .max(pcbf.category2Rejected)
  .rename('removedBreakMask');
var compositingBreakMask = compositingCcdc.select('changeProb')
  .gt(0)
  .rename('compositingBreakMask');

var removedBreak = buildVerticalBreakFeatures(
  allObservationCcdc,
  removedBreakMask,
  'removedBreakMask',
  'PCBF-removed candidate'
);
var compositingBreak = buildVerticalBreakFeatures(
  compositingCcdc,
  compositingBreakMask,
  'compositingBreakMask',
  'Retained break'
);

var modelGrid = buildModelGridFeatures();
var pcbfChartRows = buildObservationFeatures(
  inputCollection,
  'All valid observations'
)
  .merge(modelGrid)
  .merge(ee.FeatureCollection(removedBreak.get('features')));
pcbfChartRows = addModelPrediction(
  pcbfChartRows,
  allObservationCcdc
).sort('system:time_start');

var compositingChartRows = buildObservationFeatures(
  compositingCollection,
  '32-day selected observations'
)
  .merge(modelGrid)
  .merge(ee.FeatureCollection(compositingBreak.get('features')));
compositingChartRows = addModelPrediction(
  compositingChartRows,
  compositingCcdc
).sort('system:time_start');


function buildComparisonChart(features, title, observationSeries,
                              breakSeries, observationColor,
                              observationShape, breakColor, dashedBreak) {
  var options = {
    title: title,
    titleTextStyle: {fontSize: 16, bold: true, color: '#202124'},
    hAxis: {
      title: 'Observation date',
      format: 'yyyy',
      viewWindow: {
        min: new Date(startDate + 'T00:00:00Z'),
        max: new Date(endDate + 'T00:00:00Z')
      },
      gridlines: {color: '#E2E6EA'},
      minorGridlines: {color: '#F3F5F7'},
      baselineColor: '#6F7378',
      textStyle: {fontSize: 11, color: '#3C4043'},
      titleTextStyle: {fontSize: 12, bold: true, italic: false}
    },
    vAxis: {
      title: 'SWIR2 × 10,000',
      viewWindow: {min: CHART_Y_MIN, max: CHART_Y_MAX},
      gridlines: {color: '#E2E6EA'},
      minorGridlines: {color: '#F3F5F7'},
      baselineColor: '#6F7378',
      textStyle: {fontSize: 11, color: '#3C4043'},
      titleTextStyle: {fontSize: 12, bold: true, italic: false}
    },
    legend: {
      position: 'top',
      alignment: 'center',
      textStyle: {fontSize: 11, color: '#30343B'}
    },
    backgroundColor: '#FFFFFF',
    chartArea: {left: 82, top: 72, width: '87%', height: '66%'},
    interpolateNulls: false,
    lineWidth: 0,
    pointSize: 0,
    width: 1120,
    height: 350,
    series: {
      0: {
        color: observationColor,
        lineWidth: 0,
        pointSize: observationShape === 'square' ? 6 : 4,
        pointShape: observationShape
      },
      1: {
        color: '#4E4E4E',
        lineWidth: 2.6,
        pointSize: 0
      },
      2: {
        color: breakColor,
        lineWidth: 1.5,
        pointSize: 0
      }
    }
  };
  if (dashedBreak) {
    options.series[2].lineDashStyle = [6, 4];
  }

  return ui.Chart.feature.byFeature({
    features: features,
    xProperty: 'system:time_start',
    yProperties: [observationSeries, 'Harmonic model', breakSeries]
  })
    .setChartType('LineChart')
    .setOptions(options);
}


print(buildComparisonChart(
  pcbfChartRows,
  '(a) CCDC+PCBF',
  'All valid observations',
  'PCBF-removed candidate',
  '#1593C4',
  'circle',
  '#6B6B6B',
  true
));
print(buildComparisonChart(
  compositingChartRows,
  '(b) CCDC+Compositing',
  '32-day selected observations',
  'Retained break',
  '#E99A18',
  'square',
  '#D73027',
  false
));
