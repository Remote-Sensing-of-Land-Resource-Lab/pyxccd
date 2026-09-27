// GEE Code Editor 示例：公开 Landsat 8/9 -> CCDC -> PCBF
//
// 用途：让没有私有资产的读者先在一个 1.5 km 小范围内跑通完整流程。
// 输出：地图预览、原始 SWIR2 时序、点位置的 CCDC/PCBF 数组诊断。
// 注意：本脚本不会自动创建任何 Export 任务，也不会修改已有 CCDC 结果。


// ============================================================================
// 1. PCBF 函数（与已核准的 pyxccd 规则保持一致）
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
  // GEE 原生 CCDC 使用 0/1；pyxccd 常见结果使用 0/100。
  // 两种编码中，正值都表示该断点已确认，不改变 PCBF 判定规则。
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

  // 与 getcategory_cold(t_c = -200) 一致的 category-2 排除。
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

  // 最末段没有后续段可供恢复判断，因此保守保留其已确认断点。
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
// 2. 可直接运行的公开数据示例
// ============================================================================

// 示例点位位于黄土高原。只需修改经纬度即可检查其他位置。
var point = ee.Geometry.Point([108.83949275954184, 36.91699927676264]);
var roi = point.buffer(1500);
var startDate = '2017-01-01';
var endDate = '2026-01-01';
var analysisBands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'];


function prepareLandsat89(image) {
  // QA_PIXEL bits 0-5：fill、dilated cloud、cirrus、cloud、shadow、snow。
  var clear = image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0));

  // Collection 2 L2: SR = DN * 0.0000275 - 0.2。
  // 再乘 10000，使数据尺度与函数内部固定的 -200 类别规则保持一致。
  var reflectance = image
    .select(
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
      analysisBands
    )
    .multiply(0.0000275)
    .add(-0.2)
    .multiply(10000)
    .toFloat();

  return reflectance
    .updateMask(clear)
    .copyProperties(image, ['system:time_start', 'SPACECRAFT_ID']);
}


var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .map(prepareLandsat89);

var landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .map(prepareLandsat89);

var inputCollection = landsat8
  .merge(landsat9)
  .sort('system:time_start');

print('有效 Landsat 8/9 影像数量', inputCollection.size());
print('第一景预处理后的波段', ee.Image(inputCollection.first()).bandNames());


// CCDC 的 dateFormat 必须为 0，PCBF 才能按天计算 spectral change persistence。
var ccdc = ee.Algorithms.TemporalSegmentation.Ccdc({
  collection: inputCollection,
  breakpointBands: analysisBands,
  tmaskBands: ['GREEN', 'SWIR1'],
  minObservations: 6,
  chiSquareProbability: 0.99,
  minNumOfYearsScaler: 1.33,
  dateFormat: 0,
  lambda: 20,
  maxIterations: 25000
});


var pcbf = applyPCBF(ccdc, {
  durationThresholdDays: 192,
  zValue: 2.326,
  requiredBands: 4
});


// 将数组沿 segment 轴压缩成单波段，便于地图预览。
var originalBreakMask = ccdc.select('tBreak')
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
Map.addLayer(roi, {color: 'FFFFFF'}, '示例范围', false);
Map.addLayer(originalBreakMask.selfMask().clip(roi),
             {palette: ['BDBDBD']}, 'CCDC 检出的断点像元', false);
Map.addLayer(retainedBreak.selfMask().clip(roi),
             {palette: ['E6550D']}, 'CCDC+PCBF 保留的断点像元', true);
Map.addLayer(recoveryRejected.selfMask().clip(roi),
             {palette: ['2B8CBE']}, '因光谱恢复而过滤', false);
Map.addLayer(category2Rejected.selfMask().clip(roi),
             {palette: ['31A354']}, 'category-2 排除', false);
Map.addLayer(point, {color: '00FFFF'}, '检查点', true);


// 原始观测曲线用于确认输入数据是否合理；它不是拟合曲线。
var swir2Chart = ui.Chart.image.series({
  imageCollection: inputCollection.select('SWIR2'),
  region: point,
  reducer: ee.Reducer.first(),
  scale: 30,
  xProperty: 'system:time_start'
}).setOptions({
  title: '示例点原始 Landsat SWIR2 观测',
  hAxis: {title: '日期'},
  vAxis: {title: 'SWIR2 × 10000'},
  pointSize: 4,
  lineWidth: 0,
  colors: ['#2B8CBE']
});
print(swir2Chart);


// 在 Console 中查看点位置的完整数组。数组顺序与 CCDC segment 顺序一致。
function valueAtPoint(image) {
  return image.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 30,
    maxPixels: 1e6
  });
}

print('原始 CCDC：tBreak 与 changeProb',
      valueAtPoint(ccdc.select(['tBreak', 'changeProb'])));
print('PCBF：保留断点日期', valueAtPoint(pcbf.retainedTBreak));
print('PCBF：恢复日期', valueAtPoint(pcbf.recoveryDate));
print('PCBF：光谱变化持续天数', valueAtPoint(pcbf.persistenceDays));
print('PCBF：恢复时满足条件的波段数',
      valueAtPoint(pcbf.passingBandCountAtRecovery));
print('PCBF：过滤与保留原因', valueAtPoint(ee.Image.cat([
  pcbf.category2Rejected,
  pcbf.recoveryRejected,
  pcbf.persistentRetained,
  pcbf.terminalRetained
])));
