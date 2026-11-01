// =====================================================
// CCNL annual NTL statistics by settlement
// One CSV export per year
//
// Output fields match the Python cleaning script:
//     sett
//     year
//     ntl_sum
//     ntl_mean
//     ntl_median
//     ntl_min
//     ntl_max
//     ntl_stdDev
//     ntl_count
// =====================================================


// =====================================================
// INPUTS
// =====================================================

var settlements = ee.FeatureCollection(
  'projects/global-snow-436817-q9/assets/named_sett_500m_dissolved_buffers_v2'
);

var ccnl = ee.ImageCollection('BNU/FGS/CCNL/v1');


// =====================================================
// SETTINGS
// =====================================================

// Original settlement-name field in the asset.
var SETTLEMENT_FIELD = 'sett';

var START_YEAR = 1996;
var END_YEAR = 1996;

var EXPORT_FOLDER = 'ntl_ccnl_settlements';
var EXPORT_PREFIX = 'rus_settlement_ccnl_annual_ntl';

var SCALE = 1000;
var TILE_SCALE = 8;


// =====================================================
// INSPECT INPUTS
// =====================================================

print('Settlement count:', settlements.size());
print('First settlement:', settlements.first());
print('Settlement fields:', settlements.first().propertyNames());

print('CCNL collection size:', ccnl.size());
print('CCNL first image:', ccnl.first());


// =====================================================
// PREPARE SETTLEMENTS
// =====================================================

settlements = settlements.map(function(feature) {
  feature = ee.Feature(feature);

  return feature.set({
    sett: feature.get(SETTLEMENT_FIELD)
  });
});


Map.centerObject(settlements, 3);

Map.addLayer(
  settlements.style({
    color: 'red',
    fillColor: '00000000',
    width: 1
  }),
  {},
  'Settlement buffers'
);


// =====================================================
// SUMMARY REDUCER
// =====================================================

var ntlReducer = ee.Reducer.mean()
  .combine({
    reducer2: ee.Reducer.median(),
    sharedInputs: true
  })
  .combine({
    reducer2: ee.Reducer.minMax(),
    sharedInputs: true
  })
  .combine({
    reducer2: ee.Reducer.stdDev(),
    sharedInputs: true
  })
  .combine({
    reducer2: ee.Reducer.count(),
    sharedInputs: true
  })
  .combine({
    reducer2: ee.Reducer.sum(),
    sharedInputs: true
  });


// =====================================================
// GET IMAGE FOR YEAR
// =====================================================

function getYearImage(year) {

  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');

  var yearlyCollection = ccnl
    .filterDate(start, end)
    .select('b1');

  return ee.Image(yearlyCollection.first())
    .rename('ntl')
    .toFloat();
}


// =====================================================
// EXPORT ONE YEAR
// =====================================================

function exportYear(year) {

  var image = getYearImage(year);

  // Retain valid zero-valued pixels.
  image = image.updateMask(image.gte(0));

  var reduced = image.reduceRegions({
    collection: settlements,
    reducer: ntlReducer,
    scale: SCALE,
    tileScale: TILE_SCALE
  });

  var output = reduced.map(function(feature) {
    feature = ee.Feature(feature);

    return ee.Feature(null, {
      sett: feature.get('sett'),
      year: year,

      ntl_sum: feature.get('sum'),
      ntl_mean: feature.get('mean'),
      ntl_median: feature.get('median'),
      ntl_min: feature.get('min'),
      ntl_max: feature.get('max'),
      ntl_stdDev: feature.get('stdDev'),
      ntl_count: feature.get('count')
    });
  });

  Export.table.toDrive({
    collection: output,
    description: EXPORT_PREFIX + '_' + year,
    folder: EXPORT_FOLDER,
    fileNamePrefix: EXPORT_PREFIX + '_' + year,
    fileFormat: 'CSV',

    selectors: [
      'sett',
      'year',
      'ntl_sum',
      'ntl_mean',
      'ntl_median',
      'ntl_min',
      'ntl_max',
      'ntl_stdDev',
      'ntl_count'
    ]
  });
}


// =====================================================
// CREATE ONE EXPORT TASK PER YEAR
// =====================================================

for (var year = START_YEAR; year <= END_YEAR; year++) {
  exportYear(year);
}
