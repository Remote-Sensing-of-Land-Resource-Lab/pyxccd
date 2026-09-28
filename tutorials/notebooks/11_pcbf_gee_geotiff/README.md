# Apply PCBF to GEE CCDC GeoTIFFs

This example exports flattened CCDC parameters from Google Earth Engine (GEE),
downloads the exported GeoTIFFs through the Google Drive webpage, and applies
PCBF locally. PCBF does not rerun or refit CCDC.

The example contains four files:

```text
README.md
gee_export_ccdc_392bands.js
run_pcbf_from_geotiffs.py
pcbf_geotiff_core.py
```

## 1. Export CCDC parameters from GEE

Open `gee_export_ccdc_392bands.js` in the GEE Code Editor and change the user
settings at the beginning of the script:

```javascript
var PROJECT_ID = 'YOUR_EE_PROJECT_ID';
var TILE = 'YOUR_MGRS_TILE';
var DRIVE_FOLDER = 'YOUR_DRIVE_FOLDER';
var RUN_ALLOBS = true;
var RUN_RNB32 = false;
```

Click **Run**, then start the export task in the **Tasks** panel. The script
runs CCDC for the requested HLS MGRS tile and exports one logical image to the
specified Google Drive folder. With `fileDimensions: 2048`, a 3660 × 3660 HLS
tile is normally delivered as four spatial GeoTIFF shards.

After the task is complete, download the four GeoTIFFs from the Google Drive
webpage and place them in one local directory.

## 2. Input structure

The input directory contains the four spatial GeoTIFF shards created by one
whole-tile GEE export. The number of exported bands depends on the maximum
number of CCDC segments retained for each pixel (`SEGMENT_DEPTH`). For a depth
of `D`, the flattened export contains:

- `5D` metadata bands: `tStart`, `tEnd`, `tBreak`, `changeProb` and `numObs`;
- two diagnostic bands: `segment_count` and `segment_overflow`;
- `10D` bands for each of the six spectral bands: eight harmonic coefficients,
  one RMSE and one change magnitude for every retained segment.

The total number of bands is therefore:

```text
5D + 2 + 6 × 10D = 65D + 2
```

This example uses `SEGMENT_DEPTH = 6`, giving `65 × 6 + 2 = 392` bands. If a
different segment depth is selected during the GEE export, the expected band
count changes accordingly. `segment_count` records the actual number of CCDC
segments for each pixel, while `segment_overflow` identifies pixels whose
actual segment count exceeds the exported depth.

For the depth-six example, each input file must have:

- 392 Float32 bands and a six-segment maximum depth;
- GEE CCDC `dateFormat: 0`;
- reflectance scaled by 10,000;
- the same CRS and 30 m grid, with non-overlapping spatial bounds.

The 392 bands are ordered as follows:

| Bands | Content |
|---|---|
| 1–6 | `tStart` |
| 7–12 | `tEnd` |
| 13–18 | `tBreak` |
| 19–24 | `changeProb` |
| 25–30 | `numObs` |
| 31–32 | `segment_count`, `segment_overflow` |
| 33–392 | BLUE, GREEN, RED, NIR, SWIR1 and SWIR2 coefficients, RMSE and magnitude |

The shards do not need to be merged first. Their CRS, transform and bounds
already define their original positions.

## 3. Run PCBF locally

Install the two dependencies:

```bash
pip install numpy rasterio
```

Run the local processor:

```bash
python run_pcbf_from_geotiffs.py \
  --input-dir /path/to/downloaded_geotiffs \
  --output-dir /path/to/pcbf_results
```

The default PCBF settings are:

```text
duration threshold = 192 days
prediction-band multiplier = 2.326
required recovery bands = 4 of 5
```

They can be changed when needed:

```bash
python run_pcbf_from_geotiffs.py \
  --input-dir /path/to/downloaded_geotiffs \
  --output-dir /path/to/pcbf_results \
  --duration-threshold-days 192 \
  --z-value 2.326 \
  --required-bands 4
```

The category rule used by the study remains fixed inside the code and is not a
user parameter.

## 4. Output

For each input shard, the script writes:

```text
<original_name>_pcbf.tif
```

Each output keeps the original 392-band structure, CRS, transform, bounds,
dates, coefficients, RMSE and magnitude. Only the `changeProb` value of a break
removed by PCBF is changed from 1 to 0. Therefore, the four outputs remain
spatially aligned and can be opened together as one tile in GIS software.

The output directory also contains:

```text
pcbf_summary.csv
```

This table reports the number of confirmed CCDC breaks before PCBF, the number
removed by PCBF, the number retained, and the number of pixels excluded because
their segment count exceeded the six-segment export depth.
