# Feature and database fields

The training CSVs use the 19 predictors below and a `label` column. The annual Parquet uses the same predictors and the four database attributes listed below. The public names are used in the Yajiang example. The workflow also reads the original names in the legacy column. If both names occur in one file, it reads the public column.

`tc` stands for tasseled cap. `6obs` denotes the window from the first through the sixth candidate observation. `x100` identifies values already multiplied by 100 during feature preparation. Pass the stored values directly to the workflow, including the scaling used for training and severity grading.

| Public field | Legacy field | Meaning and stored units |
| --- | --- | --- |
| `elevation_m` | `dem` | Elevation in metres. |
| `slope_deg` | `slope` | Terrain slope in degrees. |
| `seasonal_doy_input` | `sin_doy` | Seasonal predictor as stored. The mapping database uses the untransformed detection day of year. Some legacy training rows use a different convention. Preserve the stored values when reproducing the model. |
| `seasonal_aux_input` | `cos_doy` | Auxiliary seasonal input retained for model compatibility. It is zero in the mapping database and in the bundled training CSVs. |
| `tc_brightness_baseline` | `brightness_c0` | Brightness predicted by the pre-change S-CCD model at the first candidate observation, in stored tasseled-cap units. |
| `tc_greenness_baseline` | `greeness_c0` | Greenness predicted by the pre-change S-CCD model at the first candidate observation, in stored tasseled-cap units. |
| `tc_wetness_baseline` | `wetness_c0` | Wetness predicted by the pre-change S-CCD model at the first candidate observation, in stored tasseled-cap units. |
| `elapsed_days_6obs` | `day_num_c6` | Days elapsed from the first to the sixth candidate observation. |
| `change_angle_6obs_x100` | `angle_c6` | Mean angle between consecutive multiband residual vectors, stored as degrees × 100. |
| `normalized_change_magnitude_6obs_x100` | `norm_cm_c6` | Minimum across the six observations of the sum of squared standardized residuals across five bands, stored as the score × 100, truncated and capped at 32767. This value also determines pixel severity. |
| `tc_brightness_slope_6obs_x100` | `brightness_sl_c6` | Theil–Sen brightness slope, stored as tasseled-cap units per day × 100. |
| `tc_brightness_intercept_6obs` | `brightness_in_c6` | Theil–Sen brightness intercept at elapsed day zero, in stored tasseled-cap units. |
| `tc_brightness_residual_skewness_6obs_x100` | `brightness_sk_c6` | Skewness of brightness residuals from the fit, weighted by elapsed days and stored as skewness × 100. |
| `tc_greenness_slope_6obs_x100` | `greeness_sl_c6` | Theil–Sen greenness slope, stored as tasseled-cap units per day × 100. |
| `tc_greenness_intercept_6obs` | `greeness_in_c6` | Theil–Sen greenness intercept at elapsed day zero, in stored tasseled-cap units. |
| `tc_greenness_residual_skewness_6obs_x100` | `greeness_sk_c6` | Skewness of greenness residuals from the fit, weighted by elapsed days and stored as skewness × 100. |
| `tc_wetness_slope_6obs_x100` | `wetness_sl_c6` | Theil–Sen wetness slope, stored as tasseled-cap units per day × 100. |
| `tc_wetness_intercept_6obs` | `wetness_in_c6` | Theil–Sen wetness intercept at elapsed day zero, in stored tasseled-cap units. |
| `tc_wetness_residual_skewness_6obs_x100` | `wetness_sk_c6` | Skewness of wetness residuals from the fit, weighted by elapsed days and stored as skewness × 100. |

| Public field | Legacy field | Where used and meaning |
| --- | --- | --- |
| `row` | `row` | Parquet. Zero-based raster row on the native HLS grid. |
| `col` | `col` | Parquet. Zero-based raster column on the native HLS grid. |
| `detection_date` | `time` | Parquet. Detection date as an integer `YYYYMMDD`. |
| `detection_doy` | `doy` | Parquet. Detection day of year, from 1 to 365 or 366. This field supplies the day encoded in the output raster. |
| `label` | `label` | Training CSV. Class `1` is non-disturbance anomaly, `2` wildfire, `3` logging, and `4` stress. |
