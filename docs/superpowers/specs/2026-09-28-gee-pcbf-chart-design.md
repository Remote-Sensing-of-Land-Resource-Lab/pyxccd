# GEE PCBF Time-Series Chart Design

## Goal

Present the public Landsat CCDC + PCBF example directly in the Earth Engine Code Editor with a clear, publication-style SWIR2 time-series chart.

## Scope

- Preserve `applyPCBF`, its four methodological settings, and all return values.
- Replace verbose Chinese comments and Console labels with concise professional English.
- Plot valid SWIR2 observations, the piecewise CCDC harmonic trajectory, retained breaks, and PCBF-removed candidates in one interactive GEE chart.
- Keep break dates available through chart tooltips and a compact Console summary.
- Retain map previews, with concise English layer names.
- Do not add exports, assets, or changes to CCDC/PCBF detection rules.

## Visual design

- Observations: blue circles, no connecting line.
- CCDC trajectory: dark-gray continuous line sampled every eight days.
- Retained breaks: red circles.
- PCBF-removed candidates: gray diamonds.
- White background, light gridlines, top legend, English axes, and SWIR2 values on the existing x10,000 scale.

## Data flow

The existing CCDC array image remains the sole model source. For each regularly spaced display date, the visualization helper identifies the active segment and evaluates its SWIR2 harmonic coefficients. Separate features are created for valid observations and break markers, then merged into one feature collection for `ui.Chart.feature.byFeature`.

## Verification

Source-level tests will confirm that the public PCBF parameters remain unchanged, prohibited verbose comments are removed, chart series and styles are present, and the two script copies remain identical. JavaScript syntax will be checked locally; the saved Code Editor example remains map-preview only and starts no export task.
