# CCDC + PCBF in Google Earth Engine

`pcbf_gee_landsat_example.js` is a self-contained Google Earth Engine Code
Editor example that runs CCDC on public Landsat 8/9 Collection 2 Level 2
observations and then applies the Persistence-Constrained Break Filter (PCBF)
to the resulting CCDC array image.

PCBF is a post-processing step. It does not refit CCDC or modify the source
CCDC model coefficients.

## Quick start

1. Open the [Earth Engine Code Editor](https://code.earthengine.google.com/)
   and select an enabled Cloud Project.
2. Create a script and paste the full contents of
   `pcbf_gee_landsat_example.js`.
3. Run the example without changing the point, dates, or 1,500 m buffer.
4. Inspect the retained and rejected break layers on the map and the segment
   arrays printed in the Console.
5. After the example runs, edit `point`, `startDate`, and `endDate` for a new
   location.

The example does not create or start an export task.

## Public PCBF parameters

```javascript
var pcbf = applyPCBF(ccdc, {
  durationThresholdDays: 192,
  zValue: 2.326,
  requiredBands: 4
});
```

- `durationThresholdDays` is the maximum duration used to assess spectral
  change persistence.
- `zValue` is the pre-break model RMSE multiplier used to construct the
  prediction band.
- `requiredBands` is the minimum number of the five PCBF bands that must meet
  the spectral recovery condition on the same date.

PCBF always evaluates Green, Red, NIR, SWIR1, and SWIR2. The existing -200
category rule is fixed inside the function and is not a user parameter.

## Input requirements

`applyPCBF` expects a standard CCDC array image containing:

- `tStart`, `tEnd`, `tBreak`, and `changeProb`;
- `<BAND>_coefs`, `<BAND>_rmse`, and `<BAND>_magnitude` for the five PCBF
  bands;
- eight coefficients on the coefficient axis;
- dates produced with CCDC `dateFormat: 0`;
- reflectance on the x10,000 scale used by the fixed -200 category rule.

The native Earth Engine CCDC result commonly stores `changeProb` as 0/1,
whereas pyxccd commonly uses 0/100. The example treats any positive value as a
confirmed break so that the meaning is consistent without changing the break
confirmation rule.

## Main outputs

- `filteredChangeProb`: break-confirmation array after PCBF;
- `retainedBreakMask`: record-aligned retained-break mask;
- `retainedTBreak`: retained break dates;
- `recoveryDate`: first date satisfying the multispectral recovery condition;
- `persistenceDays`: spectral change persistence in days;
- `category2Rejected` and `recoveryRejected`: rejection diagnostics;
- `persistentRetained` and `terminalRetained`: retention diagnostics;
- `passingBandCountAtRecovery`: number of passing bands on the recovery date.

For large-area processing, first validate the band names, scale, date format,
and array dimensions on a point or small buffer, then process by tile.
