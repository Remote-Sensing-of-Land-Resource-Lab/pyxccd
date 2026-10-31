PYXCCD
======

|GithubActions| |Pypi| |badge_downloads| |ReadTheDocs|

.. |badge_downloads| image:: https://pepy.tech/badge/pyxccd
   :target: https://pepy.tech/project/pyxccd
   :alt: Downloads

A PYthon library for latest and eXtended Continuous Change Detection
=============================================================================================================================
**Author: Su Ye (remotesensingsuy@gmail.com)**

The Continuous Change Detection and Classification (CCDC) algorithm has been popular for processing satellite-based time series datasets, particularly for Landsat-based datasets. As a CCDC user, you may already be familiar with the existing CCDC tools such as `pyccd <https://github.com/repository-preservation/lcmap-pyccd>`_ and `gee ccdc <https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc>`_.

**Wait.. so why does the pyxccd package still exist?**

We developed pyxccd mainly for the below purposes:
   
1. **Near real-time monitoring**: Implements the unique S-CCD algorithm, which recursively updates model coefficients and enables timely change detection.

2. **Latest CCDC (COLD)**: Integrates the advanced COLD algorithm, offering the highest retrospective breakpoint detection accuracy to date, validated against `Zhe's MATLAB version <https://github.com/Remote-Sensing-of-Land-Resource-Lab/COLD>`_.


3. **Efficient Large-scale time-series processing**: The core of pyxccd is written in C language, ensuring high computational efficiency and low memory usage in the desktop as well as HPC environments.

4. **Flexible multi-sensor support**: Supports arbitrary band combinations from diverse sensors (e.g., Sentinel-2, MODIS, GOSIF, and SMAP) in addition to Landsat.

5. **State-space model incoporation**: S-CCD allows modeling trend and seasonal signals as time-varying variables (namely “states”) guided by break detection, enabling (a) characterization of subtle inter-segment variations (e.g., phenological shifts) and (b) gap filling that accounts for land cover conversions (temporal breaks).


1. Installation
---------------
.. code:: console

   pip install pyxccd

Note: the installation has been cross-platform (Windows, Linux and MacOS), python >= 3.9. Contact the author (remotesensingsuy@gmail.com) if you have problems for installation 

2. Using pyxccd for pixel-based processing
----------------------------------------------------------------------------------------------------------------

COLD:

.. code:: python

   from pyxccd import cold_detect
   import pandas as pd
   data = pd.read_csv('tutorial/datasets/1_hls_sc.csv')
   dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, sensor = data.to_numpy().copy().T
   cold_result = cold_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)

COLD algorithm for any combination of band inputs from any sensor:

.. code:: python

   from pyxccd import cold_detect_flex
   # input a user-defined array instead of multiple lists
   cold_result = cold_detect_flex(dates, np.stack((reds, nirs, swir1s), axis=1), qas, lambda=20,tmask_b1_index=1, tmask_b2_index=2)

S-CCD:

.. code:: python

   # require offline processing for the first time 
   from pyxccd import sccd_detect, sccd_update
   sccd_pack = sccd_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, qas)

   # then use sccd_pack to do recursive and short-memory NRT update
   sccd_pack_new = sccd_update(sccd_pack, dates, blues, greens, reds, nirs, swir1s, swir2s, qas)

S-CCD for outputting continuous seasonal and trend states:

.. code:: python
   
   # open state output (state_ensemble) by setting state_intervaldays as a non-zero value
   sccd_result, state_ensemble = sccd_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, qas, state_intervaldays=1)


3. Tutorials
----------------

.. list-table::
   :header-rows: 1
   :widths: 5 25 25 25 15 15 15

   * - No.
     - Topics
     - Applications
     - Location
     - Time series
     - Resolution
     - Density
   * - 0
     - `Introduction`_
     - 
     - 
     - 
     - 
     - 
   * - 1
     - `Break detection`_
     - Forest fire
     - Sichuan, China
     - HLS2.0
     - 30 m
     - 2–3 days
   * - 2
     - `Parameter selection`_
     - Forest insects
     - CO & MA, United States
     - Landsat
     - 30 m
     - 8–16 days
   * - 3
     - `Flexible choice for inputs`_
     - Crop dynamics
     - Henan, China
     - Sentinel-2
     - 10 m
     - 5 days
   * - 4
     - `Tile-based processing`_
     - General disturbances
     - Zhejiang, China
     - HLS2.0
     - 30 m
     - 2–3 days
   * - 5-1
     - `State analysis 1`_
     - Greening
     - Tibet, China
     - MODIS
     - 500 m
     - 16 days
   * - 5-2
     - `State analysis 2`_
     - Precipitation seasonality
     - Arctic
     - GPCP
     - 2.5°
     - Monthly
   * - 6
     - `Anomalies vs. breaks`_
     - Agricultural drought
     - Rajasthan, India
     - GOSIF
     - 0.05°
     - 8 days
   * - 7
     - `Near real-time monitoring`_
     - Forest logging
     - Sichuan, China
     - HLS2.0
     - 30 m
     - 2–3 days
   * - 8
     - `Gap filling`_
     - Soil moisture
     - Henan, China
     - FY3B
     - 25 km
     - Daily
   * - 9
     - `Phenology`_
     - Phenology extaction
     - MI, USA
     - HLS2.0
     - 30 m
     - 2–3 days
   * - 10
     - `Evaluation`_
     - Efficiency & accuracy test
     - CONUS
     - Landsat
     - 30 m
     - 8–16 days
   * - 11
     - `Break consistency`_
     - Postprocessing for consistency
     - Example pixel
     - Landsat
     - 30 m
     - 8–16 days

.. _Introduction: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/0_intro.ipynb
.. _Break detection: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/1_break_detection_fire_hls.ipynb
.. _Parameter selection: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/2_parameter_selection_insect_landsat.ipynb
.. _Flexible choice for inputs: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/3_flexible_inputs_crop_sentinel2.ipynb
.. _Tile-based processing: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/4_tile_processing_general_hls.ipynb
.. _State analysis 1: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/5_state_analysis_greenning&precipitation_coarse.ipynb
.. _State analysis 2: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/5_state_analysis_greenning&precipitation_coarse.ipynb
.. _Anomalies vs. breaks: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/6_anomalies_break_drought_gosif.ipynb
.. _Near real-time monitoring: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/7_near_realtime_logging_hls.ipynb
.. _Gap filling: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/8_gapfilling_general_FY3B.ipynb
.. _Phenology: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/9_phenology_extraction_hls.ipynb
.. _Evaluation: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/10_performance_eval_general_landsat.ipynb
.. _Break consistency: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/blob/devel/tutorials/notebooks/11_pcbf.ipynb

Tutorial datasets: `Github link <https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/tree/devel/tutorials/datasets>`_, `夸克硬盘 (for China mainland) <https://pan.quark.cn/s/091eda7c76ff>`_

4. GUI
------------

We provided GUI to quickly test S-CCD or COLD algorithms using point-based time series formated in CSV or EXCEL:

`夸克硬盘 (for China mainland) <https://pan.quark.cn/s/c57a14eeb7fa>`_

`Dropbox (for outside China mainland) <https://www.dropbox.com/scl/fo/7l3bzkkwan51t5yqee0a0/AMYHYgxYQov3UxEVamZWd6M?rlkey=aon58nep5r5jppz1l7ky5ou2s&st=gyvmsax5&dl=0>`_


5. Documentation
----------------
API documents: `readthedocs <https://pyxccd.readthedocs.io/en/latest>`_

Technical details and performance: `pyxccd paper <https://eartharxiv.org/repository/view/12261/>`_

6. Citations
------------

If you make use of the algorithms in this repo (or to read more about them),
please cite (/see) the relevant publications from the following list:

`[PYXCCD] <https://eartharxiv.org/repository/view/12261/>`_
Ye, S., & Hu, Y. (2026). Pyxccd: An Efficient Python Package for Break-aware Time Series Analysis of Earth Observation Data.

`[S-CCD] <https://www.sciencedirect.com/science/article/pii/S003442572030540X>`_
Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time
approach for monitoring forest disturbance using Landsat time series:
Stochastic continuous change detection. *Remote Sensing of Environment*,
*252*, 112167.

`[COLD] <https://www.sciencedirect.com/science/article/am/pii/S0034425719301002>`_ 
Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., &
Zhou, C. (2020). Continuous monitoring of land disturbance based on
Landsat time series. *Remote Sensing of Environment*, *238*, 111116.


7. Applications
----------------
The following publications have used `pyxccd` in their research.


Publications by the Development Team:

```

Liao, R., Guo, C., Chen, L., Jiang, Y., Tao, Y., Liao, J., ... & Ye, S.
(2026). Near-real-time mapping for causal agents of forest disturbances in
China using harmonized Landsat and Sentinel-2 dataset. *Remote Sensing of
Environment*, **342**, 115461.

Ye, S., & Hu, Y. (2026). *Pyxccd: An Efficient Python Package for Break-Aware
Time Series Analysis of Earth Observation Data.*

Ye, S., Zhu, Z., & Suh, J. W. (2024). Leveraging past information and
machine learning to accelerate land disturbance monitoring. *Remote Sensing of
Environment*, **305**, 114071.

Ye, S., Zhu, Z., & Cao, G. (2023). Object-based continuous monitoring of land
disturbances from dense Landsat time series. *Remote Sensing of Environment*,
**287**, 113462.

Ye, S., Rogan, J., Zhu, Z., Hawbaker, T. J., Hart, S. J., Andrus, R. A.,
... & Kulakowski, D. (2021). Detecting subtle change from dense Landsat time
series: Case studies of mountain pine beetle and spruce beetle disturbance.
*Remote Sensing of Environment*, **263**, 112560.

Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time
approach for monitoring forest disturbance using Landsat time series:
Stochastic continuous change detection. *Remote Sensing of Environment*,
**252**, 112167.


Applications and Studies by Other Researchers:
```

Schiller, C., & Fassnacht, F. E. (2026). Comparative study of near real-time
monitoring algorithms for early detection of bark beetle infestations in
Germany with Sentinel-2. *European Journal of Remote Sensing*, 59(1),
2662660.

Schraik, D., Ursin, A., & Packalen, P. (2026). Detection of non-stand-replacing
cuttings using satellite image time series with a Kalman filter. *IEEE Journal
of Selected Topics in Applied Earth Observations and Remote Sensing*.

Wang, S., & Wang, L. (2026). Quantifying high spatiotemporal impervious surface
change with time series PlanetScope imagery. *Remote Sensing Applications:
Society and Environment*, 101890.

Hong, F., Hedges, S. B., Yang, Z., Suh, J. W., Qiu, S., Timyan, J., & Zhu, Z.
(2025). Decoding primary forest changes in Haiti and the Dominican Republic
using Landsat time series. *Remote Sensing of Environment*, **318**, 114590.

Abdallah, M., Ali, E., Ding, X., & Wu, S. (2025). An improved COLD approach
for monitoring construction dynamics using HLS and LULC data: A case study in
the New Capital of Egypt. *ISPRS Annals of the Photogrammetry, Remote Sensing
and Spatial Information Sciences*, 3--10.

Healey, N. C., Barber, C. P., Smith, K., Mital, R., Brown, J. F., & Robison,
C. (2025). Resiliency of land change monitoring efforts to input data
resampling. *Frontiers in Remote Sensing*, **6**, 1570580.

Suh, J. W., Zhu, Z., & Zhao, Y. (2024). Monitoring construction changes using
dense satellite time series and deep learning. *Remote Sensing of Environment*,
**309**, 114207.

Mota, F. B. D. S., Ferreira, K. R., & Escada, M. I. S. (2024). Evaluating
forest disturbance detection methods based on satellite image time series for
Amazon deforestation alerts. *The International Archives of the
Photogrammetry, Remote Sensing and Spatial Information Sciences*, **48**,
357--364.

*Last updated: October 31, 2026.*


Q&A
---

Q1: Has pyxccd been verified?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Re: Multiple rounds of verification have been conducted. A comparison based on two testing tiles indicates that differences between pyxccd and the MATLAB implementation are minimal, with discrepancies of less than 2% in both breakpoint detection and harmonic coefficients. Furthermore, the accuracy of pyxccd was evaluated against the same reference dataset used in the original COLD study (Zhu et al., 2020). The results demonstrate that COLD in pyxccd achieves equivalent accuracy (27% omission and 28% commission), confirming that the observed discrepancies do not compromise performance. The primary source of the discrepancy stems from numerical precision: MATLAB employs float64, whereas pyxccd uses float32 to reduce memory consumption and improve computational efficiency.

Q2: why was the disturbance detection accuracy of COLD or S-CCD unexpectedly lower (e.g., UA or PA < 60%)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Re: there are a variety of reasons. A common reason is that all detected breaks were treated as disturbance events without filtering out recovery-related breaks. 
COLD and S-CCD detect statistically significant changes in the time series, but not every detected break represents a disturbance. A pixel may first experience recovery, 
regrowth, or restoration, producing another valid breakpoint. If these recovery-related breaks are also counted as disturbances, the number of false positives can increase
substantially, leading to high commission error. When generating a disturbance map from COLD results, use **pyxccd.utils.getcategory_cold()** or **pyxccd.utils.getcategory_cold_flex** (flexible mode) to classify the detected breaks and  retain only disturbance-related categories. For S-CCD results, use the corresponding getcategory_sccd() function.

If accuracy is still low after break classification, also check:

(1) whether the reference disturbance definition matches what COLD/S-CCD detects;
(2) temporal tolerance between the detected break date and reference disturbance date;
(3) cloud, shadow, snow, and other preprocessing errors;
(4) parameter settings and model initialization;
(5) subtle or gradual disturbances that may not produce strong spectral changes.



Q3: how much time for production of a tile-based disturbance map (5000*5000 pixels) using pyxccd?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Re: I tested COLD in UCONN HPC environment (200 EPYC7452 cores): for
processing a 40-year Landsat ARD tile (1982-2021), the stacking
typically takes 15 mins; per-pixel COLD processing costs averagely 1
hour, while per-pixel S-CCD processing costs averagely 0.5
hour; exporting maps needs 7 mins. 


.. |Codecov| image:: https://codecov.io/github/Remote-Sensing-of-Land-Resource-Lab/pyxccd/badge.svg?branch=devel&service=github
   :target: https://codecov.io/github/Remote-Sensing-of-Land-Resource-Lab/pyxccd?branch=devel
.. |Pypi| image:: https://img.shields.io/pypi/v/pyxccd.svg
   :target: https://pypi.python.org/pypi/pyxccd
.. |Downloads| image:: https://img.shields.io/pypi/dm/pyxccd.svg
   :target: https://pypistats.org/packages/pyxccd
.. |ReadTheDocs| image:: https://readthedocs.org/projects/pyxccd/badge/?version=latest
    :target: http://pyxccd.readthedocs.io/en/latest/
.. |GithubActions| image:: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/actions/workflows/main.yml/badge.svg?branch=devel
    :target: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/actions?query=branch%3Adevel

