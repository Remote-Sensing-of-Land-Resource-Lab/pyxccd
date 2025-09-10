PYXCCD
======

|GithubActions| |Pypi| |Downloads| |ReadTheDocs|


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

5. **Continuous time-series signal decomposition**: S-CCD outputs trend and seasonal components as continuous “states”, enabling (a) detection of subtle inter-segment variations such as annual phenological shifts and (b) gap filling that accounts for land cover conversions (temporal breaks).


1. Installation
---------------
.. code:: console

   pip install pyxccd

Note: the installation has been cross-platform (Windows, Linux and MacOS). Contact the author (remotesensingsuy@gmail.com) if you have problems for installation 

2. Using pyxccd for pixel-based processing
----------------------------------------------------------------------------------------------------------------

COLD:

.. code:: python

   from pyxccd import cold_detect
   cold_result = cold_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)

COLD algorithm for any combination of band inputs from any sensor:

.. code:: python

   from pyxccd import cold_detect_flex
   # input a user-defined array instead of multiple lists
   cold_result = cold_detect_flex(dates, np.stack((band1, band2, band3), axis=1), qas, lambda=20,tmask_b1_index=1, tmask_b2_index=2)

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

3. Documentation
----------------
API documents: `readthedocs <https://pyxccd.readthedocs.io/en/latest>`_

Tutorial: under development

4. Citations
------------

If you make use of the algorithms in this repo (or to read more about them),
please cite (/see) the relevant publications from the following list:

`[S-CCD] <https://www.sciencedirect.com/science/article/pii/S003442572030540X>`_
Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time
approach for monitoring forest disturbance using Landsat time series:
Stochastic continuous change detection. *Remote Sensing of Environment*,
*252*, 112167.

`[COLD] <https://www.sciencedirect.com/science/article/am/pii/S0034425719301002>`_ 
Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., &
Zhou, C. (2020). Continuous monitoring of land disturbance based on
Landsat time series. *Remote Sensing of Environment*, *238*, 111116.

The recent applications of S-CCD could be found in `CONUS Land Watcher <https://gers.users.earthengine.app/view/nrt-conus>`_

Q&A
---

Q1: Has pyxccd been verified with original Matlab codes?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Re: yes, multiple rounds of verification have been done. Comparison
based on two testing tiles shows that pyxccd and Matlab version have
smaller than <2% differences for breakpoint detection and <2%
differences for harmonic coefficients; the accuracy of pyxccd was also
tested against the same reference dataset used in the original COLD
paper (Zhu et al., 2020), and COLD in pyxccd reached the same accuracy (27%
omission and 28% commission) showing that the discrepancy doesn't hurt
accuracy. The primary source for the discrepancy is mainly from the
rounding: MATLAB uses float64 precision, while pyxccd chose float32 to
save the run-time computing memory and boost efficiency.

Q2: how much time for production of a tile-based disturbance map (5000*5000 pixels) using pyxccd?
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
