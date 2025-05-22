PYXCCD
======

|GithubActions| |Pypi| |Downloads| |ReadTheDocs|


A PYthon library for basic and eXtended COntinuous Change Detection
=============================================================================================================================
**Author: Su Ye (remotesensingsuy@gmail.com)**

The Continuous Change Detection and Classification (CCDC) algorithm has been popular for processing satellite-based time series datasets, particularly for Landsat-based datasets. As a CCDC user, you may already be familiar with the existing CCDC tools such as `pyccd <https://github.com/repository-preservation/lcmap-pyccd>`_ and `gee ccdc <https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc>`_.

**Wait.. so why does the pyxccd package still exist?**

We developed pyxccd mainly for the below purposes:
   
1. **Near real-time monitoring**: This package provides the unique S-CCD algorithm to recursively update model coefficients and detect changes;

2. **The latest version of CCDC (COLD)**: The COLD algorithm has the highest breakpoint detection accuracy than the ever, and has been verified with `Zhe's MATLAB version <https://github.com/Remote-Sensing-of-Land-Resource-Lab/COLD>`_;

3. **Large-scale time-series processing in the desktop environment**: the core of pyxccd was coded in C language with the superior computing efficiency and small memory usage;

4. **Using dataset other than Landsat**: pyxccd supports the use of any band combination from any sensor (such as Sentinel-2, modis);

5. **Decomposing time-series signals to unveil inter-annual variation**: S-CCD allows continuously outputting trend and seasonal signal components as "states", allowing detecting inter-segment variations such as yearly phenological shifts


1. Installation
---------------
.. code:: console

   pip install pyxccd

Note: it only supports windows and linux system so far. Please contact the author if you wish to install it in the macOS system.

2. Using pyxccd for pixel-based processing (more see `jupyter examples <tool/notebook/pyxccd_example.ipynb>`_)
----------------------------------------------------------------------------------------------------------------

COLD:

.. code:: python

   from pyxccd import cold_detect
   cold_result = cold_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)

COLD algorithm for any combination of band inputs from any sensor:

.. code:: python

   from pyxccd import cold_detect_flex
   # input a user-defined array instead of multiple lists
   cold_result = cold_detect_flex(dates, np.stack((band1, band2, band3), axis=1), qas, lambda=20,tmask_b1=1, tmask_b2=2)

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
paper (Zhu et al., 2020), and pyxccd reached the same accuracy (27%
omission and 28% commission) showing that the discrepancy doesn't hurt
accuracy. The primary source for the discrepancy is mainly from the
rounding: MATLAB uses float64 precision, while pyxccd chose float32 to
save the run-time computing memory and boost efficiency.

Q2: how much time for production of a tile-based disturbance map (5000*5000 pixels) using pyxccd?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Re: I tested it in UCONN HPC environment (200 EPYC7452 cores): for
processing a 40-year Landsat ARD tile (1982-2021), the stacking
typically takes 15 mins; per-pixel COLD processing costs averagely 1
hour, per-pixel S-CCD processing costs averagely 0.5
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
