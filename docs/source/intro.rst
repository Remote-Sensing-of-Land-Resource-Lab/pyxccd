Introduction
=================

A PYthon library for basic and eXtended COntinuous Change Detection algorithms
------------------------------------------------------------------------------

The Continuous Change Detection and Claasification (CCDC) algorithm has been popular for processing satellite-based time series datasets, particularly for Landsat-based datasets. As a CCDC user, you may already be familiar with the existing CCDC tools such as `pyccd <https://github.com/repository-preservation/lcmap-pyccd>`_ and `gee ccdc <https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc>`_.

**Wait.. so why does the pyxccd package still exist?**

We developed pyxccd mainly for the below purposes:
   
1. **Near real-time monitoring**: Implements the unique S-CCD algorithm, which recursively updates model coefficients and enables timely change detection.

2. **Latest CCDC (COLD)**: Integrates the advanced COLD algorithm, offering the highest retrospective breakpoint detection accuracy to date, validated against `Zhe's MATLAB version <https://github.com/Remote-Sensing-of-Land-Resource-Lab/COLD>`_.

3. **Efficient Large-scale time-series processing**: The core of pyxccd is written in C language, ensuring high computational efficiency and low memory usage in the desktop as well as HPC environments.

4. **Flexible multi-sensor support**: Supports arbitrary band combinations from diverse sensors (e.g., Sentinel-2, MODIS, GOSIF, and SMAP) in addition to Landsat.

5. **Continuous time-series signal decomposition **: S-CCD outputs trend and seasonal components as continuous “states”, enabling (a) detection of subtle inter-segment variations such as annual phenological shifts and (b) gap filling that accounts for land cover conversions (temporal breaks).