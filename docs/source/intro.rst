Introduction
=================

A PYthon library for basic and eXtended COntinuous Change Detection algorithms
------------------------------------------------------------------------------

The Continuous Change Detection and Claasification (CCDC) algorithm has been popular for processing satellite-based time series datasets, particularly for Landsat-based datasets. As a CCDC user, you may already be familiar with the existing CCDC tools such as `pyccd <https://github.com/repository-preservation/lcmap-pyccd>`_ and `gee ccdc <https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc>`_.

**Wait.. so why does the pyxccd package still exist?**

We developed pyxccd mainly for the below purposes:
   
1. **Near real-time monitoring**: this package provides the unique S-CCD algorithm to recursively update model coefficients and detect changes

2. **The latest version of CCDC (COLD) with the highest breakpoint detection accuracy**: The COLD algorithm has been verified with `Zhe's MATLAB version <https://github.com/Remote-Sensing-of-Land-Resource-Lab/COLD>`_.
 
3. **Large-scale time-series processing in the desktop environment**: the core of xccd was coded in C with the superior computing efficiency and small memory usage

5. **Using dataset other than Landsat (such as Sentinel-2, modis)**: pyxccd supports the use of any band combination from any sensor (the flexible mode)

6. **Decomposing time-series signals to unveil inter-season/inter-annual variation (such as phenological shifts)**: S-CCD allows continuously outputting trend and seasonal signal components as "states"