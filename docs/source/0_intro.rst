Lesson 0: introduction
======================

**Author: Su Ye (remotesensingsuy@gmail.com)**

This tutorial was made to present the examples for using ``pyxccd`` for
multiple remote sensing and ecological applications using multi-source
time-series datasets.

Special thanks to Tianjia Chu, Ronghua Liao, Yingchu Hu, and Yulin Jiang
for preparing tutorial datasets.

Preparation
-----------

First, please install ``pyxccd``. In a Jupyter notebook cell, run:

::

   pip install pyxccd

Additionally, you need to install the visualization package:

::

   pip install seaborn

Download the recent source codes of
`pyxccd <https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd>`__
in the devel branch, unzipped it, and under the directory
``/pyxccd/tutorial``, the directory should look like:

::

   └── notebooks
   └── datasets

Learning Pyxccd with Examples
-----------------------------

To illustrate the utilities of pyxccd, we prepared multiple notebook
examples using multivariate satellite-based time series across a wide
range of applications in this tutorial:

+---------+------------+---------------+------------+------------+------------+---------+
| No.     | Topics     | Applications  | Location   | Time       | Resolution | Density |
|         |            |               |            | series     |            |         |
+=========+============+===============+============+============+============+=========+
| 1       | Break      | Forest fire   | Sichuan,   | HLS2.0     | 30m        | 2-3     |
|         | detection  |               | China      |            |            | days    |
+---------+------------+---------------+------------+------------+------------+---------+
| 2       | Parameter  | Forest        | CO & MA,   | Landsat    | 30m        | 8-16    |
|         | selection  | Insects       | United     |            |            | days    |
|         |            |               | States     |            |            |         |
+---------+------------+---------------+------------+------------+------------+---------+
| 3       | Flexible   | Crop dynamics | Henan,     | Sentinel-2 | 10m        | 5 days  |
|         | choice for |               | China      |            |            |         |
|         | inputs     |               |            |            |            |         |
+---------+------------+---------------+------------+------------+------------+---------+
| 4       | Tile-based | General       | Zhejiang,  | HLS2.0     | 30m        | 2-3     |
|         | processing | disturbances  | China      |            |            | days    |
+---------+------------+---------------+------------+------------+------------+---------+
| 5       | State      | Greening      | Tibet,     | MODIS      | 500m       | 16 days |
|         | analysis 1 |               | China      |            |            |         |
+---------+------------+---------------+------------+------------+------------+---------+
| 5       | State      | Precipitation | Arctic     | GPCP       | 2.5°       | Monthly |
|         | analysis 2 | seasonality   |            |            |            |         |
+---------+------------+---------------+------------+------------+------------+---------+
| 6       | Anomalies  | Agricultural  | Rajasthan, | GOSIF      | 0.05°      | 8 days  |
|         | vs. breaks | drought       | India      |            |            |         |
+---------+------------+---------------+------------+------------+------------+---------+
| 7       | Near       | Forest        | Sichuan,   | HLS2.0     | 30m        | 2-3     |
|         | real-time  | logging       | China      |            |            | days    |
|         | monitoring |               |            |            |            |         |
+---------+------------+---------------+------------+------------+------------+---------+
| 8       | Gap        | Soil moisture | Henan,     | FY3B       | 25km       | Daily   |
|         | filling    |               | China      |            |            |         |
+---------+------------+---------------+------------+------------+------------+---------+

Note:

(1) The tutorial primarily provides pixel-based time series examples for
    educational purposes; however, in practical applications, analyses
    are typically performed on image-based datasets. In Lesson 4, we
    will specifically demonstrate the procedures for applying pyxccd to
    real-world image-based time series;

(2) All date columns in the tutorial are formatted as Gregorian
    proleptic ordinal numbers, representing the number of days elapsed
    since 0001-01-01. Users can convert the ordinal date format to
    human-readable date format using the Python function
    ``datetime.date.fromordinal()``.
