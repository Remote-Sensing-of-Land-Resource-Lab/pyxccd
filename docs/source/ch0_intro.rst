第0课：引言
======================

**作者: Su Ye (remotesensingsuy@gmail.com)**

本教程旨在通过使用多源时间序列数据进行遥感与生态学应用的示例，演示如何使用 ``pyxccd``。

特别感谢 Tianjia Chu, Ronghua Liao, Yingchu Hu, 以及 Yulin Jiang 为教程准备数据集。

准备工作
-----------

首先，请安装 ``pyxccd``。在 Jupyter notebook 的一个单元格中运行：

::

   pip install pyxccd

此外，您还需要安装可视化包：

::

   pip install seaborn

从开发分支下载最新的 `pyxccd <https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd>`__ 源代码，解压后，在 ``/pyxccd/tutorial`` 目录下，目录结构应如下所示：

::

   └── notebooks
   └── datasets

通过示例学习 Pyxccd
-----------------------------

为了展示 pyxccd 的各项功能，本教程准备了多个 notebook 示例，这些示例使用了多变量卫星时间序列数据，涵盖了广泛的应用领域：

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