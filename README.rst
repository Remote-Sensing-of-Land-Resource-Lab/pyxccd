PYXCCD
======

|GithubActions| |Pypi| |Downloads| 

.. .. TODO: uncomment these after docs / pypi / coverage are online
.. .. |ReadTheDocs| |Codecov| 

A PYthon library for basic and eXtended COntinuous Change Detection algorithms
=============================================================================================================================

This library mainly provides the below two algorithms and their toolsets for a large-scale time-series processing in an local environment: 

1. Stochastic Continuous Change Detection (S-CCD, a near real-time and short-memory implementation of COLD)

2. COntinuous monitoring of Land Disturbance (COLD): the latest CCDC algorithm for retrospective time-series analysis 

 
Aside from popular CCDC toolsets such as `pyccd <https://github.com/repository-preservation/lcmap-pyccd>`_ and `gee ccdc <https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc>`_, you may consider using pyxccd for the below situations:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Near real-time monitoring: this package provides unique S-CCD algorithm to recursively update model coefficients and detect changes

2. Large-scale time-series processing in the desktop environment: the core of xccd was coded in C with the superior computing efficiency and small memory usage

3. The latest version of CCDC (COLD) with the highest breakpoint detection accuracy: The basic CCDC has been verified with `Zhe's MATLAB version <https://github.com/Remote-Sensing-of-Land-Resource-Lab/COLD>`_.

4. Using dataset other than Landsat (such as Sentinel-2, modis): pyxccd supports the use of any band combination from any sensor (the flexible mode)

5. Decomposing time-series signals to unveil inter-season/inter-annual variation (such as phenological shifts): S-CCD allows continuously outputting trend and seasonal signal components as "states"

           

1. Pre-work
-----------
   
Clone github repo to your local code directory for the first use:

.. code:: bash

   git clone https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd.git

Or you call pull the recent repo if you want to update the existing
pyxccd repo:

.. code:: bash

   git pull origin devel:devel

2. Installation
---------------

The steps to install this library in development mode are consolidated
into a single script: ``run_developer_setup.sh``.  On debian-based systems,
this will install all of the developer requirements and ensure you are setup
with a working opencv-python-headless Python modules, as well as other
requirements and then it will compile and install pyxccd in editable
development mode.


The following is an overview of these details and alternative choices that
could be made.

2.1 Install Required Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZLIB, GSL libraries are required.

For Ubuntu/Debian systems, they can be installed via:

.. code:: bash

   sudo apt-get update
   sudo apt-get install build-essential  -y
   sudo apt-get install zlib1g-dev -y
   sudo apt-get install gfortran -y
   sudo apt-get install libgsl-dev -y

On CentOS systems run:

.. code:: bash

   sudo apt-get install gcc gcc-c++ make  -y
   sudo apt-get install zlib-devel -y
   sudo apt-get install gcc-gfortran -y
   # Yum provides an gsl 1.5, but we need 2.7
   # sudo apt-get install gsl-devel -y
   curl https://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz  > gsl.tar.gz && tar xfv gsl.tar.gz && cd gsl-2.7.1 && ./configure --prefix=/usr --disable-static && make && make install

2.2 Compile and Install pyxccd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following instructure assume you are inside a Python virtual environment
(e.g. via conda or pyenv). 

.. code:: bash

    # Install required packages
    pip install -r requirements.txt
    
.. code:: bash

Additionally, to access the ``cv2`` module, pyxccd will require either
``opencv-python`` or ``opencv-python-headless``, which are mutually exclusive.
This is exposed as optional dependencies in the package via either "graphics"
or "headless" extras.  Headless mode is recommended as it is more compatible
with other libraries. These can be obtained manually via:

.. code:: bash

    pip install -r requirements/headless.txt
    
    # XOR (choose only one!)

    pip install -r requirements/graphics.txt


**Option 1: Install in development mode**

For details on installing in development mode see the
`developer install instructions <docs/source/developer_install.rst>`_.

We note that all steps in the above document and other minor details are
consolidated in the ``run_developer_setup.sh`` script.


**Option 2: Build and install a wheel**

Scikit-build will invoke CMake and build everything. (you may need to
remove any existing ``_skbuild`` directory).

.. code:: bash

   python -m build --wheel .

Then you can pip install the wheel (the exact path will depend on your system
and version of python).

.. code:: bash

   pip install dist/pyxccd-0.1.0-cp38-cp38-linux_x86_64.whl


You can also use the ``build_wheels.sh`` script to invoke cibuildwheel to
produce portable wheels that can be installed on different than they were built
on. You must have docker and cibuildwheel installed to use this.


**Option 3: build standalone binaries with CMake by itself (recommended
for C development)**

.. code:: bash

   mkdir -p build
   cd build
   cmake ..
   make 

**Option 4: Use a docker image.**

This repo provides dockerfiles that illustrate a reproduceable method for
compling and installing pyxccd. See `dockerfiles/README.rst
<dockerfiles/README.rst>`__ for details.

3. Using pyxccd for pixel-based processing (more see `jupyter examples <tool/notebook/pyxccd_example.ipynb>`)
------------------------------------------

COLD:

.. code:: python

   from pyxccd import cold_detect
   cold_result = cold_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)

COLD algorithm for any combination of band inputs from any sensor:

.. code:: python

   from pyxccd import cold_detect
   # input a user-defined array instead of multiple lists
   cold_result = cold_detect_flex(dates, np.stack((band1, band2, band3), axis=1), qas, tmask_b1=1, tmask_b2=2)

S-CCD:

.. code:: python

   # require offline processing for the first time 
   from pyxccd import sccd_detect, sccd_update
   sccd_pack = sccd_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)

   # then use sccd_pack to do recursive and short-memory NRT update
   sccd_pack_new = sccd_update(sccd_pack, dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)

Q&A
~~~

Q1: Has pyxccd been verified with original Matlab codes?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Re: yes, multiple rounds of verification have been done. Comparison
based on two testing tiles shows that pyxccd and Matlab version have
smaller than <2% differences for breakpoint detection and <2%
differences for harmonic coefficients; the accuracy of pyxccd was also
tested against the same reference dataset used in the original COLD
paper (Zhu et al., 2020), and pyxccd reached the same accuracy (27%
omission and 28% commission) showing that the discrepancy doesn’t hurt
accuracy. The primary source for the discrepancy is mainly from the
rounding: MATLAB uses float64 precision, while pyxccd chose float32 to
save the run-time computing memory and boost efficiency.

Q2: how much time for production of a tile-based disturbance map (5000*5000 pixels) using pyxccd?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Re: I tested it in UCONN HPC environment (200 EPYC7452 cores): for
processing a 40-year Landsat ARD tile (1982-2021), the stacking
typically takes 15 mins; per-pixel COLD processing costs averagely 1
hour; exporting maps needs 7 mins.

4. Citations
------------

If you make use of the algorithms in this repo (or to read more about them),
please cite (/see) the relevant publications from the following list:

`[COLD] <https://www.sciencedirect.com/science/article/am/pii/S0034425719301002>`_ 
Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., &
Zhou, C. (2020). Continuous monitoring of land disturbance based on
Landsat time series. *Remote Sensing of Environment*, *238*, 111116.

`[S-CCD] <https://www.sciencedirect.com/science/article/pii/S003442572030540X>`_
Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time
approach for monitoring forest disturbance using Landsat time series:
Stochastic continuous change detection. *Remote Sensing of Environment*,
*252*, 112167.

The recent applications of S-CCD could be found in `CONUS Land Watcher <https://gers.users.earthengine.app/view/nrt-conus>`_


.. |Codecov| image:: https://codecov.io/github/Remote-Sensing-of-Land-Resource-Lab/pyxccd/badge.svg?branch=devel&service=github
   :target: https://codecov.io/github/Remote-Sensing-of-Land-Resource-Lab/pyxccd?branch=devel
.. |Pypi| image:: https://img.shields.io/pypi/v/pyxccd.svg
   :target: https://pypi.python.org/pypi/pyxccd
.. |Downloads| image:: https://img.shields.io/pypi/dm/pyxccd.svg
   :target: https://pypistats.org/packages/pyxccd
.. |ReadTheDocs| image:: https://readthedocs.org/projects/pyxccd/badge/?version=latest
    :target: http://pyxccd.readthedocs.io/en/latest/
.. |GithubActions| image:: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/actions/workflows/tests.yml/badge.svg?branch=devel
    :target: https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd/actions?query=branch%3Adevel
