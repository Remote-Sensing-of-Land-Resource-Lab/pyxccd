============
Installation
============

Installation of the pyxsccd package is complicated by its dependency on libgdal
and other C libraries. There are easy installations paths and an advanced
installation path.

1. Easy installation
=================

.. code-block:: console

    pip install pyxsccd

These wheels are mainly intended to make installation easy for simple
applications, not so much for production. They are not tested for compatibility
with all other binary wheels, conda packages, or QGIS.

Pyxccd 1.0 requires Python 3.8 or higher.

2. Advanced installation
=====================
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

2.2 Install required python packages
The following instructure assume you are inside a Python virtual environment
(e.g. via conda or pyenv). 

.. code:: bash

    # Install required packages
    pip install -r requirements.txt

Additionally, to access the ``cv2`` module, pyxccd will require either
``opencv-python`` or ``opencv-python-headless``, which are mutually exclusive.
This is exposed as optional dependencies in the package via either "graphics"
or "headless" extras.  Headless mode is recommended as it is more compatible
with other libraries. These can be obtained manually via:

.. code:: bash

    pip install -r requirements/headless.txt
    
    # XOR (choose only one!)

    pip install -r requirements/graphics.txt

2.3 Install pyxccd
**Option 1: Install in development mode**

For details on installing in development mode see the
`developer install instructions <docs/source/developer_install.rst>`_.

We note that all steps in the above document and other minor details are
consolidated in the ``run_developer_setup.sh`` script.

.. code:: bash

    bash run_developer_setup.sh


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


**Option 3: build standalone binaries with CMake by itself (recommended for C development)**

.. code:: bash

   mkdir -p build
   cd build
   cmake ..
   make 