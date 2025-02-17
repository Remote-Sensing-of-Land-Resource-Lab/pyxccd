============
Installation
============

Installation of the pyxsccd package is complicated by its dependency on libgdal
and other C libraries. There are easy installations paths and an advanced
installation path.

Easy installation
=================

.. code-block:: console

    pip install pyxsccd

These wheels are mainly intended to make installation easy for simple
applications, not so much for production. They are not tested for compatibility
with all other binary wheels, conda packages, or QGIS.

Pyxccd 1.0 requires Python 3.9 or higher.

Advanced installation
=====================

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