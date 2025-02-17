We must also manually install the Python build dependencies. This is done via:

.. code:: bash

    # In the future the pyproject.toml should take care of this, but 
    # for now, we must do it manually.
    pip install -r requirements/build.txt
   
While the next command will take care of installing all other non-gdal Python
runtime dependencies, if desired these runtime dependencies can be installed manually via:
``pip install -r requirements.txt`` or via invoking
``pip install -r`` on a specific files in the ``requirements`` subdirectory.

Once GDAL and the Pyton build dependences are installed, pyxccd can be compiled
and installed in development mode via:

.. code:: bash

    export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
    pip install --no-build-isolation --verbose -e .[headless]

This will install the Python dependencies, then compile the C dependencies
in-place, and finally install the pyxccd Python package in development mode.
Note specifying "headless" tells pyxccd which version of opencv to use. The
alternative is "graphics", which is not recommended for fresh builds. Note: you
may need to remove any existing ``_skbuild`` directly if you encounter a build
error.

NOTE: Due to a bug in scikit-build, the editable install link does not point to
the correct path. This can be corrected via:


.. code:: bash

    # Workaround for a scikit-build editable install bug
    REPO_DPATH=$(pwd)
    SITE_DPATH=$(python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib())")
    pyxccd_EGG_LINK_FPATH="$SITE_DPATH"/pyxccd.egg-link
    EASY_INSTALL_FPATH=$SITE_DPATH/easy-install.pth
    echo "$REPO_DPATH/src/python" > "$pyxccd_EGG_LINK_FPATH"
    echo "../../" >> "$pyxccd_EGG_LINK_FPATH"
    mv pyxccd.egg-info ./src/python/
    echo "$REPO_DPATH/src/python" >> "$EASY_INSTALL_FPATH"

Again we note that the above steps and other minor details are consolidated in
the ``run_developer_setup.sh`` script.
