#!/bin/bash
__doc__="
Clean the repo, removing all compiled files to ensure you are in a fresh state.
"

rm -rf __pycache__
rm -rf dist
rm -rf _skbuild

CLEAN_PYTHON='find . -iname *.pyc -delete ; find . -iname *.pyo -delete ; find . -regex ".*\(__pycache__\|\.py[co]\)" -delete'
bash -c "$CLEAN_PYTHON"

rm -rf htmlcov
rm -rf coverage.xml
rm -rf .coverage
rm -rf tests/htmlcov
rm -rf tests/coverage.xml

# Also clean up any shared libraries
rm -f src/python/pyxccd/_colds_cython.*.so
rm -f src/python/pyxccd/*.so
# Clean up old egg links and such
rm -rf src/python/pyxccd.egg-info
rm -rf pyxccd.egg-info

# Remove debugging MWE
rm -rf dev/src
