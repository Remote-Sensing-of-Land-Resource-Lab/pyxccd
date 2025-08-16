#!/bin/bash
__doc__="
Script that does its best to ensure a robust installation of pyxccd in
development mode on macOS and Debian/Ubuntu Linux.
"

if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "WARNING: NOT INSIDE OF A Python VIRTUAL_ENV. This script may not run correctly"
fi

apt_ensure(){
    __doc__="
    Checks to see if the packages are installed and installs them if needed.

    The main reason to use this over normal apt install is that it avoids sudo
    if we already have all requested packages.

    Args:
        *ARGS : one or more requested packages 

    Example:
        apt_ensure git curl htop 

    Ignore:
        REQUESTED_PKGS=(git curl htop) 
    "
    # Note the $@ is not actually an array, but we can convert it to one
    # https://linuxize.com/post/bash-functions/#passing-arguments-to-bash-functions
    ARGS=("$@")
    MISS_PKGS=()
    HIT_PKGS=()
    for PKG_NAME in "${ARGS[@]}"
    do
        #apt_ensure_single $EXE_NAME
        RESULT=$(dpkg -l "$PKG_NAME" | grep "^ii *$PKG_NAME")
        if [ "$RESULT" == "" ]; then 
            echo "Do not have PKG_NAME='$PKG_NAME'"
            # shellcheck disable=SC2268,SC2206
            MISS_PKGS=(${MISS_PKGS[@]} "$PKG_NAME")
        else
            echo "Already have PKG_NAME='$PKG_NAME'"
            # shellcheck disable=SC2268,SC2206
            HIT_PKGS=(${HIT_PKGS[@]} "$PKG_NAME")
        fi
    done

    if [ "${#MISS_PKGS}" -gt 0 ]; then
        sudo apt install -y "${MISS_PKGS[@]}"
    else
        echo "No missing packages"
    fi
}

brew_ensure(){
    __doc__="
    Checks to see if the Homebrew packages (formulae) are installed and
    installs them if needed.
    Args:
        *ARGS : one or more requested packages
    "
    ARGS=("$@")
    MISS_PKGS=()
    for PKG_NAME in "${ARGS[@]}"
    do
        if ! brew list --formula -1 | grep -q "^${PKG_NAME}\$"; then
            echo "Do not have PKG_NAME='$PKG_NAME'"
            MISS_PKGS+=("$PKG_NAME")
        else
            echo "Already have PKG_NAME='$PKG_NAME'"
        fi
    done

    if [ "${#MISS_PKGS[@]}" -gt 0 ]; then
        echo "Attempting to install missing packages: ${MISS_PKGS[*]}"
        brew install "${MISS_PKGS[@]}"
    else
        echo "No missing Homebrew packages"
    fi
}


###  ENSURE DEPENDENCIES ###

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install it first."
        echo "See: https://brew.sh/"
        exit 1
    fi
    brew_ensure gsl gcc cmake
elif command -v apt &> /dev/null; then
    # Debian/Ubuntu
    apt_ensure build-essential zlib1g-dev libgsl-dev gfortran
else
    echo "
    WARNING: Check and install of system packages is currently only supported
    on Debian Linux and macOS. For other systems, you will need to verify that
    GSL, a C compiler, and a Fortran compiler are installed.
    "
fi



###  CLEANUP PREVIOUS INSTALLS ###


# Reference: https://stackoverflow.com/questions/59895/bash-script-dir
REPO_DPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Check that specific files exist to indicate we aren't in the wrong place.
# Exit if we are.
echo "REPO_DPATH = $REPO_DPATH"
if [[ ! -f "$REPO_DPATH/setup.py" ]] || [[ ! -f "$REPO_DPATH/src/python/pyxccd/__init__.py" ]]; then
    echo "NOT RUNNING FROM THE CORRECT DIRECTORY. CANNOT PREFORM POST-INSTALL HACKS. EXITING"
    set -e
    false
fi

if [ -d "./_skbuild" ]; then
    # Cleanup the scikit build directory
    echo "Removing old _skbuild directory"
    rm -rf _skbuild
else
    echo "Detected clean state (no _skbuild dir)"
fi

# Also clean up any shared libraries
rm src/python/pyxccd/_ccd_cython.*.so
# Clean up old egg links and such
rm -rf src/python/pyxccd.egg-info
rm -rf pyxccd.egg-info


# There editable install seems to have bugs. This is an attempt to fix them.
SITE_DPATH=$(python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib())")
pyxccd_EGG_LINK_FPATH="$SITE_DPATH"/pyxccd.egg-link
echo "pyxccd_EGG_LINK_FPATH = $pyxccd_EGG_LINK_FPATH"


pyxccd_EDITABLE_PTH_FPATH="$SITE_DPATH"/__editable__.pyxccd-0.1.0.pth
echo "pyxccd_EDITABLE_PTH_FPATH = $pyxccd_EDITABLE_PTH_FPATH"

# Need to get rid of the easy install entry if it exists
EASY_INSTALL_FPATH=$SITE_DPATH/easy-install.pth
if [ -f "$EASY_INSTALL_FPATH" ] && grep -q "$REPO_DPATH" "$EASY_INSTALL_FPATH"; then
    echo "Detected pyxccd in easy install path. Removing it before we reinstall."
    grep -v "$REPO_DPATH" "$EASY_INSTALL_FPATH" > tmpfile && mv tmpfile "$EASY_INSTALL_FPATH"
else
    echo "Easy install pth seems clean"
fi

### Handle installing some of the tricker requirements. ###

fix_opencv_conflicts(){
    __doc__="
    Check to see if the wrong opencv is installed, and perform steps to clean
    up the incorrect libraries and install the desired (headless) ones.
    "
    # Fix opencv issues
    if python -m pip freeze | grep -q "opencv-python=="; then
        HAS_OPENCV=1
    else
        HAS_OPENCV=0
    fi

    if python -m pip freeze | grep -q "opencv-python-headless=="; then
        HAS_OPENCV_HEADLESS=1
    else
        HAS_OPENCV_HEADLESS=0
    fi

    if [[ "$HAS_OPENCV_HEADLESS" == "1" && "$HAS_OPENCV" == "1" ]]; then
        echo "Found conflicting opencv packages. Reinstalling headless."
        python -m pip uninstall opencv-python opencv-python-headless -y
        python -m pip install opencv-python-headless
    elif [[ "$HAS_OPENCV_HEADLESS" == "0" ]]; then
        if [[ "$HAS_OPENCV" == "1" ]]; then
            echo "Found opencv-python, but headless is required. Replacing."
            python -m pip uninstall opencv-python -y
        fi
        echo "Installing opencv-python-headless."
        python -m pip install opencv-python-headless
    else
        echo "opencv-python-headless is correctly installed."
    fi
}

fix_opencv_conflicts
python -m pip install -r requirements/build.txt
python -m pip install -r requirements/runtime.txt
python -m pip install -r requirements/optional.txt
python -m pip install -r requirements/tests.txt


# Hack for setuptools while scikit-build sorts things out
# https://github.com/scikit-build/scikit-build/issues/740
export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"


###  COMPILE STEP ###
echo "Compiling and installing pyxccd in development mode..."
#
# Compile and install pyxccd in development mode.
# 
# It is important to have no-build-isolation to work
# around an issue with finding numpy headers.
# 
###
python setup.py develop


# This script will analyze the editable install and detect if anything is wrong
# It require some dependencies so it is commented by default
DEBUG_EDITABLE_INSTALL=0
if [[ "$DEBUG_EDITABLE_INSTALL" == "1" ]]; then
    python dev/setup.py mwe analyize --mod_name=pyxccd --repo_dpath="."
fi


### BEGINNING OF PROPOSED FIX ###
# Manually find and copy the compiled extension module as a fallback.
# The scikit-build editable install seems to be unreliable.
echo "Searching for the compiled Cython module in _skbuild..."
COMPILED_SO_FILE=$(find _skbuild -name "_ccd_cython*.so" -print -quit)

if [ -f "$COMPILED_SO_FILE" ]; then
    echo "Found compiled module at: $COMPILED_SO_FILE"
    DEST_PATH="src/python/pyxccd/"
    echo "Copying it to $DEST_PATH"
    cp "$COMPILED_SO_FILE" "$DEST_PATH"
else
    echo "WARNING: Could not find the compiled _ccd_cython module in _skbuild."
    echo "The installation will likely fail."
fi
### END OF PROPOSED FIX ###


###  HACKY FIXUP STEP ###
#
# Fix the contents of the egg-link in site-packages. See
# [EggFormatDetails]_ for egg format details.
# 
# References:
#     .. [EggFormatDetails] https://svn.python.org/projects/sandbox/trunk/setuptools/doc/formats.txt
###
echo "Applying post-install fixups for editable mode..."

if [ -f "$pyxccd_EGG_LINK_FPATH" ]; then
    echo "Fixing egg-link file: $pyxccd_EGG_LINK_FPATH"
    echo "$REPO_DPATH/src/python" > "$pyxccd_EGG_LINK_FPATH"
    echo "../../" >> "$pyxccd_EGG_LINK_FPATH"
else
    echo "Warning: egg-link file not found. Skipping fixup."
fi


###
# HACK #2 - Move the .egg-info folder into the correct location
###
# Notes about egg-link files:
if [ -d "pyxccd.egg-info" ]; then
    echo "Moving .egg-info folder to the correct location"
    mv pyxccd.egg-info ./src/python/
else
    echo "Warning: pyxccd.egg-info directory not found. Skipping move."
fi

if [ -f "$EASY_INSTALL_FPATH" ]; then
    echo "Fixing up easy-install.pth..."
    # Remove old entries first
    grep -v "$REPO_DPATH" "$EASY_INSTALL_FPATH" > tmpfile && mv tmpfile "$EASY_INSTALL_FPATH"
    # Add the correct new entry
    echo "$REPO_DPATH/src/python" >> "$EASY_INSTALL_FPATH"
else
    # If the file doesn't exist, this is not necessarily an error, but we can't fix it.
    echo "Warning: easy-install.pth not found. Skipping fixup."
fi

## Quick tests that the install worked
echo "Quick pyxccd tests to verify the install:"
echo "pyxccd Version: $(python -c 'import pyxccd; print(pyxccd.__version__)')"
echo "Python Package Location: $(python -c 'import pyxccd; print(pyxccd.__file__)')"
echo "Compiled Cython Module: $(python -c 'import pyxccd; print(pyxccd._ccd_cython.__file__)')"

echo "Developer setup complete."