from ._version import __version__

# Ensure monkey patches run first
from . import monkey  # NOQA

# from . import _colds_cython  # NOQA

# from ._colds_cython import (  # NOQA
#     test_func, _cold_detect,
#     obcold_reconstruct, sccd_detect, sccd_update)

from . import ccd

# from . import _colds_cythons
from .ccd import (
    cold_detect,
    obcold_reconstruct,
    sccd_detect,
    cold_detect_flex,
    sccd_update,
    sccd_identify,
    calculate_sccd_cm,
    sccd_detect_flex,
    sccd_update_flex
)
