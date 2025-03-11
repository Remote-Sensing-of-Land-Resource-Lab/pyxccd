from collections import namedtuple
from dataclasses import dataclass, field, make_dataclass
from typing import Any
import numpy

SCCD_CONSE_OUTPUT = 8  # the default outputted observation number once S-CCD detects breakpoint or pinpoint, note it is not the conse for identifying breakpoints/pinpoints
NRT_BAND = 6  # the default S-CCD band number
SCCD_NUM_C = 6  # the S-CCD harmonic model coefficient number
TOTAL_BAND_FLEX = 10  # the maximum band input for flexible mode of COLD
TOTAL_BAND_FLEX_NRT = 8  # the maximum band input for flexible mode of S-CCD
SLOPE_SCALE = 10000


def _dtype_to_dataclass(dtype: numpy.dtype, class_name: str) -> type:
    """
    Converts a NumPy dtype to a dataclass.

    Args:
        dtype: The NumPy dtype to convert.
        class_name: The name of the generated dataclass.

    Returns:
        A new dataclass type representing the structure of the dtype.
    """

    fields = []
    for name, (subtype, offset) in dtype.fields.items():
        fields.append(
            (name, Any)
        )  # Use Any for simplicity, or map subtypes to Python types

    return make_dataclass(class_name, fields)


reccg_dt = numpy.dtype(
    [
        ("t_start", numpy.int32),
        ("t_end", numpy.int32),
        ("t_break", numpy.int32),
        ("pos", numpy.int32),
        ("num_obs", numpy.int32),
        ("category", numpy.short),
        ("change_prob", numpy.short),
        ("coefs", numpy.float32, (7, 8)),
        ("rmse", numpy.float32, 7),
        ("magnitude", numpy.float32, 7),
    ]
)


sccd_dt = numpy.dtype(
    [
        ("t_start", numpy.int32),
        ("t_break", numpy.int32),
        ("num_obs", numpy.int32),
        ("coefs", numpy.float32, (NRT_BAND, SCCD_NUM_C)),
        ("rmse", numpy.float32, NRT_BAND),
        ("magnitude", numpy.float32, NRT_BAND),
    ],
    align=True,
)

# sccd_dt.__doc__ = """
# historical temporal segment info obtained from S-CCD algorithm as a structured array, a composition of simple datatypes:

#     t_start: int
#         ordinal date when series model gets started
#     t_break: int
#         ordinal date when the structural break (change) is detected
#     pos: int
#         location of each time series model (i * n_row + j), e.g., the pos of (1000, 1) is 5000*1000+1
#     coefs: numpy.ndarray
#         2-d array of shape (nbands, ncoefs) to keep multispectral harmonic coefficients from the last lasso regression.
#         Spectral bands include blue, green, red, nir, swir1, swir2 (from  row 1 to 6); eight harmonic
#         coefficients include intercept, slope, cos_annual, sin_annual, cos_semi, sin_semi,
#         cos_trimodel, sin_trimodel (from col 1 to 6). Note the slope has been multiplied by
#         10000, S-CCD uses 6-coefs model and 6 spectral bands
#     rmse: numpy.ndarray
#         1-d array of shape (nbands,), multispectral RMSE of predicted and actiual observations
#     magntiude: numpy.ndarray
#         1-d array of shape (nbands,), multispectral median difference between model prediction and
#         observations of a window of conse observations following detected breakpoint
# """

nrtqueue_dt = numpy.dtype(
    [("clry", numpy.short, NRT_BAND), ("clrx_since1982", numpy.short)], align=True
)

nrtmodel_dt = numpy.dtype(
    [
        ("t_start_since1982", numpy.short),
        ("num_obs", numpy.short),
        ("obs", numpy.short, (NRT_BAND, SCCD_CONSE_OUTPUT)),
        ("obs_date_since1982", numpy.short, SCCD_CONSE_OUTPUT),
        ("covariance", numpy.float32, (NRT_BAND, 36)),
        ("nrt_coefs", numpy.float32, (NRT_BAND, SCCD_NUM_C)),
        ("H", numpy.float32, NRT_BAND),
        ("rmse_sum", numpy.uint32, NRT_BAND),
        ("norm_cm", numpy.short),
        ("cm_angle", numpy.short),
        ("conse_last", numpy.ubyte),
    ],
    align=True,
)


pinpoint_dt = numpy.dtype(
    [
        ("t_break", numpy.int32),
        ("coefs", numpy.float32, (NRT_BAND, SCCD_NUM_C)),
        ("obs", numpy.short, (NRT_BAND, SCCD_CONSE_OUTPUT)),
        ("obs_date_since1982", numpy.short, SCCD_CONSE_OUTPUT),
        ("norm_cm", numpy.short, SCCD_CONSE_OUTPUT),
        ("cm_angle", numpy.short, SCCD_CONSE_OUTPUT),
    ],
    align=True,
)

# the below is for sccd flex mode
reccg_dt_flex = numpy.dtype(
    [
        ("t_start", numpy.int32),
        ("t_end", numpy.int32),
        ("t_break", numpy.int32),
        ("pos", numpy.int32),
        ("num_obs", numpy.int32),
        ("category", numpy.short),
        ("change_prob", numpy.short),
        ("coefs", numpy.float32, (TOTAL_BAND_FLEX, 8)),
        ("rmse", numpy.float32, TOTAL_BAND_FLEX),
        ("magnitude", numpy.float32, TOTAL_BAND_FLEX),
    ]
)

sccd_dt_flex = numpy.dtype(
    [
        ("t_start", numpy.int32),
        ("t_break", numpy.int32),
        ("num_obs", numpy.int32),
        ("coefs", numpy.float32, (TOTAL_BAND_FLEX_NRT, SCCD_NUM_C)),
        ("rmse", numpy.float32, TOTAL_BAND_FLEX_NRT),
        ("magnitude", numpy.float32, TOTAL_BAND_FLEX_NRT),
    ],
    align=True,
)

nrtqueue_dt_flex = numpy.dtype(
    [("clry", numpy.short, TOTAL_BAND_FLEX_NRT), ("clrx_since1982", numpy.short)],
    align=True,
)

nrtmodel_dt_flex = numpy.dtype(
    [
        ("t_start_since1982", numpy.short),
        ("num_obs", numpy.short),
        ("obs", numpy.short, (TOTAL_BAND_FLEX_NRT, SCCD_CONSE_OUTPUT)),
        ("obs_date_since1982", numpy.short, SCCD_CONSE_OUTPUT),
        ("covariance", numpy.float32, (TOTAL_BAND_FLEX_NRT, 36)),
        ("nrt_coefs", numpy.float32, (TOTAL_BAND_FLEX_NRT, SCCD_NUM_C)),
        ("H", numpy.float32, TOTAL_BAND_FLEX_NRT),
        ("rmse_sum", numpy.uint32, TOTAL_BAND_FLEX_NRT),
        ("norm_cm", numpy.short),
        ("cm_angle", numpy.short),
        ("conse_last", numpy.ubyte),
    ],
    align=True,
)


pinpoint_dt_flex = numpy.dtype(
    [
        ("t_break", numpy.int32),
        ("coefs", numpy.float32, (TOTAL_BAND_FLEX_NRT, SCCD_NUM_C)),
        ("obs", numpy.short, (TOTAL_BAND_FLEX_NRT, SCCD_CONSE_OUTPUT)),
        ("obs_date_since1982", numpy.short, SCCD_CONSE_OUTPUT),
        ("norm_cm", numpy.short, SCCD_CONSE_OUTPUT),
        ("cm_angle", numpy.short, SCCD_CONSE_OUTPUT),
    ],
    align=True,
)


@dataclass
class DatasetInfo:
    """Store information for the dataset to be processed"""

    n_rows: int
    """The number of rows"""
    n_cols: int
    """The number of columns"""
    n_block_x: int
    """The block number in x direction"""
    n_block_y: int
    """The block number in y direction"""
    nblocks: int = field(init=False)
    """The total block number, n_block_x * n_block_y"""
    block_width: int = field(init=False)
    """The width of a block, n_cols / n_block_x"""
    block_height: int = field(init=False)
    """The height of a block, n_rows / n_block_y"""

    def __post_init__(self) -> None:
        """generate nblocks, block_width, and block_height  once the dataclass is initialized"""
        self.nblocks = self.n_block_x * self.n_block_y
        self.block_width = int(self.n_cols / self.n_block_x)
        self.block_height = int(self.n_rows / self.n_block_y)


SccdOutput = namedtuple(
    "SccdOutput", "position rec_cg min_rmse nrt_mode nrt_model nrt_queue"
)
SccdOutput.__doc__ = ": A namedtuple of standard S-CCD ouputs"
SccdOutput.position.__doc__ = ": int. Location of each time series model. \n The position is computed as (i * (n_row-1) + j), e.g., the pos of (1000, 1) is 5000*(1000-1)+1"
SccdOutput.rec_cg.__doc__ = ": numpy.ndarray. 1-d structured array with :py:class:`rec_cg`. \n Historical temporal segment info were obtained from S-CCD algorithm as a structured array"
SccdOutput.min_rmse.__doc__ = """: numpy.ndarray. 1-d array of shape (nbands,). \n The minimum RMSE  was obtained by temporal semivariogram. This array is unchanged with sccd_update since sccd_detect is first set."""
SccdOutput.nrt_mode.__doc__ = """:int. \n
                                    (first digit) \n
                                    0 - has predictability \n
                                    1 - not has predictability \n
                                    (second digit) \n
                                    The current sccd operating mode for the pixel: \n
                                    0 - void mode, not intialized yet \n
                                    1 - monitoring mode \n
                                    2 - queue mode. Once the break is detected, the mode is transition from monitoring to queue mode \n
                                    3 - monitoring mode for snow \n
                                    4 - queue mode for snow \n
                                    5 - transition mode from monitoring to queue mode (keep nrt_model and nrt_queue both), keeping 15 days since the break is first detected"""
SccdOutput.nrt_model.__doc__ = ": numpy.ndarray. 1-d structured array with :py:class:`nrt_model` dtype \n Current sccd model for the monitoring modes (1/3/5/11) that could be used for NRT monitoring"
SccdOutput.nrt_queue.__doc__ = """: numpy.ndarray. 1-d array of structured array with :py:class:`nrt_queue` dtype. \n 
                                    Observation collection to build initialization model for queue modes (2/4/5). 
                                    Store the observations in the queue, and cannot perform NRT monitoring as no model yet."""


# the below dataclass are created only for docstring purpose. Python so far doesn't support docstring for dtype yet.
@dataclass
class cold_rec_cg:
    """records of segments obtained from COLD algorithm as a structured array, a composition of simple datatypes"""

    t_start: numpy.int32
    """Ordinal date when series model gets started"""

    t_end: numpy.int32
    """Ordinal date when series model gets ended"""

    t_break: numpy.int32
    """Ordinal date when the structural break (change) is detected"""

    pos: numpy.int32
    """Location of each time series model (i * n_row + j), e.g., the pos of (1000, 1) is 5000*1000+1"""

    num_obs: numpy.int32
    """Number of clear observations used for model estimation"""

    category: numpy.short
    """Quality of the model estimation (what model is used, what process is used)
    
        (first digit)
        
        0 - normal model (no change)
        
        1 - change at the beginning of time series model
        
        2 - change at the end of time series model
        
        3 - disturbance change in the middle
        
        4 -  fmask fail scenario
        
        5 - permanent snow scenario
        
        6 - outside user mask
        
        (second digit)
        
        1 - model has only constant term
        
        4 - model has 3 coefs + 1 const
        
        6 - model has 5 coefs + 1 const
        
        8 - model has 7 coefs + 1 const
    """

    change_prob: numpy.short
    """Probability of a pixel that have undergone change (between 0 and 100)"""

    coefs: numpy.ndarray
    """2-d array of shape (nbands, ncoefs). It keeps multispectral harmonic coefficients from the last lasso regression. If not flexible mode, the spectral bands are fix, including blue, green, red, nir, swir1, swir2, thermal (from row 1 to 7). If flexible mode, the rows follow the order of inputted bands.The columns correspond to the eight harmonic coefficients including intercept, slope, cos_annual, sin_annual, cos_semi, sin_semi, cos_trimodel, sin_trimodel (from col 1 to 8). Note that the slope has been multiplied by 10000."""

    rmse: numpy.ndarray
    """1-d array of shape (nbands,), multispectral RMSE of predicted and actiual observations"""

    magntiude: numpy.ndarray
    """1-d array of shape (nbands,), multispectral median difference between model prediction and observations of a window of conse observations following detected breakpoint"""


@dataclass
class rec_cg:
    """historical temporal segment info obtained from S-CCD algorithm as a structured array, a composition of simple datatypes"""

    t_start: numpy.int32
    """Ordinal date when series model gets started."""
    t_break: numpy.int32
    """Ordinal date when the structural break (change) is detected."""
    num_obs: numpy.int32
    """the number of "good" observations used for model estimation."""
    coefs: numpy.ndarray
    """2-d array of shape (nbands, ncoefs) to keep multispectral harmonic coefficients from the last lasso regression. If not flexible mode, the spectral bands are fixed, including blue, green, red, nir, swir1, swir2 (from  row 1 to 6).If flexible mode, the rows follow the order of inputted bands.The columns are six harmonic coefficients including intercept, slope, cos_annual, sin_annual, cos_semi, sin_semi, (from col 1 to 6). Note the slope has been multiplied by 10000, S-CCD uses 6-coefs model and 6 spectral bands. """
    rmse: numpy.ndarray
    """1-d array of shape (nbands,), multispectral RMSE of predicted and actiual observations."""
    magnitude: numpy.ndarray
    """1-d array of shape (nbands,), multispectral median difference between model prediction and observations of a window of conse observations following detected breakpoint."""


@dataclass
class nrtqueue:
    """1-d array of structured type for queue modes (2/4/5). It was used to store the observations and the dates
    in the queue until the CCDC initialization condition is met."""

    clry: numpy.ndarray
    """1-d array of shape (nbands, 1), multispectral clear observation"""
    clrx_since1982: numpy.short
    """the date number since 1982/7/16, equal to ordinal date - 723742"""


@dataclass
class nrtmodel:
    """The sccd model for the monitoring modes (1/3/5/11), which allows NRT monitoring."""

    t_start_since1982: numpy.short
    """Date number since 1982/7/16, equal to ordinal date - 723742. """
    num_obs: numpy.short
    """Accumulated observation number for the current segment. """
    obs: numpy.ndarray
    """2-d array of shape (nbands, nobs). The 6 spectral bands follow the order (blue, green, red, nir, swir1, swir2) for last 8 observations"""
    obs_date_since1982: numpy.ndarray
    """1-d array of shape (nobs,). The date number since 1982/7/16 for the last 8 observations. """
    covariance: numpy.ndarray
    """2-d array of shape (nbands, ncofs_cov) | (nbands, n_cov_coefs). The covariance matrix for six bands (blue, green, red, nir, swir1, swir2). Each band has a 6*6 matrix as the covariance matrix
    was flatten into 1d. """
    nrt_coefs: numpy.ndarray
    """2-d array of shape (nbands, ncoefs) | (nbands, ncoefs). Each row is for each spectral for six spectral bands (blue, green, red, nir, swir1, swir2). Each row has 6 coefficients for each band. """
    H: numpy.ndarray
    """1-d array of shape (nbands,). Observation noise for six bands (blue, green, red, nir, swir1, swir2)."""
    rmse_sum: numpy.ndarray
    """1-d array of shape (nbands,). RMSE for six bands (blue, green, red, nir, swir1, swir2). """
    norm_cm: numpy.short
    """The current normalized change magnitude for the last conse_last spectral anomalies, multiplied by 100 and rounded. """
    cm_angle: numpy.short
    """The included angle for the last conse_last spectral anomalies, multiplied by 100 and rounded. """
    conse_last: numpy.byte
    """The current anomaly number at the tail of the time series. between 1 and 8. The anomalies were defined as the obs that are larger than gate_pcg. """


@dataclass
class pinpoint:
    """pinpoint segments as a structured array. S-CCD overdetected the spectral anomalies as "pinpoints"
    using conse =3 and threshold=gate_pcg, which is used to trained a retrospective machine learning
    model for NRT scenario."""

    t_break: numpy.int32
    """ordinal date when the pinpoint break is detected"""
    coefs: numpy.ndarray
    """2-d array of shape (nbands, ncoefs) to keep multispectral harmonic coefficients from the last lasso regression. Spectral bands include blue, green, red, nir, swir1, swir2, thermal (from row 1 to 7); eight harmonic coefficients include intercept, slope, cos_annual, sin_annual, cos_semi, sin_semi, cos_trimodel, sin_trimodel (from col 1 to 8). Note the slope has been multiplied by 10000."""
    obs: numpy.ndarray
    """2-d array of shape (nbands, nobs). The 6 spectral bands follow the order (blue, green, red, nir, swir1, swir2) for last 8 observations."""
    obs_date_since1982: numpy.ndarray
    """1-d array of shape (nobs,). The date number since 1982/7/16 for the last 8 observations. """
    norm_cm: numpy.short
    """Normalized change magnitude for the last conse_last spectral anomalies, multiplied by 100 and rounded. """
    cm_angle: numpy.short
    """included angale fot the last conse_last spectral anomalies, multiplied by 100 and rounded. """
