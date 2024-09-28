import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field

DEFAULT_CONSE = 8
NRT_BAND = 6
SCCD_NUM_C = 6
TOTAL_BAND_FLEX = 10
TOTAL_BAND_FLEX_NRT = 8

reccg_dt = np.dtype(
    [
        ("t_start", np.int32),  # time when series model gets started
        ("t_end", np.int32),  # time when series model gets ended
        ("t_break", np.int32),  # time when the first break (change) is observed
        ("pos", np.int32),  # the location of each time series model
        ("num_obs", np.int32),  # the number of "good" observations used for model estimation
        # the quality of the model estimation (what model is used, what process is used)
        ("category", np.short),
        # the probability of a pixel that have undergone change (between 0 and 100)
        ("change_prob", np.short),
        # coefficients for each time series model for each spectral band
        ("coefs", np.float32, (7, 8)),
        ("rmse", np.float32, 7),  # RMSE for each time series model for each spectral band
        ("magnitude", np.float32, 7),
    ]
)  # the magnitude of change difference between model prediction
# and observation for each spectral band)


SccdOutput = namedtuple("SccdOutput", "position rec_cg min_rmse nrt_mode nrt_model nrt_queue")

sccd_dt = np.dtype(
    [
        ("t_start", np.int32),
        ("t_break", np.int32),
        ("num_obs", np.int32),
        ("coefs", np.float32, (NRT_BAND, SCCD_NUM_C)),
        ("rmse", np.float32, NRT_BAND),
        ("magnitude", np.float32, NRT_BAND),
    ],
    align=True,
)

nrtqueue_dt = np.dtype([("clry", np.short, NRT_BAND), ("clrx_since1982", np.short)], align=True)

nrtmodel_dt = np.dtype(
    [
        ("t_start_since1982", np.short),
        ("num_obs", np.short),
        ("obs", np.short, (NRT_BAND, DEFAULT_CONSE)),
        ("obs_date_since1982", np.short, DEFAULT_CONSE),
        ("covariance", np.float32, (NRT_BAND, 36)),
        ("nrt_coefs", np.float32, (NRT_BAND, SCCD_NUM_C)),
        ("H", np.float32, NRT_BAND),
        ("rmse_sum", np.uint32, NRT_BAND),
        ("norm_cm", np.short),
        ("cm_angle", np.short),
        ("conse_last", np.ubyte),
    ],
    align=True,
)


pinpoint_dt = np.dtype(
    [
        ("t_break", np.int32),
        ("coefs", np.float32, (NRT_BAND, SCCD_NUM_C)),
        ("obs", np.short, (NRT_BAND, DEFAULT_CONSE)),
        ("obs_date_since1982", np.short, DEFAULT_CONSE),
        ("norm_cm", np.short, DEFAULT_CONSE),
        ("cm_angle", np.short, DEFAULT_CONSE),
    ],
    align=True,
)

# the below is for sccd flex mode
reccg_dt_flex = np.dtype(
    [
        ("t_start", np.int32),  # time when series model gets started
        ("t_end", np.int32),  # time when series model gets ended
        ("t_break", np.int32),  # time when the first break (change) is observed
        ("pos", np.int32),  # the location of each time series model
        ("num_obs", np.int32),  # the number of "good" observations used for model estimation
        # the quality of the model estimation (what model is used, what process is used)
        ("category", np.short),
        # the probability of a pixel that have undergone change (between 0 and 100)
        ("change_prob", np.short),
        # coefficients for each time series model for each spectral band
        ("coefs", np.float32, (TOTAL_BAND_FLEX, 8)),
        ("rmse", np.float32, TOTAL_BAND_FLEX),  # RMSE for each time series model for each spectral band
        ("magnitude", np.float32, TOTAL_BAND_FLEX),
    ]
)  

sccd_dt_flex = np.dtype(
    [
        ("t_start", np.int32),
        ("t_break", np.int32),
        ("num_obs", np.int32),
        ("coefs", np.float32, (TOTAL_BAND_FLEX_NRT, SCCD_NUM_C)),
        ("rmse", np.float32, TOTAL_BAND_FLEX_NRT),
        ("magnitude", np.float32, TOTAL_BAND_FLEX_NRT),
    ],
    align=True,
)

nrtqueue_dt_flex = np.dtype([("clry", np.short, TOTAL_BAND_FLEX_NRT), ("clrx_since1982", np.short)], align=True)

nrtmodel_dt_flex = np.dtype(
    [
        ("t_start_since1982", np.short),
        ("num_obs", np.short),
        ("obs", np.short, (TOTAL_BAND_FLEX_NRT, DEFAULT_CONSE)),
        ("obs_date_since1982", np.short, DEFAULT_CONSE),
        ("covariance", np.float32, (TOTAL_BAND_FLEX_NRT, 36)),
        ("nrt_coefs", np.float32, (TOTAL_BAND_FLEX_NRT, SCCD_NUM_C)),
        ("H", np.float32, TOTAL_BAND_FLEX_NRT),
        ("rmse_sum", np.uint32, TOTAL_BAND_FLEX_NRT),
        ("norm_cm", np.short),
        ("cm_angle", np.short),
        ("conse_last", np.ubyte),
    ],
    align=True,
)


pinpoint_dt_flex = np.dtype(
    [
        ("t_break", np.int32),
        ("coefs", np.float32, (TOTAL_BAND_FLEX_NRT, SCCD_NUM_C)),
        ("obs", np.short, (TOTAL_BAND_FLEX_NRT, DEFAULT_CONSE)),
        ("obs_date_since1982", np.short, DEFAULT_CONSE),
        ("norm_cm", np.short, DEFAULT_CONSE),
        ("cm_angle", np.short, DEFAULT_CONSE),
    ],
    align=True,
)



@dataclass
class DatasetInfo:
    """data class for storing dataset basic info"""

    n_rows: int
    n_cols: int
    n_block_x: int  # the block number at x axis direction
    n_block_y: int  # the block number at y axis direction
    nblocks: int = field(init=False)
    block_width: int = field(init=False)
    block_height: int = field(init=False)

    def __post_init__(self) -> None:
        self.nblocks = self.n_block_x * self.n_block_y
        self.block_width = int(self.n_cols / self.n_block_x)
        self.block_height = int(self.n_rows / self.n_block_y)
