import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field

SCCD_CONSE_OUTPUT = 8  # the default outputted observation number once S-CCD detects breakpoint or pinpoint, note it is not the conse for identifying breakpoints/pinpoints 
NRT_BAND = 6    # the default S-CCD band number
SCCD_NUM_C = 6  # the S-CCD harmonic model coefficient number
TOTAL_BAND_FLEX = 10    # the maximum band input for flexible mode of COLD
TOTAL_BAND_FLEX_NRT = 8 # the maximum band input for flexible mode of S-CCD

reccg_dt = np.dtype(
    [
        ("t_start", np.int32),  # time when series model gets started
        ("t_end", np.int32),  # time when series model gets ended
        ("t_break", np.int32),  # time when the first break (change) is observed
        ("pos", np.int32),  # the location of each time series model
        ("num_obs", np.int32),  # the number of "good" observations used for model estimation
        ("category", np.short), # the quality of the model estimation as a two-digit number (what model is used, what process is used)
            # first digit:
            #     0: normal model (no change)
            #     1: change at the beginning of time series model
            #     2: change at the end of time series model
            #     3: disturbance change in the middle
            #     4: fmask fail scenario
            #     5: permanent snow scenario
            #     6: outside user mask
            # second digit:
            #     1: model has only constant term
            #     4: model has 4 coefs
            #     6: model has 6 coefs
            #     8: model has 8 coefs*/
            # for example, 8 represents "normal model + 8 coefficients"
        ("change_prob", np.short), # the probability of a pixel that have undergone change (between 0 and 100)
        ("coefs", np.float32, (7, 8)), # coefficients for each time series model for seven spectral band, seven bands follow the order of "blue, green, red, nir, swir1, swir2, thermal"
                                       # seven row has 8 coefficients representing a 'annual-semiannual-trimode' harmonic model
        ("rmse", np.float32, 7),  # RMSE for each time series model for each seven band
        ("magnitude", np.float32, 7), # the magnitude of difference between model prediction and observation for each spectral band
    ]
)  


SccdOutput = namedtuple("SccdOutput", "position rec_cg min_rmse nrt_mode nrt_model nrt_queue")

sccd_dt = np.dtype(
    [
        ("t_start", np.int32), # ordenal date for the start of the time-series segment
        ("t_break", np.int32), # ordenal date for the break of the time-series segment
        ("num_obs", np.int32),  # the number of "good" observations used for model estimation
        ("coefs", np.float32, (NRT_BAND, SCCD_NUM_C)), # coefficients for each time series model for six spectral band 
        ("rmse", np.float32, NRT_BAND),    # RMSE for each time series model for each seven band
        ("magnitude", np.float32, NRT_BAND), # the magnitude of difference between model prediction and observation for each spectral band
    ],
    align=True,
)

nrtqueue_dt = np.dtype([("clry", np.short, NRT_BAND), ("clrx_since1982", np.short)], align=True)

nrtmodel_dt = np.dtype(
    [
        ("t_start_since1982", np.short),    # the date number since 1982-1-1 for the start of the time-series segment, equal to ordinal date + 723546 
        ("num_obs", np.short), # the number of "good" observations used for model estimation
        ("obs", np.short, (NRT_BAND, SCCD_CONSE_OUTPUT)),  # eight multispectral observations at tail (6 * 8)
        ("obs_date_since1982", np.short, SCCD_CONSE_OUTPUT),   # eight observation dates (counted since 1982-1-1) at tail (6 * 8)
        ("covariance", np.float32, (NRT_BAND, 36)),  # covariance matrix for six bands (6 * 36)
        ("nrt_coefs", np.float32, (NRT_BAND, SCCD_NUM_C)), # the current nrt_coefs (6 * 6)
        ("H", np.float32, NRT_BAND),    # the cobservation uncertainties (6 * 1)
        ("rmse_sum", np.uint32, NRT_BAND), # the sum of RMSE (6 * 1)
        ("norm_cm", np.short), # the normalized change magnitude
        ("cm_angle", np.short), # the included change angle
        ("conse_last", np.ubyte),
    ],
    align=True,
)


pinpoint_dt = np.dtype(
    [
        ("t_break", np.int32),
        ("coefs", np.float32, (NRT_BAND, SCCD_NUM_C)),
        ("obs", np.short, (NRT_BAND, SCCD_CONSE_OUTPUT)),
        ("obs_date_since1982", np.short, SCCD_CONSE_OUTPUT),
        ("norm_cm", np.short, SCCD_CONSE_OUTPUT),
        ("cm_angle", np.short, SCCD_CONSE_OUTPUT),
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
        ("category", np.short),  # the quality of the model estimation (what model is used, what process is used)
        ("change_prob", np.short), # the probability of a pixel that have undergone change (between 0 and 100)
        ("coefs", np.float32, (TOTAL_BAND_FLEX, 8)), # coefficients for each time series model for each spectral band
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
        ("obs", np.short, (TOTAL_BAND_FLEX_NRT, SCCD_CONSE_OUTPUT)),
        ("obs_date_since1982", np.short, SCCD_CONSE_OUTPUT),
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
        ("obs", np.short, (TOTAL_BAND_FLEX_NRT, SCCD_CONSE_OUTPUT)),
        ("obs_date_since1982", np.short, SCCD_CONSE_OUTPUT),
        ("norm_cm", np.short, SCCD_CONSE_OUTPUT),
        ("cm_angle", np.short, SCCD_CONSE_OUTPUT),
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
