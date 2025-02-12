from ._colds_cython import (
    _sccd_update,
    _sccd_detect,
    _obcold_reconstruct,
    _cold_detect,
    _cold_detect_flex,
    _sccd_detect_flex,
)
import numpy as np
from .common import SccdOutput
from ._param_validation import (
    validate_parameter_constraints,
    Integral,
    Interval,
    Real,
    check_consistent_length,
    check_1d,
)
from .utils import calculate_sccd_cm
from .app import defaults
from scipy.stats import chi2

_parameter_constraints: dict = {
    "t_cg": [Interval(Real, 0.0, None, closed="neither")],
    "p_cg": [Interval(Real, 0.0, 1, closed="neither")],
    "pos": [Interval(Integral, 0, None, closed="neither")],
    "starting_date": [Interval(Integral, 0, None, closed="left")],
    "n_cm": [Interval(Integral, 0, None, closed="left")],
    "conse": [Interval(Integral, 0, 12, closed="neither")],
    "b_output_cm": ["boolean"],
    "gap_days": [Interval(Real, 0.0, None, closed="left")],
    "b_pinpoint": ["boolean"],
    "gate_pcg": [Interval(Real, 0.0, 1, closed="neither")],
    "predictability_pcg": [Interval(Real, 0.0, 1, closed="neither")],
    "dist_conse": [Interval(Integral, 0, 6, closed="right")],
    "t_cg_scale100": [Interval(Real, 0.0, None, closed="neither")],
    "t_cg_singleband": [Interval(Real, None, None, closed="neither")],
    "t_angle": [Interval(Integral, 0, 180, closed="neither")],
    "transform_mode": ["boolean"],
    "state_intervaldays": [Interval(Real, 0.0, None, closed="left")],
    "fitlam": [Interval(Real, 0.0, 100, closed="left")],
}

NUM_FC = 40  # define the maximum number of outputted curves
NUM_FC_SCCD = 40
NUM_NRT_QUEUE = 240
MAX_FLEX_BAND = 10
MAX_FLEX_BAND_SCCD = 8
DEFAULT_BANDS = 5


def _validate_params(func_name, **kwargs):
    """Validate types and values of constructor parameters
    The expected type and values must be defined in the `_parameter_constraints`
    class attribute, which is a dictionary `param_name: list of constraints`. See
    the docstring of `validate_parameter_constraints` for a description of the
    accepted constraints.
    """
    # params = dict()
    # for key in kwargs:
    #     dict[key] = kwargs[key]

    validate_parameter_constraints(
        _parameter_constraints,
        kwargs,
        caller_name=func_name,
    )


def _validate_data(
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates=None
):
    """
    validate and forcibly change the data format
    Parameters
    ----------
    dates: 1d array of shape(n_obs,), list of ordinal dates
    ts_b: 1d array of shape(n_obs,), time series of blue band.
    ts_g: 1d array of shape(n_obs,), time series of green band
    ts_r: 1d array of shape(n_obs,), time series of red band
    ts_n: 1d array of shape(n_obs,), time series of nir band
    ts_s1: 1d array of shape(n_obs,), time series of swir1 band
    ts_s2: 1d array of shape(n_obs,), time series of swir2 band
    ts_t: 1d array of shape(n_obs,), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud

    Returns
    ----------
    """
    check_consistent_length(dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas)
    check_1d(dates, "dates")
    check_1d(ts_b, "ts_b")
    check_1d(ts_g, "ts_g")
    check_1d(ts_r, "ts_r")
    check_1d(ts_n, "ts_n")
    check_1d(ts_s1, "ts_s1")
    check_1d(ts_s2, "ts_s2")
    check_1d(qas, "qas")
    if break_dates is not None:
        check_1d(break_dates, "break_dates")

    dates = dates.astype(dtype=np.int64, order="C")
    ts_b = ts_b.astype(dtype=np.int64, order="C")
    ts_g = ts_g.astype(dtype=np.int64, order="C")
    ts_r = ts_r.astype(dtype=np.int64, order="C")
    ts_n = ts_n.astype(dtype=np.int64, order="C")
    ts_s1 = ts_s1.astype(dtype=np.int64, order="C")
    ts_s2 = ts_s2.astype(dtype=np.int64, order="C")
    ts_t = ts_t.astype(dtype=np.int64, order="C")
    qas = qas.astype(dtype=np.int64, order="C")
    if break_dates is not None:
        break_dates = break_dates.astype(dtype=np.int64, order="C")
        return dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates
    return dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas


def _validate_data_flex(dates, ts_data, qas):
    """
    validate and forcibly change the data format
    Parameters
    ----------
    dates: 1d array of shape(n_obs,), list of ordinal dates
    ts_data: 2d array of shape(n_obs,), time series stack for inputs
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud

    Returns
    ----------
    """
    check_consistent_length(dates, ts_data, qas)
    check_1d(dates, "dates")
    check_1d(qas, "qas")

    dates = dates.astype(dtype=np.int64, order="C")
    ts_data = ts_data.astype(dtype=np.int64, order="C")
    qas = qas.astype(dtype=np.int64, order="C")
    return dates, ts_data, qas


# def _nrt_stablity_test(sccd_plot, threshold, parameters, conse=6):
#     """
#     calculate the valid conse pixel, and the stable observation number out of valid conse
#     Parameters
#     ----------
#     sccd_plot: sccd pack
#     threshold: the change magnitude threshold to define stable obs
#     parameters: pyxccd parameters
#     conse: the default conse
#
#     Returns
#     -------
#     number of test obs, number of stable obs
#     """
#
#     if sccd_plot.nrt_mode == 0 or sccd_plot.nrt_mode == 3 or sccd_plot.nrt_mode == 4: # snow mode
#         return 0, 0
#     else:
#         pred_ref = np.asarray([[predict_ref(sccd_plot.nrt_model[0]['nrt_coefs'][b],
#                                             sccd_plot.nrt_model[0]['obs_date_since1982'][
#                                                 i_conse] + parameters['COMMON'][
#                                                 'JULIAN_LANDSAT4_LAUNCH'])
#                                 for i_conse in range(parameters['COMMON']['default_conse'])]
#                                for b in range(parameters['SCCD']['NRT_BAND'])])
#         cm = (sccd_plot.nrt_model[0]['obs'][:, 0:parameters['COMMON']['default_conse']] - pred_ref)[1:6, :]  # exclude blue band
#         if sccd_plot.nrt_model['num_obs'] < 18:
#             df = 4
#         else:
#             df = 6
#         cm_normalized = np.sum((cm.T / np.max([sccd_plot.min_rmse[1:6],
#                                                np.sqrt(sccd_plot.nrt_model['rmse_sum'][0][1:6] /
#                                                        (sccd_plot.nrt_model['num_obs'] - df))], axis=0)).T ** 2, axis=0)
#         if sccd_plot.nrt_mode == 2 or sccd_plot.nrt_mode == 5:  # bi mode - legacy
#             valid_test_num = np.min([len(sccd_plot.nrt_queue) - conse, conse])
#             cm_normalized = cm_normalized[(conse - valid_test_num): conse]
#             n_stable = len(cm_normalized[cm_normalized < threshold])
#             return valid_test_num, n_stable
#         elif sccd_plot.nrt_mode == 1:  # monitor mode
#             n_stable = len(cm_normalized[cm_normalized < threshold])
#             return conse, n_stable
#
#
# def test_stablity(sccd_plot, parameters, threshold=15.086, min_test_obs=3, stable_ratio=0.5):
#     """
#     test if the harmonic model is stable
#     Parameters
#     ----------
#     sccd_plot: sccd pack
#     threshold: the change magnitude threshold to define stable obs
#     parameters: pyxccd parameters
#
#     Returns
#     -------
#     True or false
#     """
#     test_conse, n_stable = _nrt_stablity_test(sccd_plot, threshold, parameters, conse=6)
#     if n_stable < min_test_obs:
#         return False
#     else:
#         if n_stable * 1.0 / test_conse > stable_ratio:
#             return True
#         else:
#             return False


def cold_detect(
    dates,
    ts_b,
    ts_g,
    ts_r,
    ts_n,
    ts_s1,
    ts_s2,
    ts_t,
    qas,
    p_cg=0.99,
    conse=6,
    pos=1,
    b_output_cm=False,
    starting_date=0,
    n_cm=0,
    cm_output_interval=0,
    b_c2=True,
    gap_days=365.25,
    fitlam=20,
):
    """
    pixel-based COLD algorithm.
    Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., &
    Zhou, C. (2020). Continuous monitoring of land disturbance based on Landsat time series.
    Remote Sensing of Environment, 38, 111116.
    Parameters
    ----------
    dates: 1d array of shape(n_obs,), list of ordinal dates
    ts_b: 1d array of shape(n_obs,), time series of blue band.
    ts_g: 1d array of shape(n_obs,), time series of green band
    ts_r: 1d array of shape(n_obs,), time series of red band
    ts_n: 1d array of shape(n_obs,), time series of nir band
    ts_s1: 1d array of shape(n_obs,), time series of swir1 band
    ts_s2: 1d array of shape(n_obs,), time series of swir2 band
    ts_t: 1d array of shape(n_obs,), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: probability threshold of change magnitude, default is 0.99
    conse: consecutive observation number, default is 6
    pos: position id of the pixel, default is 1
    b_output_cm: bool, 'True' means outputting change magnitude and change magnitude dates, only for
                object-based COLD, default is False
    starting_date: the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile.
                    should have the same date, only for b_output_cm is True. default is 0.
    n_cm: the length of outputted change magnitude. Only b_output_cm == 'True'. default is 0.
    cm_output_interval: the temporal interval of outputting change magnitudes. Only b_output_cm == 'True'. default is 0.
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel
          test due to the current low quality. default is True
    gap_days: define the day number of the gap year for determining i_dense. The COLD will skip the i_dense days
            to set the starting point of the model. Setting a large value (e.g., 1500) if the gap year
            is in the middle of the time range. default is 365.25.
    fitlam: the lamba used for the final fitting. Won't change the detection accuracy, but will affect the outputted harmonic model

    Returns: a 1-d array of structured type "output_rec_cg"
    ----------
    change records: the COLD outputs that characterizes each temporal segment if b_output_cm==False
    Or
    [change records, cm_outputs, cm_outputs_date] if b_output_cm==True

    output_rec_cg: [{
        t_start: int, ordinal date when series model gets started
        t_end: int, ordinal date when series model gets ended
        t_break: int, ordinal date when the first break (change) is detected
        pos: int, the location of each time series model (i * n_row + j), e.g., the pos of (1000, 1) is 5000*1000+1
        num_obs: int,  the number of clear observations used for model estimation
        category: short, the quality of the model estimation (what model is used, what process is used)
            The current category in output structure:
                first digit:
                0: normal model (no change)
                1: change at the beginning of time series model
                2: change at the end of time series model
                3: disturbance change in the middle
                4: fmask fail scenario
                5: permanent snow scenario
                6: outside user mask
                second digit:
                1: model has only constant term
                4: model has 3 coefs + 1 const
                6: model has 5 coefs + 1 const
                8: model has 7 coefs + 1 const*
        change_prob: short, the probability of a pixel that have undergone change (between 0 and 100)
        coefs: 2-d array of shape (nbands, coefs_number), eight harmonic coefficients for each
                spectral band (intercept, slope, cos_annual, sin_annual, cos_semi, sin_semi, cos_trimodel,
                sin_trimodel). Note the slope has been multiplied by 10000.
        rmse: 1-d array of shape (nbands,)
        magntiude: 1-d array of shape (nbands, ), the median difference between model prediction and
                    observations of a window of conse observations following detected breakpoint
        }
    ]
    """

    _validate_params(
        func_name="cold_detect",
        p_cg=p_cg,
        pos=pos,
        conse=conse,
        b_output_cm=b_output_cm,
        starting_date=starting_date,
        n_cm=n_cm,
        cm_output_interval=cm_output_interval,
        b_c2=b_c2,
        gap_days=gap_days,
        fitlam=fitlam,
    )

    # make sure it is c contiguous array and 64 bit
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas = _validate_data(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas
    )
    t_cg = chi2.ppf(p_cg, DEFAULT_BANDS)

    return _cold_detect(
        dates,
        ts_b,
        ts_g,
        ts_r,
        ts_n,
        ts_s1,
        ts_s2,
        ts_t,
        qas,
        t_cg,
        conse,
        pos,
        b_output_cm,
        starting_date,
        n_cm,
        cm_output_interval,
        b_c2,
        gap_days,
        fitlam,
    )


def obcold_reconstruct(
    dates,
    ts_b,
    ts_g,
    ts_r,
    ts_n,
    ts_s1,
    ts_s2,
    ts_t,
    qas,
    break_dates,
    pos=1,
    conse=6,
    b_c2=True,
):
    """
    re-contructructing change records using break dates.
    Parameters
    ----------
    dates: 1d array of shape (n_obs,, ), list of ordinal dates
    ts_b: 1d array of shape (n_obs,, ), time series of blue band.
    ts_g: 1d array of shape (n_obs,, ), time series of green band
    ts_r: 1d array of shape (n_obs,, ), time series of red band
    ts_n: 1d array of shape (n_obs,, ), time series of nir band
    ts_s1: 1d array of shape (n_obs,, ), time series of swir1 band
    ts_s2: 1d array of shape (n_obs,, ), time series of swir2 band
    ts_t: 1d array of shape (n_obs,, ), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    break_dates: 1d array, the break dates obtained from other procedures such as obia
    conse: consecutive observation number (for calculating change magnitudes)
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test
          due to its current low quality

    Returns
    ----------
    a 1-d array of structured type
    output_rec_cg:
    {
        t_start: int, ordinal date when series model gets started
        t_end: int, ordinal date when series model gets ended
        t_break: int, ordinal date when the first break (change) is detected
        pos: int, the location of each time series model (i * (n_row-1) + j), e.g., the pos of (1000, 1) is 5000*(1000-1)+1
        num_obs: int,  the number of clear observations used for model estimation
        category: short, the quality of the model estimation (what model is used, what process is used)
            The current category in output structure:
                first digit:
                0: normal model (no change)
                1: change at the beginning of time series model
                2: change at the end of time series model
                3: disturbance change in the middle
                4: fmask fail scenario
                5: permanent snow scenario
                6: outside user mask
                second digit:
                1: model has only constant term
                4: model has 3 coefs + 1 const
                6: model has 5 coefs + 1 const
                8: model has 7 coefs + 1 const*
        change_prob: short, the probability of a pixel that have undergone change (between 0 and 100)
        coefs: 2-d array of shape (7, 8), each row represents the bands following the order (blue, green, red, nir, swir1, swir2, thermal)
            which has eight harmonic coefficients, i.e., intercept, slope, cos_annual, sin_annual, cos_semi, sin_semi, cos_trimodel, sin_trimodel.
            Note the slope has been multiplied by 10000.
        rmse: 1-d array of shape (7, 8),  RMSE for each time series model for seven spectral band
        magntiude: 1-d array of shape (7, ), the median difference between model prediction and
                    observations of a window of conse observations following detected breakpoint
    }
    """
    _validate_params(func_name="sccd_detect", pos=pos, conse=conse, b_c2=b_c2)
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates = (
        _validate_data(
            dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates
        )
    )

    return _obcold_reconstruct(
        dates,
        ts_b,
        ts_g,
        ts_r,
        ts_n,
        ts_s1,
        ts_s2,
        ts_t,
        qas,
        break_dates,
        pos,
        conse,
        b_c2,
    )


def sccd_detect(
    dates,
    ts_b,
    ts_g,
    ts_r,
    ts_n,
    ts_s1,
    ts_s2,
    qas,
    p_cg=0.99,
    conse=6,
    pos=1,
    b_c2=True,
    b_pinpoint=False,
    gate_pcg=0.90,
    state_intervaldays=0.0,
    b_fitting_coefs=False,
):
    """
    pixel-based offline SCCD algorithm.
    Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time approach for monitoring forest
    disturbance using Landsat time series: Stochastic continuous change detection.
    Remote Sensing of Environment, 252, 112167.

    Parameters
    ----------
    dates: 1d array of shape(n_obs,), list of ordinal dates
    ts_b: 1d array of shape(n_obs,), time series of blue band.
    ts_g: 1d array of shape(n_obs,), time series of green band
    ts_r: 1d array of shape(n_obs,), time series of red band
    ts_n: 1d array of shape(n_obs,), time series of nir band
    ts_s1: 1d array of shape(n_obs,), time series of swir1 band
    ts_s2: 1d array of shape(n_obs,), time series of swir2 band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: probability threshold of change magnitude. The default 0.99
    conse: consecutive observation number. by default 6.
    pos: position id of the pixel, i.e.,  (row -1) * ncols + col, row and col starts from 1. by default 1.
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 ignores thermal band for valid pixel
                test due to its current low quality. by default True.
    b_pinpoint: bool, output pinpoint break where pinpoint is an overdetection of break using conse =3
                        and threshold = gate_tcg, which overdetects anomalies to simulate the situation
                        of NRT scenario and for training a retrospective model. by default True.
    gate_pcg: float, the gate change probability threshold for defining spectral anomalies (NRT)/pinpoints. by default 0.90.
    state_intervaldays: float, the day interval for output states. by default 0.0.
    b_fitting_coefs: True indicates using curve fitting to get harmonic coefficients for the segment,
                    otherwise use the local coefficients from kalman filter. by default False.
    Returns
    ----------
    SccdOutput: namedtype, if b_pinpoint==False and b_output_state==False
    [SccdOutput, state_days, state_ensemble]:[namedtype, 1-d array, 2-d array], if b_pinpoint==False and b_output_state==True
    [SccdOutput, SccdReccgPinpoint]: [namedtype, 1-d array of structured type], if b_pinpoint==True
    SccdOutput: namedtype[pos, output_rec_cg, min_rmse, nrt_mode, nrt_model, nrt_queue]
        pos: int, the location of each time series model (i * (n_row-1) + j), e.g., the pos of (1000, 1) is 5000*(1000-1)+1
        output_rec_cg: a structured type,
        {
            t_start: int, ordinal date when series model gets started
            t_break: int, ordinal date when the first break (change) is detected
            num_obs: int,  the number of clear observations used for model estimation
            coefs: 2-d array of shape (6, 6), each row represents the bands following the order
            (blue, green, red, nir, swir1, swir2) which has six harmonic coefficients, i.e., intercept,
            slope, cos_annual, sin_annual, cos_semi, sin_semi.
            Note the slope has been multiplied by 10000, and S-CCD uses 6-coefs model instead of 8-coefs that COLD uses
            rmse: 1-d array of shape (6,), RMSE for each time series model for seven spectral band
            magntiude: 1-d array of shape (6, ) for six spectral bands, the median difference
                        between model prediction and observations of a window of conse observations
                        following detected breakpoint
        }
        min_rmse: a 1-d array of shape(6,), minimum RMSE which was obtained by temporal semivariogram.
            This array is unchanged since it is first set
        nrt_mode: the current sccd mode for the pixel
            0 - void mode, not start yet
            1 - monitoring mode.
            11 - monitoring mode, but not passing predictability test
            2 - queue mode. Once the break is detected, the mode is transition from monitoring to queue mode
            3 - monitoring mode for snow
            4 - queue mode for snow
            5 - transition mode from monitoring to queue mode (keep nrt_model and nrt_queue both), keeping 15 days since the break is first detected
        nrt_model: a structured type. Has valid values when nrt_mode = 1/3/5
        {
            t_start_since1982: short, the date number since 1982/1/1, equal to ordinal date - 723546
            num_obs: short, the accumulated observation number for the current segment
            obs: 2-d array of shape (6, 8). The 6 spectral bands (blue, green, red, nir, swir1, swir2) for last 8 observations
            obs_date_since1982: 1-d array of shape (8,). The date number since 1982/1/1 for the last 8 observations
            covariance: 2-d array of shape (6, 36). The covariance matrix for six bands (blue, green, red, nir, swir1, swir2). Each band has a 6*6 matrix (the matrix has been flatten into 1d)
            nrt_coefs: 2-d array of shape (6, 6). The current harmonic models for six spectral bands (blue, green, red, nir, swir1, swir2). Each row has 6 coefficients for each band.
            H:1-d array of shape (6,1). Observation noise for six bands (blue, green, red, nir, swir1, swir2)
            rmse_sum: 1-d array of shape (6,). RMSE for six bands (blue, green, red, nir, swir1, swir2)
            norm_cm: short, ,the current normalized change magnitude for the last conse_last spectral anomalies, multiplied by 100
            cm_angle: short, the included angale fot the last conse_last spectral anomalies, multiplied by 100
            conse_last: char, the current anomaly number at the tail of the time series. The anomalies were defined as the obs that are larger than gate_pcg
        }
        nrt_queue: a 1-d array of structured type "nrtqueue_dt". Store the observations in the queue.
                    Has valid values when nrt_mode = 2/4/5.
            nrtqueue_dt
            {
                clry: 1-d array of shape (6,), the spectral bands
                clrx_since1982: short, the date number since 1982/1/1, equal to ordinal date - 723546
            }


    state_days: int, 1-d array (n_states,). The ordinal dates for the output states.
    states_ensemble: 2-d array (n_states, 18). The column index are
                    ['blue_trend', 'green_trend', 'red_trend', 'nir_trend', 'swir1_trend', 'swir2_trend',
                     'blue_annual', 'green_annual', 'red_annual', 'nir_annual', 'swir1_annual', 'swir2_annual',
                     'blue_semiannual', 'green_semiannual', 'red_semiannual', 'nir_semiannual', 'swir1_semiannual', 'swir2_semiannual']

    SccdReccgPinpoint: a structured type. Overdetected the spectral anomalies as "pinpoints" using conse =3 and threshold=gate_pcg
                        SccdReccgPinpoint is used to trained a retrospective machine learning model for NRT scenario
    {
        t_break: int, ordinal date when the first break (change) is detected
        coefs: 2-d array of shape (6, 6), each row represents the bands following the order
        obs: 2-d array of shape (6, 8). The 6 spectral bands (blue, green, red, nir, swir1, swir2) for last 8 observations tracing from the break
        obs_date_since1982: 1-d array of shape (8,). The date number since 1982/1/1 for the last 8 observations tracing from the break.
        norm_cm: short, normalized change magnitude for the last conse_last spectral anomalies, multiplied by 100 and rounded
        cm_angle: short, included angale fot the last conse_last spectral anomalies, multiplied by 100 and rounded
    }
    """
    _validate_params(
        func_name="sccd_detect",
        p_cg=p_cg,
        pos=pos,
        conse=conse,
        b_c2=b_c2,
        b_pinpoint=b_pinpoint,
        gate_pcg=gate_pcg,
        state_intervaldays=state_intervaldays,
    )
    ts_t = np.zeros(ts_b.shape)
    # make sure it is c contiguous array and 64 bit
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas = _validate_data(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas
    )
    t_cg = chi2.ppf(p_cg, DEFAULT_BANDS)
    gate_tcg = chi2.ppf(gate_pcg, DEFAULT_BANDS)
    # sccd_wrapper = SccdDetectWrapper()
    # tmp = copy.deepcopy(sccd_wrapper.sccd_detect(dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, t_cg,
    #                                 pos, conse, b_c2, b_pinpoint, gate_tcg, b_monitor_init))
    # return tmp
    if state_intervaldays == 0:
        b_output_state = False
    else:
        b_output_state = True
    return _sccd_detect(
        dates,
        ts_b,
        ts_g,
        ts_r,
        ts_n,
        ts_s1,
        ts_s2,
        ts_t,
        qas,
        t_cg,
        conse,
        pos,
        b_c2,
        b_pinpoint,
        gate_tcg,
        9.236,
        b_output_state,
        state_intervaldays,
        b_fitting_coefs,
    )


def sccd_update(
    sccd_pack,
    dates,
    ts_b,
    ts_g,
    ts_r,
    ts_n,
    ts_s1,
    ts_s2,
    qas,
    p_cg=0.99,
    conse=6,
    pos=1,
    gate_pcg=0.90,
    predictability_pcg=0.90,
):
    """
    SCCD online update for new observations
    Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time approach for monitoring forest
    disturbance using Landsat time series: Stochastic continuous change detection.
    Remote Sensing of Environment, 252, 112167.

    Parameters
    ----------
    sccd_pack:
    dates: 1d array of shape(n_obs,), list of ordinal dates
    ts_b: 1d array of shape(n_obs,), time series of blue band.
    ts_g: 1d array of shape(n_obs,), time series of green band
    ts_r: 1d array of shape(n_obs,), time series of red band
    ts_n: 1d array of shape(n_obs,), time series of nir band
    ts_s1: 1d array of shape(n_obs,), time series of swir1 band
    ts_s2: 1d array of shape(n_obs,), time series of swir2 band
    ts_t: 1d array of shape(n_obs,), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: float, probability threshold of change magnitude, default is chi2.ppf(0.99,5), by default 0.99
    pos: int, position id of the pixel, i.e.,  (row -1) * ncols + col, row and col starts from 1. by default 1.
    conse: int, consecutive observation number. by default 6.
    gate_pcg: float, the gate probability threshold for defining spectral anomalies. by default 0.90.
    predictability_pcg: float, the probability threshold for predictability test. If not passed, the nrt_mode will return 11. by default 0.90.
    Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands
    Returns
    ----------
    SccdOutput: namedtype [pos, output_rec_cg, min_rmse, nrt_mode, nrt_model, nrt_queue]
        pos: int, the location of each time series model (i * (n_row-1) + j), e.g., the pos of (1000, 1) is 5000*(1000-1)+1
        output_rec_cg: a structured type,
        {
            t_start: int, ordinal date when series model gets started
            t_break: int, ordinal date when the first break (change) is detected
            num_obs: int,  the number of clear observations used for model estimation
            coefs: 2-d array of shape (6, 6), each row represents the bands following the order
            (blue, green, red, nir, swir1, swir2) which has six harmonic coefficients, i.e., intercept,
            slope, cos_annual, sin_annual, cos_semi, sin_semi.
            Note the slope has been multiplied by 10000, and S-CCD uses 6-coefs model instead of 8-coefs that COLD uses
            rmse: 1-d array of shape (6,), RMSE for each time series model for seven spectral band
            magntiude: 1-d array of shape (6, ) for six spectral bands, the median difference
                        between model prediction and observations of a window of conse observations
                        following detected breakpoint
        }
        min_rmse: a 1-d array of shape(6,), minimum RMSE which was obtained by temporal semivariogram.
            This array is unchanged since it is first set
        nrt_mode: the current sccd mode for the pixel
            0 - void mode, not start yet
            1 - monitoring mode.
            11 - monitoring mode, but not passing predictability test
            2 - queue mode. Once the break is detected, the mode is transition from monitoring to queue mode
            3 - monitoring mode for snow
            4 - queue mode for snow
            5 - transition mode from monitoring to queue mode (keep nrt_model and nrt_queue both), keeping 15 days since the break is first detected
        nrt_model: a structured type. Has valid values when nrt_mode = 1/3/5
        {
            t_start_since1982: short, the date number since 1982/1/1, equal to ordinal date - 723546
            num_obs: short, the accumulated observation number for the current segment
            obs: 2-d array of shape (6, 8). The 6 spectral bands (blue, green, red, nir, swir1, swir2) for last 8 observations
            obs_date_since1982: 1-d array of shape (8,). The date number since 1982/1/1 for the last 8 observations
            covariance: 2-d array of shape (6, 36). The covariance matrix for six bands (blue, green, red, nir, swir1, swir2). Each band has a 6*6 matrix (the matrix has been flatten into 1d)
            nrt_coefs: 2-d array of shape (6, 6). The current harmonic models for six spectral bands (blue, green, red, nir, swir1, swir2). Each row has 6 coefficients for each band.
            H:1-d array of shape (6,1). Observation noise for six bands (blue, green, red, nir, swir1, swir2)
            rmse_sum: 1-d array of shape (6,). RMSE for six bands (blue, green, red, nir, swir1, swir2)
            norm_cm: short, ,the current normalized change magnitude for the last conse_last spectral anomalies, multiplied by 100
            cm_angle: short, the included angale fot the last conse_last spectral anomalies, multiplied by 100
            conse_last: char, the current anomaly number at the tail of the time series. The anomalies were defined as the obs that are larger than gate_pcg
        }
    """
    # if not isinstance(sccd_pack, SccdOutput):
    #     raise ValueError("The type of sccd_pack has to be namedtuple 'SccdOutput'!")

    _validate_params(
        func_name="sccd_update",
        p_cg=p_cg,
        pos=pos,
        conse=conse,
        b_pinpoint=False,
        gate_pcg=gate_pcg,
        predictability_pcg=predictability_pcg,
    )
    ts_t = np.zeros(ts_b.shape)
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas = _validate_data(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas
    )
    t_cg = chi2.ppf(p_cg, DEFAULT_BANDS)
    gate_tcg = chi2.ppf(gate_pcg, DEFAULT_BANDS)
    predictability_tcg = chi2.ppf(predictability_pcg, DEFAULT_BANDS)
    return _sccd_update(
        sccd_pack,
        dates,
        ts_b,
        ts_g,
        ts_r,
        ts_n,
        ts_s1,
        ts_s2,
        ts_t,
        qas,
        t_cg,
        conse,
        pos,
        True,
        gate_tcg,
        predictability_tcg,
    )


def sccd_identify(
    sccd_pack,
    dist_conse=6,
    p_cg=0.99,
    t_cg_singleband=-200,
    t_angle=45,
    transform_mode=False,
):
    """
    identify disturbance date from sccd_pack
    Parameters
    ----------
    sccd_pack: namedtuple
    dist_conse: int, the observation number, by default 6
    p_cg: float, change magnitude probability threshold, by default 0.99
    t_cg_singleband: float, single-band change magnitude to identify greenning breaks, by default -200
      see Eq. 10 in Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., & Zhou, C. (2020).
      Continuous monitoring of land disturbance based on Landsat time series. Remote Sensing of Environment, 238, 111116.
    t_angle_scale100: float, threshold for included angle (scale by 100), by default 45
    transform_mode: bool, transform the mode to untested predictability once the change has been detected, by default False
    Returns
    -------
    int
        0 (no disturbance) or the ordinal date of disturbance occurrence
    """
    _validate_params(
        func_name="sccd_identify",
        dist_conse=dist_conse,
        p_cg=p_cg,
        t_cg_singleband=t_cg_singleband,
        t_angle=t_angle,
        transform_mode=transform_mode,
    )
    t_cg_scale100 = chi2.ppf(p_cg, DEFAULT_BANDS) * 100
    t_angle_scale100 = t_angle * 100
    if (
        sccd_pack.nrt_mode == defaults["SCCD"]["NRT_MONITOR_SNOW"]
        or sccd_pack.nrt_mode == defaults["SCCD"]["NRT_QUEUE_SNOW"]
        or int(sccd_pack.nrt_mode / 10) == 1
    ):
        return sccd_pack, 0

    if (
        sccd_pack.nrt_model[0]["conse_last"] >= dist_conse
        and sccd_pack.nrt_model[0]["norm_cm"] > t_cg_scale100
        and sccd_pack.nrt_model[0]["cm_angle"] < t_angle_scale100
    ):
        cm_median = calculate_sccd_cm(sccd_pack)
        if (
            cm_median[2] < -t_cg_singleband
            and cm_median[3] > t_cg_singleband
            and cm_median[4] < -t_cg_singleband
        ):  # greening
            return sccd_pack, 0
        else:
            if transform_mode:
                if sccd_pack.nrt_mode == defaults["SCCD"]["NRT_MONITOR_STANDARD"]:
                    sccd_pack = sccd_pack._replace(nrt_mode=sccd_pack.nrt_mode + 10)
                # elif sccd_pack.nrt_mode == defaults["SCCD"]["NRT_MONITOR2QUEUE"]:
                #     sccd_pack = sccd_pack._replace(
                #         nrt_mode=defaults["SCCD"]["NRT_QUEUE_STANDARD"] + 10
                #     )
            return (
                sccd_pack,
                sccd_pack.nrt_model[0]["obs_date_since1982"][
                    defaults["SCCD"]["DEFAULT_CONSE"]
                    - sccd_pack.nrt_model[0]["conse_last"]
                ]
                + defaults["COMMON"]["JULIAN_LANDSAT4_LAUNCH"],
            )
    else:
        return sccd_pack, 0


def cold_detect_flex(
    dates,
    ts_stack,
    qas,
    p_cg=0.99,
    conse=6,
    pos=1,
    b_output_cm=False,
    starting_date=0,
    n_cm=0,
    cm_output_interval=0,
    gap_days=365.25,
    tmask_b1=1,
    tmask_b2=1,
    fitlam=20,
):
    """
    pixel-based COLD algorithm.
    Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., &
    Zhou, C. (2020). Continuous monitoring of land disturbance based on Landsat time series.
    Remote Sensing of Environment, 38, 111116.
    Parameters
    ----------
    dates: 1d array of shape (n_obs,), list of ordinal dates
    ts_stack: 2d array of shape (n_obs,), horizontally stacked multispectral time series.
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: probaility threshold of change magnitude, default is 0.99
    pos: position id of the pixel
    conse: consecutive observation number
    b_output_cm: bool, 'True' means outputting change magnitude and change magnitude dates, only for object-based COLD
    starting_date: the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile
                    should have the same date, only for b_output_cm is True. Only b_output_cm == 'True'
    n_cm: the length of outputted change magnitude. Only b_output_cm == 'True'
    cm_output_interval: the temporal interval of outputting change magnitudes. Only b_output_cm == 'True'
    gap_days: define the day number of the gap year for determining i_dense. Setting a large value (e.g., 1500)
                if the gap year in the middle of the time range
    tmask_b1: the first band id for tmask
    tmask_b2: the second band id for tmask
    fitlam: the lamba used for the final fitting. Won't change the detection accuracy, but will affect the outputted harmonic model
    Returns
    ----------
    change records: the COLD outputs that characterizes each temporal segment if b_output_cm==False
    Or
    [change records, cm_outputs, cm_outputs_date] if b_output_cm==True
    """

    _validate_params(
        func_name="cold_detect_flex",
        p_cg=p_cg,
        pos=pos,
        conse=conse,
        b_output_cm=b_output_cm,
        starting_date=starting_date,
        n_cm=n_cm,
        cm_output_interval=cm_output_interval,
        gap_days=gap_days,
        fitlam=fitlam,
    )

    # make sure it is c contiguous array and 64 bit
    dates, ts_stack, qas = _validate_data_flex(dates, ts_stack, qas)
    valid_num_scenes = ts_stack.shape[0]
    nbands = ts_stack.shape[1] if ts_stack.ndim > 1 else 1
    if nbands > MAX_FLEX_BAND:
        raise RuntimeError(
            f"Can't input more than {MAX_FLEX_BAND} bands ({nbands} > {MAX_FLEX_BAND})"
        )
    if (tmask_b1 > nbands) or (tmask_b2 > nbands):
        raise RuntimeError(f"tmask_b1 or tmask_b2 is larger than the input band number")
    t_cg = chi2.ppf(p_cg, nbands)
    max_t_cg = chi2.ppf(0.99999, nbands)
    rec_cg = _cold_detect_flex(
        dates,
        ts_stack.flatten(),
        qas,
        valid_num_scenes,
        nbands,
        t_cg,
        max_t_cg,
        conse,
        pos,
        b_output_cm,
        starting_date,
        n_cm,
        cm_output_interval,
        gap_days,
        tmask_b1,
        tmask_b2,
        fitlam,
    )
    # dt = np.dtype([('t_start', np.int32), ('t_end', np.int32), ('t_break', np.int32), ('pos', np.int32),
    #                ('nm_obs', np.int32), ('category', np.int16), ('change_prob', np.int16), ('change_prob', np.int16)])
    return rec_cg


def sccd_detect_flex(
    dates,
    ts_stack,
    qas,
    p_cg=0.99,
    conse=6,
    pos=1,
    b_c2=True,
    b_pinpoint=False,
    gate_pcg=0.90,
    state_intervaldays=0.0,
    tmask_b1=1,
    tmask_b2=1,
    b_fitting_coefs=False,
):
    """
    pixel-based offline SCCD algorithm.
    Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time approach for monitoring forest
    disturbance using Landsat time series: Stochastic continuous change detection.
    Remote Sensing of Environment, 252, 112167.

    Parameters
    ----------
    dates: 1d array of shape(n_obs,), list of ordinal dates
    ts_stack: 2d array of shape (n_obs, nbands), horizontally stacked multispectral time series.
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: probaility threshold of change magnitude, default is 0.99
    conse: consecutive observation number
    pos: position id of the pixel, i.e.,  (row -1) * ncols + col, row and col starts from 1
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel
                test due to its current low quality
    b_pinpoint: bool, output pinpoint break where pinpoint is an overdetection of break using conse =3
                        and threshold = gate_tcg, which are used to simulate the situation of NRT scenario and
                        for training a machine-learning model
    gate_pcg: the gate change probability threshold for defining anomaly
    state_intervaldays: the day interval for output states (only b_output_state is True)
    tmask_b1: the first band id for tmask
    tmask_b2: the second band id for tmask
    b_fitting_coefs: True indicates using curve fitting to get global harmonic coefficients, otherwise use the local coefficients
    Returns
    ----------
    SccdOutput: 1-d array of structured type, if b_pinpoint==False
    Or
    [SccdOutput, SccdReccgPinpoint], if b_pinpoint==True
    """
    _validate_params(
        func_name="sccd_detect_flex",
        p_cg=p_cg,
        pos=pos,
        conse=conse,
        b_c2=b_c2,
        b_pinpoint=b_pinpoint,
        gate_pcg=gate_pcg,
        state_intervaldays=state_intervaldays,
    )
    # make sure it is c contiguous array and 64 bit
    dates, ts_stack, qas = _validate_data_flex(dates, ts_stack, qas)
    valid_num_scenes = ts_stack.shape[0]
    nbands = ts_stack.shape[1] if ts_stack.ndim > 1 else 1
    if nbands > MAX_FLEX_BAND_SCCD:
        raise RuntimeError(
            f"Can't input more than {MAX_FLEX_BAND_SCCD} bands ({nbands} > {MAX_FLEX_BAND_SCCD})"
        )
    if (tmask_b1 > nbands) or (tmask_b2 > nbands):
        raise RuntimeError(f"tmask_b1 or tmask_b2 is larger than the input band number")

    t_cg = chi2.ppf(p_cg, nbands)
    max_t_cg = chi2.ppf(0.9999, nbands)
    gate_tcg = chi2.ppf(gate_pcg, nbands)
    # sccd_wrapper = SccdDetectWrapper()
    # tmp = copy.deepcopy(sccd_wrapper.sccd_detect(dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, t_cg,
    #                                 pos, conse, b_c2, b_pinpoint, gate_tcg, b_monitor_init))
    # return tmp
    b_output_state = False if state_intervaldays == 0 else True

    return _sccd_detect_flex(
        dates,
        ts_stack.flatten(),
        qas,
        valid_num_scenes,
        nbands,
        t_cg,
        max_t_cg,
        conse,
        pos,
        b_c2,
        b_pinpoint,
        gate_tcg,
        0,
        b_output_state,
        state_intervaldays,
        tmask_b1,
        tmask_b2,
        b_fitting_coefs,
    )


def sccd_update_flex(
    sccd_pack,
    dates,
    ts_stack,
    qas,
    p_cg=0.99,
    conse=6,
    pos=1,
    b_c2=True,
    gate_pcg=0.90,
    predictability_pcg=0.90,
    tmask_b1=1,
    tmask_b2=1,
):
    """
    SCCD online update for new observations
    Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time approach for monitoring forest
    disturbance using Landsat time series: Stochastic continuous change detection.
    Remote Sensing of Environment, 252, 112167.

    Parameters
    ----------
    sccd_pack: namedtuple
    dates: 1-d array of shape(n_obs,), list of ordinal dates
    ts_stack: 2-d array of shape (n_obs,), horizontally stacked multispectral time series.
    qas: 1-d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: float, probability threshold of change magnitude, by default 0.99
    conse: int, consecutive observation number, by default 6
    pos: int, position id of the pixel, i.e.,  (row -1) * ncols + col, row and col starts from 1, by default 1
    b_c2: bool, indicate if the dataset is collection 2. C2 needs ignoring thermal band for valid pixel
                test due to its current low quality, by default True
    gate_pcg: float, the gate change magnitude probability threshold for defining anomalies, by default 0.90
    predictability_pcg: float, the probability threshold for predictability test, by default 0.90
    tmask_b1: int, the first band id for tmask, by default 1
    tmask_b2: int, the second band id for tmask, by default 1
    Returns
    ----------
    change records: the SCCD outputs that characterizes each temporal segment
    """
    # if not isinstance(sccd_pack, SccdOutput):
    #     raise ValueError("The type of sccd_pack has to be namedtuple 'SccdOutput'!")

    _validate_params(
        func_name="sccd_update_flex",
        p_cg=p_cg,
        pos=pos,
        conse=conse,
        b_c2=b_c2,
        b_pinpoint=False,
        gate_pcg=gate_pcg,
        predictability_pcg=predictability_pcg,
    )

    dates, ts_stack, qas = _validate_data_flex(dates, ts_stack, qas)
    valid_num_scenes = ts_stack.shape[0]
    nbands = ts_stack.shape[1] if ts_stack.ndim > 1 else 1
    if nbands > MAX_FLEX_BAND_SCCD:
        raise RuntimeError(
            f"Can't input more than {MAX_FLEX_BAND_SCCD} bands ({nbands} > {MAX_FLEX_BAND_SCCD})"
        )
    if (tmask_b1 > nbands) or (tmask_b2 > nbands):
        raise RuntimeError(f"tmask_b1 or tmask_b2 is larger than the input band number")

    t_cg = chi2.ppf(p_cg, nbands)
    max_t_cg = chi2.ppf(0.9999, nbands)
    gate_tcg = chi2.ppf(gate_pcg, nbands)
    predictability_tcg = chi2.ppf(predictability_pcg, nbands)

    return _sccd_detect_flex(
        sccd_pack,
        dates,
        ts_stack.flatten(),
        qas,
        valid_num_scenes,
        nbands,
        t_cg,
        max_t_cg,
        conse,
        pos,
        b_c2,
        False,
        gate_tcg,
        predictability_tcg,
        False,
        0,
        tmask_b1,
        tmask_b2,
    )
