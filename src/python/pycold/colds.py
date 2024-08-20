from ._colds_cython import _sccd_update, _sccd_detect, _obcold_reconstruct, _cold_detect, _cold_detect_flex
import numpy as np
from .common import SccdOutput
from ._param_validation import (
    validate_parameter_constraints,
    Interval,
    Integral,
    Real,
    check_consistent_length,
    check_1d,
)
from .utils import calculate_sccd_cm
from .app import defaults

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
    "gate_tcg": [Interval(Real, 0.0, None, closed="neither")],
    "predictability_tcg": [Interval(Real, 0.0, None, closed="neither")],
    "dist_conse": [Interval(Integral, 0, 6, closed="right")],
    "t_cg_scale100": [Interval(Real, 0.0, None, closed="neither")],
    "t_cg_singleband_scale1000": [Interval(Real, None, None, closed="neither")],
    "t_angle_scale100": [Interval(Integral, 0, 18000, closed="neither")],
    "transform_mode": ["boolean"],
    "state_intervaldays": [Interval(Real, 0.0, None, closed="left")]
}

NUM_FC = 40  # define the maximum number of outputted curves
NUM_FC_SCCD = 40
NUM_NRT_QUEUE = 240
MAX_FLEX_BAND = 10


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


def _validate_data(dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates=None):
    """
    validate and forcibly change the data format
    Parameters
    ----------
    dates: 1d array of shape(observation numbers), list of ordinal dates
    ts_b: 1d array of shape(observation numbers), time series of blue band.
    ts_g: 1d array of shape(observation numbers), time series of green band
    ts_r: 1d array of shape(observation numbers), time series of red band
    ts_n: 1d array of shape(observation numbers), time series of nir band
    ts_s1: 1d array of shape(observation numbers), time series of swir1 band
    ts_s2: 1d array of shape(observation numbers), time series of swir2 band
    ts_t: 1d array of shape(observation numbers), time series of thermal band
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
    else:
        return dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas


def _validate_data_flex(dates, ts_data, qas):
    """
    validate and forcibly change the data format
    Parameters
    ----------
    dates: 1d array of shape(observation numbers), list of ordinal dates
    ts_data: 2d array of shape(observation numbers), time series stack for inputs
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
#     parameters: pycold parameters
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
#     parameters: pycold parameters
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
    t_cg=15.0863,
    conse=6,
    pos=1,
    b_output_cm=False,
    starting_date=0,
    n_cm=0,
    cm_output_interval=0,
    b_c2=False,
    gap_days=365.25,
):
    """
    pixel-based COLD algorithm.
    Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., &
    Zhou, C. (2020). Continuous monitoring of land disturbance based on Landsat time series.
    Remote Sensing of Environment, 38, 111116.
    Parameters
    ----------
    dates: 1d array of shape(observation numbers), list of ordinal dates
    ts_b: 1d array of shape(observation numbers), time series of blue band.
    ts_g: 1d array of shape(observation numbers), time series of green band
    ts_r: 1d array of shape(observation numbers), time series of red band
    ts_n: 1d array of shape(observation numbers), time series of nir band
    ts_s1: 1d array of shape(observation numbers), time series of swir1 band
    ts_s2: 1d array of shape(observation numbers), time series of swir2 band
    ts_t: 1d array of shape(observation numbers), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    t_cg: threshold of change magnitude, default is chi2.ppf(0.99,5)
    pos: position id of the pixel
    conse: consecutive observation number
    b_output_cm: bool, 'True' means outputting change magnitude and change magnitude dates, only for object-based COLD
    starting_date: the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile
                    should have the same date, only for b_output_cm is True. Only b_output_cm == 'True'
    n_cm: the length of outputted change magnitude. Only b_output_cm == 'True'
    cm_output_interval: the temporal interval of outputting change magnitudes. Only b_output_cm == 'True'
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel
          test due to the current low quality
    gap_days: define the day number of the gap year for determining i_dense. Setting a large value (e.g., 1500)
                if the gap year in the middle of the time range

    Returns
    ----------
    change records: the COLD outputs that characterizes each temporal segment if b_output_cm==False
    Or
    [change records, cm_outputs, cm_outputs_date] if b_output_cm==True
    """

    _validate_params(
        func_name="cold_detect",
        t_cg=t_cg,
        pos=pos,
        conse=conse,
        b_output_cm=b_output_cm,
        starting_date=starting_date,
        n_cm=n_cm,
        cm_output_interval=cm_output_interval,
        b_c2=b_c2,
        gap_days=gap_days,
    )

    # make sure it is c contiguous array and 64 bit
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas = _validate_data(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas
    )

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
    )


def obcold_reconstruct(
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates, pos=1, conse=6, b_c2=False
):
    """
    re-contructructing change records using break dates.
    Parameters
    ----------
    dates: 1d array of shape(observation numbers), list of ordinal dates
    ts_b: 1d array of shape(observation numbers), time series of blue band.
    ts_g: 1d array of shape(observation numbers), time series of green band
    ts_r: 1d array of shape(observation numbers), time series of red band
    ts_n: 1d array of shape(observation numbers), time series of nir band
    ts_s1: 1d array of shape(observation numbers), time series of swir1 band
    ts_s2: 1d array of shape(observation numbers), time series of swir2 band
    ts_t: 1d array of shape(observation numbers), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    break_dates: 1d array, the break dates obtained from other procedures such as obia
    conse: consecutive observation number (for calculating change magnitudes)
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test
          due to its current low quality

    Returns
    ----------
    change records: the COLD outputs that characterizes each temporal segment
    """
    _validate_params(func_name="sccd_detect", pos=pos, conse=conse, b_c2=b_c2)
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates = _validate_data(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates
    )

    return _obcold_reconstruct(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, break_dates, pos, conse, b_c2
    )


def sccd_detect(
    dates,
    ts_b,
    ts_g,
    ts_r,
    ts_n,
    ts_s1,
    ts_s2,
    ts_t,
    qas,
    t_cg=15.0863,
    conse=6,
    pos=1,
    b_c2=False,
    b_pinpoint=False,
    gate_tcg=9.236,
    state_intervaldays=0.0
):
    """
    pixel-based offline SCCD algorithm.
    Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time approach for monitoring forest
    disturbance using Landsat time series: Stochastic continuous change detection.
    Remote Sensing of Environment, 252, 112167.

    Parameters
    ----------
    dates: 1d array of shape(observation numbers), list of ordinal dates
    ts_b: 1d array of shape(observation numbers), time series of blue band.
    ts_g: 1d array of shape(observation numbers), time series of green band
    ts_r: 1d array of shape(observation numbers), time series of red band
    ts_n: 1d array of shape(observation numbers), time series of nir band
    ts_s1: 1d array of shape(observation numbers), time series of swir1 band
    ts_s2: 1d array of shape(observation numbers), time series of swir2 band
    ts_t: 1d array of shape(observation numbers), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    t_cg: threshold of change magnitude, default is chi2.ppf(0.99,5)
    pos: position id of the pixel, i.e.,  (row -1) * ncols + col, row and col starts from 1
    conse: consecutive observation number
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel
                test due to its current low quality
    b_pinpoint: bool, output pinpoint break where pinpoint is an overdetection of break using conse =3
                        and threshold = gate_tcg, which are used to simulate the situation of NRT scenario and
                        for training a machine-learning model
    gate_tcg: the gate change magnitude threshold for defining anomaly
    b_output_state: indicate whether to output state variables
    state_intervaldays: the day interval for output states (only b_output_state is True)
    Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands
    Returns
    ----------
    SccdOutput: the SCCD outputs that characterizes each temporal segment if b_pinpoint==False
    Or
    [SccdOutput, SccdReccgPinpoint]: b_pinpoint==True
    """
    _validate_params(
        func_name="sccd_detect",
        t_cg=t_cg,
        pos=pos,
        conse=conse,
        b_c2=b_c2,
        b_pinpoint=b_pinpoint,
        gate_tcg=gate_tcg,
        state_intervaldays=state_intervaldays
    )
    # make sure it is c contiguous array and 64 bit
    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas = _validate_data(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas
    )

    # sccd_wrapper = SccdDetectWrapper()
    # tmp = copy.deepcopy(sccd_wrapper.sccd_detect(dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas, t_cg,
    #                                 pos, conse, b_c2, b_pinpoint, gate_tcg, b_monitor_init))
    # return tmp
    if state_intervaldays == 0:
        b_output_state = False;
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
        state_intervaldays
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
    ts_t,
    qas,
    t_cg=15.0863,
    conse=6,
    pos=1,
    b_c2=False,
    gate_tcg=9.236,
    predictability_tcg=9.236,
):
    """
    SCCD online update for new observations
    Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A near-real-time approach for monitoring forest
    disturbance using Landsat time series: Stochastic continuous change detection.
    Remote Sensing of Environment, 252, 112167.

    Parameters
    ----------
    sccd_pack:
    dates: 1d array of shape(observation numbers), list of ordinal dates
    ts_b: 1d array of shape(observation numbers), time series of blue band.
    ts_g: 1d array of shape(observation numbers), time series of green band
    ts_r: 1d array of shape(observation numbers), time series of red band
    ts_n: 1d array of shape(observation numbers), time series of nir band
    ts_s1: 1d array of shape(observation numbers), time series of swir1 band
    ts_s2: 1d array of shape(observation numbers), time series of swir2 band
    ts_t: 1d array of shape(observation numbers), time series of thermal band
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    t_cg: threshold of change magnitude, default is chi2.ppf(0.99,5)
    pos: position id of the pixel, i.e.,  (row -1) * ncols + col, row and col starts from 1
    conse: consecutive observation number
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel
                test due to its current low quality
    gate_tcg: the gate change magnitude threshold for defining anomaly
    predictability_tcg: the threshold for predictability test
    Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands
    Returns
    ----------
    change records: the SCCD outputs that characterizes each temporal segment
    """
    # if not isinstance(sccd_pack, SccdOutput):
    #     raise ValueError("The type of sccd_pack has to be namedtuple 'SccdOutput'!")

    _validate_params(
        func_name="sccd_update",
        t_cg=t_cg,
        pos=pos,
        conse=conse,
        b_c2=b_c2,
        b_pinpoint=False,
        gate_tcg=gate_tcg,
        predictability_tcg=predictability_tcg,
    )

    dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas = _validate_data(
        dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas
    )

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
        b_c2,
        gate_tcg,
        predictability_tcg
    )


def sccd_identify(
    sccd_pack,
    dist_conse=6,
    t_cg_scale100=1508.6,
    t_cg_singleband_scale1000=-200,
    t_angle_scale100=4500,
    transform_mode=True,
):
    """
    identify disturbance date from sccd_pack
    Parameters
    ----------
    sccd_pack: sccd data structure
    dist_conse: user-specify disturbance conse
    t_cg_scale100: user-specify change magnitudes (scale by 100)
    t_cg_singleband_scale1000: user-specify change magnitude for each band to identify greener direction
      see Eq. 10 in Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., & Zhou, C. (2020).
      Continuous monitoring of land disturbance based on Landsat time series. Remote Sensing of Environment, 238, 111116.
    t_angle_scale100: (scale by 100)
    transform_mode: true -> transform the mode to untested predictability once the change has been detected
    Returns
    -------
    0 (no disturbance) or the ordinal date of disturbance occurrence
    """
    _validate_params(
        func_name="sccd_identify",
        dist_conse=dist_conse,
        t_cg_scale100=t_cg_scale100,
        t_cg_singleband_scale1000=t_cg_singleband_scale1000,
        t_angle_scale100=t_angle_scale100,
        transform_mode=transform_mode,
    )
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
            cm_median[2] < -t_cg_singleband_scale1000
            and cm_median[3] > t_cg_singleband_scale1000
            and cm_median[4] < -t_cg_singleband_scale1000
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
                    defaults["SCCD"]["DEFAULT_CONSE"] - sccd_pack.nrt_model[0]["conse_last"]
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
    tmask_b2=1
):
    """
    pixel-based COLD algorithm.
    Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., &
    Zhou, C. (2020). Continuous monitoring of land disturbance based on Landsat time series.
    Remote Sensing of Environment, 38, 111116.
    Parameters
    ----------
    dates: 1d array of shape (observation numbers), list of ordinal dates
    ts_stack: 2d array of shape (observation numbers), horizontally stacked multispectral time series.
    qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    t_cg: threshold of change magnitude, default is chi2.ppf(0.99,5)
    pos: position id of the pixel
    conse: consecutive observation number
    b_output_cm: bool, 'True' means outputting change magnitude and change magnitude dates, only for object-based COLD
    starting_date: the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile
                    should have the same date, only for b_output_cm is True. Only b_output_cm == 'True'
    n_cm: the length of outputted change magnitude. Only b_output_cm == 'True'
    cm_output_interval: the temporal interval of outputting change magnitudes. Only b_output_cm == 'True'
    b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel
          test due to the current low quality
    gap_days: define the day number of the gap year for determining i_dense. Setting a large value (e.g., 1500)
                if the gap year in the middle of the time range

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
    )

    # make sure it is c contiguous array and 64 bit
    dates, ts_stack, qas = _validate_data_flex(
        dates, ts_stack, qas
    )
    valid_num_scenes = ts_stack.shape[0]
    nbands = ts_stack.shape[1] if ts_stack.ndim > 1 else 1
    if nbands > MAX_FLEX_BAND:
        raise RuntimeError(f"Can't input more than {MAX_FLEX_BAND} bands ({nbands} > {MAX_FLEX_BAND})")
        
    rec_cg = _cold_detect_flex(
        dates,
        ts_stack.flatten(),
        qas,
        valid_num_scenes,
        nbands,
        p_cg,
        conse,
        pos,
        b_output_cm,
        starting_date,
        n_cm,
        cm_output_interval,
        gap_days,
        tmask_b1,
        tmask_b2)
    # dt = np.dtype([('t_start', np.int32), ('t_end', np.int32), ('t_break', np.int32), ('pos', np.int32), 
    #                ('nm_obs', np.int32), ('category', np.int16), ('change_prob', np.int16), ('change_prob', np.int16)]) 
    return rec_cg
