from ._ccd_cython import (
    _sccd_update,
    _sccd_detect,
    _obcold_reconstruct,
    _cold_detect,
    _cold_detect_flex,
    _sccd_detect_flex,
    _sccd_update_flex,
)
import numpy
from .common import SccdOutput, rec_cg
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
import pandas as pd

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
    "lam": [Interval(Real, 0.0, None, closed="left")],
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

    dates = dates.astype(dtype=numpy.int64, order="C")
    ts_b = ts_b.astype(dtype=numpy.int64, order="C")
    ts_g = ts_g.astype(dtype=numpy.int64, order="C")
    ts_r = ts_r.astype(dtype=numpy.int64, order="C")
    ts_n = ts_n.astype(dtype=numpy.int64, order="C")
    ts_s1 = ts_s1.astype(dtype=numpy.int64, order="C")
    ts_s2 = ts_s2.astype(dtype=numpy.int64, order="C")
    ts_t = ts_t.astype(dtype=numpy.int64, order="C")
    qas = qas.astype(dtype=numpy.int64, order="C")
    if break_dates is not None:
        break_dates = break_dates.astype(dtype=numpy.int64, order="C")
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

    dates = dates.astype(dtype=numpy.int64, order="C")
    ts_data = ts_data.astype(dtype=numpy.int64, order="C")
    qas = qas.astype(dtype=numpy.int64, order="C")
    return dates, ts_data, qas


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
    lam=20,
):
    """running pixel-based COLD algorithm.

    Parameters
    ----------
    dates: numpy.ndarray
        1d time series of ordinal dates of shape(n_obs,)
    ts_b: numpy.ndarray
        1d time series of blue band of shape(n_obs,)
    ts_g: numpy.ndarray
        1d time series of green band of shape(n_obs,)
    ts_r: numpy.ndarray
        1d time series of red band of shape(n_obs,)
    ts_n: numpy.ndarray
        1d time series of nir band of shape(n_obs,)
    ts_s1: numpy.ndarray
        1d time series of swir1 band of shape(n_obs,)
    ts_s2: numpy.ndarray
        1d time series of swir2 band of shape(n_obs,)
    ts_t: numpy.ndarray
        1d time series of thermal band of shape(n_obs,)
    qas: numpy.ndarray
        1d time series of QA cfmask band of shape(n_obs,). '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: float
        probability threshold of change magnitude, by default 0.99
    conse: int
        consecutive observation number, by default 6
    pos: int
        position id of the pixel, by default 1
    b_output_cm: bool
        'True' means outputting change magnitude and change magnitude dates (OB-COLD), i.e., (cold_results, cm_outputs, cm_outputs_date); 'False' will output only cold_results
    starting_date: int
        the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile. should have the same date, only for b_output_cm is True. by default 0.
    n_cm: int
        length of outputted change magnitude. Only b_output_cm == 'True'. by default 0.
    cm_output_interval: int
        temporal interval of outputting change magnitudes. Only b_output_cm == 'True'. by default 0.
    b_c2: bool
        a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to the current low quality. by default True
    gap_days: int
        define the day number of the gap year for determining i_dense. The COLD will skip the i_dense days to set the starting point of the model. Setting a large value (e.g., 1500) if the gap year is in the middle of the time range. by default 365.25.
    lam: float
        The lamba parameter used for lasso fitting that controls the regularization of the regression model. When lambda is 0, it is OLS regression.For landsat-like images (i.e., range is [0, 10000]), lambda is suggested to be 20.

    Returns
    -------
    numpy.ndarray | (numpy.ndarray, numpy.ndarray, numpy.ndarray)

        If  b_output_cm is False, a structured array of dtype = :py:type:`~pyxccd.common.cold_rec_cg` is returnd;
        if b_output_cm is True, a tuple (cold_results, cm_outputs, cm_outputs_date) will be returned

            cold_results: numpy.ndarray
                A structured array of dtype = :py:type:`~pyxccd.common.cold_rec_cg`

            cm_outputs: numpy.ndarray
                Change magnitude list, shape (n_cm,)

            cm_outputs_date: numpy.ndarray
                Change magnitude date list, shape (n_cm,)
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
        lam=lam,
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
        lam,
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
    conse=6,
    pos=1,
    b_c2=True,
    lam=20,
):
    """re-contructructing change records using break dates.

    Parameters
    ----------
    dates: numpy.ndarray
        1d time series of ordinal dates of shape(n_obs,)
    ts_b: numpy.ndarray
        1d time series of blue band of shape(n_obs,)
    ts_g: numpy.ndarray
        1d time series of green band of shape(n_obs,)
    ts_r: numpy.ndarray
        1d time series of red band of shape(n_obs,)
    ts_n: numpy.ndarray
        1d time series of nir band of shape(n_obs,)
    ts_s1: numpy.ndarray
        1d time series of swir1 band of shape(n_obs,)
    ts_s2: numpy.ndarray
        1d time series of swir2 band of shape(n_obs,)
    ts_t: numpy.ndarray
        1d time series of thermal band of shape(n_obs,)
    qas: numpy.ndarray
        1d time series of QA cfmask band of shape(n_obs,). '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    break_dates: numpy.ndarray
        1d time series of break dates obtained from other procedures such as obia
    conse: int
        consecutive observation number, by default 6
    pos: int
        position id of the pixel, by default 1
    b_c2: bool
        a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to the current low quality. by default True
    lam: float
        The lamba parameter used for lasso fitting that controls the regularization of the regression model. When lambda is 0, it is OLS regression.For landsat-like images (i.e., range is [0, 10000]), lambda is suggested to be 20.

    Returns
    -------
    numpy.ndarray

        cold_results: numpy.ndarray
            A structured array of dtype = :py:type:`~pyxccd.common.cold_rec_cg`
    """
    _validate_params(func_name="obcold_construct", pos=pos, conse=conse, b_c2=b_c2)
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
        lam,
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
    lam=20,
):
    """Offline SCCD algorithm for processing historical time series.


    Parameters
    ----------
    dates: numpy.ndarray
        1d time series of ordinal dates of shape(n_obs,)
    ts_b: numpy.ndarray
        1d time series of blue band of shape(n_obs,)
    ts_g: numpy.ndarray
        1d time series of green band of shape(n_obs,)
    ts_r: numpy.ndarray
        1d time series of red band of shape(n_obs,)
    ts_n: numpy.ndarray
        1d time series of nir band of shape(n_obs,)
    ts_s1: numpy.ndarray
        1d time series of swir1 band of shape(n_obs,)
    ts_s2: numpy.ndarray
        1d time series of swir2 band of shape(n_obs,)
    qas: numpy.ndarray
        1d time series of QA cfmask band of shape(n_obs,). '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: float
        Probability threshold of change magnitude, by default 0.99
    conse: int
        Consecutive observation number, by default 6
    pos: int
        Position id of the pixel, by default 1
    b_c2: bool
        A temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid
        pixel test due to the current low quality. by default True
    b_pinpoint: bool
        If true, output pinpoint breaks where a pinpoint is an overdetection of break using conse 3 and threshold = gate_tcg,
        which overdetects anomalies to simulate the situation of NRT scenario and for training a retrospective model, by default False.
        Note that pinpoints is a type of breaks that do not trigger model initialization, against structural breaks (i.e., normal breaks).
    gate_pcg: float
        Change probability threshold for defining spectral anomalies (for NRT)/pinpoints, by default 0.90.
    state_intervaldays: float
        If larger than 0, output states at a day interval of state_intervaldays, by default 0.0 (meaning that no states will be outputted). For more details, refer to state-space models (e.g., http://www.scholarpedia.org/article/State_space_model)
    b_fitting_coefs: bool
        If True, use curve fitting to get harmonic coefficients for the temporal segment, otherwise use the local coefficients from kalman filter, by default False.
    lam: float
        The lamba parameter used for lasso fitting that controls the regularization of the regression model. When lambda is 0, it is OLS regression.For landsat-like images (i.e., range is [0, 10000]), lambda is suggested to be 20.
    Returns
    -------
    :py:type:`~pyxccd.common.SccdOutput` | (:py:type:`~pyxccd.common.SccdOutput`, pd.DataFrame) | (:py:type:`~pyxccd.common.SccdOutput`, numpy.ndarray)

        If b_pinpoint is False and b_output_state is False, sccdoutput will be returned (by default);
        if b_pinpoint is False and b_output_state is True,  (sccdoutput, states_info) will be returned;
        if b_pinpoint is True, (sccdoutput, pinpoints) will be returned

            sccdoutput: :py:type:`~pyxccd.common.SccdOutput`
                A namedtuple (position, rec_cg, min_rmse, nrt_mode, nrt_model, nrt_queue)

            states_info: pd.DataFrame
                A table of three state time series (trend, annual, semiannual) for six inputted spectral bands

            pinpoints: numpy.ndarray
                A structured array of dtype = :py:type:`~pyxccd.common.pinpoint`

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
        lam=lam,
    )
    ts_t = numpy.zeros(ts_b.shape)
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
        lam,
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
    lam=20,
):
    """
    SCCD online update for new observations

    Parameters
    ----------
    sccd_pack: :py:type:`~pyxccd.common.SccdOutput`
        The SCCD results outputted by the last process
    dates: numpy.ndarray
        1d new time series of ordinal dates of shape(n_obs,)
    ts_b: numpy.ndarray
        1d new time series of blue band of shape(n_obs,)
    ts_g: numpy.ndarray
        1d new time series of green band of shape(n_obs,)
    ts_r: numpy.ndarray
        1d new time series of red band of shape(n_obs,)
    ts_n: numpy.ndarray
        1d new time series of nir band of shape(n_obs,)
    ts_s1: numpy.ndarray
        1d new time series of swir1 band of shape(n_obs,)
    ts_s2: numpy.ndarray
        1d new time series of swir2 band of shape(n_obs,)
    qas: numpy.ndarray
        1d new time series of QA cfmask band of shape(n_obs,). '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    p_cg: float
        probability threshold of change magnitude, by default 0.99
    conse: int
        consecutive observation number, by default 6
    pos: int
        position id of the pixel, by default 1
    gate_pcg: float
        change probability threshold for defining spectral anomalies (for NRT)/pinpoints, by default 0.90.
    predictability_pcg: float
        probability threshold for predictability test. If not passed, the nrt_mode will return 11. by default 0.90.
    lam: float
        The lamba parameter used for lasso fitting that controls the regularization of the regression model. When lambda is 0, it is OLS regression.For landsat-like images (i.e., range is [0, 10000]), lambda is suggested to be 20.
    Returns
    ----------
    :py:type:`~pyxccd.common.SccdOutput`
        A namedtuple (position, rec_cg, min_rmse, nrt_mode, nrt_model, nrt_queue)

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
        lam=lam,
    )
    ts_t = numpy.zeros(ts_b.shape)
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
        lam,
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
    identify disturbance date for the current monitoring model from SccdOutput

    Parameters
    ----------
    sccd_pack: :py:type:`~pyxccd.common.SccdOutput`
        S-CCD output
    dist_conse: int
        Minimum consecutive anomaly number required to identify disturbance, by default 6
    p_cg: float
        Change magnitude probability threshold to identify disturbance, by default 0.99
    t_cg_singleband: float
        single-band change magnitude to identify greenning breaks, by default -200
      see Eq. 10 in Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., & Zhou, C. (2020). Continuous monitoring of land disturbance based on Landsat time series. Remote Sensing of Environment, 238, 111116.
    t_angle_scale100: float
        Threshold for included angle (scale by 100), by default 45
    transform_mode: bool
        Transform the mode to untested predictability once the change has been detected. Default isFalse
    Returns
    -------
    tuple
        (:py:type:`~pyxccd.common.SccdOutput`, int)
            The return date is either 0 (no disturbance) or the ordinal date of disturbance occurrence
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
    lam,
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
):
    """running pixel-based COLD algorithm for any band combination (flexible mode).

    Parameters
    ----------
    dates: numpy.ndarray
        1d time series of ordinal dates of shape(n_obs,)
    ts_stack: numpy.ndarray
        2d array of shape (n_obs, nbands), horizontally stacked multispectral time series.
    qas: numpy.ndarray
        1d time series of QA cfmask band of shape(n_obs,). '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    lam: float
        The lamba parameter used for lasso fitting that controls the regularization of the regression model. When lambda is 0, it is OLS regression.For landsat-like images (i.e., range is [0, 10000]), lambda is suggested to be 20.
    p_cg: float
        Probability threshold of change magnitude, by default 0.99
    conse: int
        Consecutive observation number, by default 6
    pos: int
        position id of the pixel, by default 1
    b_output_cm: bool
        'True' means outputting change magnitude and change magnitude dates (OB-COLD), i.e.,
        (cold_results, cm_outputs, cm_outputs_date); 'False' will output only cold_results
    starting_date: int
        The starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile should have the same date, only for b_output_cm is True. by default 0.
    n_cm: int
        Length of outputted change magnitude. Only b_output_cm == 'True'. by default 0.
    cm_output_interval: int
        Temporal interval of outputting change magnitudes. Only b_output_cm == 'True'. by default 0.
    b_c2: bool
        Indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to the current low quality. by default True
    gap_days: int
        Define the day number of the gap year for determining i_dense. The COLD will skip the i_dense days to set the starting point of the model. Setting a large value (e.g., 1500) if the gap year is in the middle of the time range. by default 365.25.

    Returns
    -------
    numpy.ndarray | (numpy.ndarray, numpy.ndarray, numpy.ndarray)

        If  b_output_cm is False, a structured array of dtype = :py:type:`~pyxccd.common.cold_rec_cg` is returnd;
        if b_output_cm is True, a tuple (cold_results, cm_outputs, cm_outputs_date) will be returned

            cold_results: numpy.ndarray
                A structured array of dtype = :py:type:`~pyxccd.common.cold_rec_cg`

            cm_outputs: numpy.ndarray
                Change magnitude list, shape (n_cm,)

            cm_outputs_date: numpy.ndarray
                Change magnitude date list, shape (n_cm,)
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
        lam=lam,
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
        raise RuntimeError("tmask_b1 or tmask_b2 is larger than the input band number")
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
        lam,
    )
    # dt = numpy.dtype([('t_start', numpy.int32), ('t_end', numpy.int32), ('t_break', numpy.int32), ('pos', numpy.int32),
    #                ('nm_obs', numpy.int32), ('category', numpy.int16), ('change_prob', numpy.int16), ('change_prob', numpy.int16)])
    return rec_cg


def sccd_detect_flex(
    dates,
    ts_stack,
    qas,
    lam,
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
    Offline SCCD algorithm for processing historical time series for any band combination.


    Parameters
    ----------
    dates: numpy.ndarray
        1d time series of ordinal dates of shape(n_obs,)
    ts_stack: numpy.ndarray
        2d array of shape (n_obs, nbands), horizontally stacked multispectral time series. The maximum band number is 10..
    qas: numpy.ndarray
        1d time series of QA cfmask band of shape(n_obs,). '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    lam: float
        The lamba parameter used for lasso fitting that controls the regularization of the regression model. When lambda is 0, it is OLS regression.For landsat-like images (i.e., range is [0, 10000]), lambda is suggested to be 20.
    p_cg: float
        Probability threshold of change magnitude, by default 0.99
    conse: int
        Consecutive observation number, by default 6
    pos: int
        Position id of the pixel, by default 1
    b_c2: bool
        A temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid
        pixel test due to the current low quality. by default True
    b_pinpoint: bool
        If true, output pinpoint breaks where a pinpoint is an overdetection of break using conse 3 and threshold = gate_tcg, which overdetects anomalies to simulate the situation of NRT scenario and for training a retrospective model, by default False.
        Note that pinpoints is a type of breaks that do not trigger model initialization, against structural breaks (i.e., normal breaks).
    gate_pcg: float
        Change probability threshold for defining spectral anomalies (for NRT)/pinpoints, by default 0.90.
    state_intervaldays: float
        If larger than 0, output states at a day interval of state_intervaldays, by default 0.0 (meaning that no states will be outputted). For more details, refer to state-space models (e.g., http://www.scholarpedia.org/article/State_space_model)
    b_fitting_coefs: bool
        If True, use curve fitting to get harmonic coefficients for the temporal segment, otherwise use the local coefficients from kalman filter, by default False.
    tmask_b1: int
        The first band id for tmask. Started from 1.
    tmask_b2: int
        The second band id for tmask. Started from 1.
    b_fitting_coefs: bool
        True indicates using curve fitting to get global harmonic coefficients, otherwise use the local coefficients. Default it False.

    Returns
    ----------
    :py:type:`~pyxccd.common.SccdOutput` | (:py:type:`~pyxccd.common.SccdOutput`, pd.DataFrame)

    If b_output_state is False, sccdoutput will be returned (by default);
    if b_output_state is True,  (sccdoutput, states_info) will be returned;
        sccdoutput: :py:type:`~pyxccd.common.SccdOutput`
            A namedtuple (position, rec_cg, min_rmse, nrt_mode, nrt_model, nrt_queue)

        states_info: pd.DataFrame
            A table of three state time series (trend, annual, semiannual) for nbands inputted spectral bands
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
        lam=lam,
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
        lam,
    )


def sccd_update_flex(
    sccd_pack,
    dates,
    ts_stack,
    qas,
    lam,
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
    SCCD online update for new observations for any band combination

    Parameters
    ----------
    sccd_pack: namedtuple
    dates: numpy.ndarray
        1d new time series of ordinal dates of shape(n_obs,)
    ts_stack: numpy.ndarray
        2d array of shape (n_obs,), horizontally stacked multispectral time series. The maximum band number is 10.
    qas: numpy.ndarray
        1d new time series of QA cfmask band of shape(n_obs,). '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
    lam: float
        The lamba parameter used for lasso fitting that controls the regularization of the regression model. When lambda is 0, it is OLS regression.For landsat-like images (i.e., range is [0, 10000]), lambda is suggested to be 20.
    p_cg: float
        Probability threshold of change magnitude, by default 0.99
    conse: int
        Consecutive observation number, by default 6
    pos: int
        Position id of the pixel, by default 1
    gate_pcg: float
        Change probability threshold for defining spectral anomalies (for NRT)/pinpoints, by default 0.90.
    predictability_pcg: float
        Probability threshold for predictability test. If not passed, the nrt_mode will return 11. by default 0.90.
    Returns
    ----------
    :py:type:`~pyxccd.common.SccdOutput`
        A namedtuple (position, rec_cg, min_rmse, nrt_mode, nrt_model, nrt_queue)

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
        lam=lam,
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

    return _sccd_update_flex(
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
        gate_tcg,
        predictability_tcg,
        tmask_b1,
        tmask_b2,
        lam,
    )
