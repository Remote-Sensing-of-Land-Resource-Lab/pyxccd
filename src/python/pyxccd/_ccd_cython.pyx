# from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen
import numpy as np
import pandas as pd
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
import math
from libcpp cimport bool
from collections import namedtuple
from copy import deepcopy
cimport cython
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
# instead of int32_t and int64_t for cross-platform compatibility,see https://github.com/ansys/pymapdl/issues/14
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from .common import reccg_dt, sccd_dt, nrtqueue_dt, nrtmodel_dt, anomaly_dt, cold_dt_flex, sccd_dt_flex, nrtqueue_dt_flex, nrtmodel_dt_flex, anomaly_dt_flex, _update_nrt_model,_update_nrtqueue,_update_sccd_reccg,update_anomaly, _update_cold_reccg

np.import_array()
from scipy.stats import chi2

try:
    import typing
    import dataclasses
except ImportError:
    pass  # The modules don't actually have to exists for Cython to use them as annotations

cdef int32_t NUM_FC = 40  # define the maximum number of outputted curves
cdef int32_t NUM_FC_SCCD = 40
cdef int32_t NUM_NRT_QUEUE = 240
DEF DEFAULT_CONSE = 8
DEF NRT_BAND = 6
DEF SCCD_NUM_C = 6
DEF FLEX_SCCD_NUM_C = 8
DEF TOTAL_BAND_FLEX_NRT = 10
DEF TOTAL_BAND_FLEX = 10


cdef extern from "../../cxx/output.h":
    ctypedef struct Output_t:
        int32_t t_start
        int32_t t_end
        int32_t t_break
        int32_t pos
        int32_t num_obs
        short int category
        short int change_prob
        float coefs[7][8]
        float rmse[7]
        float magnitude[7]

cdef extern from "../../cxx/output.h":
    ctypedef struct Output_sccd:
        int32_t t_start
        int32_t t_break
        int32_t num_obs
        float coefs[NRT_BAND][SCCD_NUM_C]
        float rmse[NRT_BAND]
        float magnitude[NRT_BAND]

cdef extern from "../../cxx/output.h":
    ctypedef struct output_nrtqueue:
        short int clry[NRT_BAND]
        short int clrx_since1982
 
cdef extern from "../../cxx/output.h":
    ctypedef struct output_nrtmodel:
        short int t_start_since1982
        short int num_obs
        short int obs[NRT_BAND][DEFAULT_CONSE]
        short int obs_date_since1982[DEFAULT_CONSE]
        float covariance[NRT_BAND][36]
        float nrt_coefs[NRT_BAND][SCCD_NUM_C]
        float H[NRT_BAND]
        uint32_t rmse_sum[NRT_BAND]
        short int norm_cm;
        short int cm_angle;
        unsigned char anomaly_conse;

cdef extern from "../../cxx/output.h":
    ctypedef struct Output_sccd_anomaly:
        int32_t t_break
        float coefs[NRT_BAND][SCCD_NUM_C]
        short int obs[NRT_BAND][DEFAULT_CONSE]
        short int obs_date_since1982[DEFAULT_CONSE]
        short int norm_cm[DEFAULT_CONSE]
        short int cm_angle[DEFAULT_CONSE]

cdef extern from "../../cxx/output.h":
    ctypedef struct output_nrtqueue:
        short int clry[NRT_BAND]
        short int clrx_since1982

cdef extern from "../../cxx/output.h":
    ctypedef struct Output_t_flex:
        int32_t t_start
        int32_t t_end
        int32_t t_break
        int32_t pos
        int32_t num_obs
        short int category
        short int change_prob
        float coefs[TOTAL_BAND_FLEX][FLEX_SCCD_NUM_C]
        float rmse[TOTAL_BAND_FLEX]
        float magnitude[TOTAL_BAND_FLEX]

cdef extern from "../../cxx/output.h":
    ctypedef struct Output_sccd_flex:
        int32_t t_start
        int t_break
        int num_obs
        float coefs[TOTAL_BAND_FLEX_NRT][FLEX_SCCD_NUM_C]
        float rmse[TOTAL_BAND_FLEX_NRT]
        float magnitude[TOTAL_BAND_FLEX_NRT]

cdef extern from "../../cxx/output.h":
    ctypedef struct Output_sccd_anomaly_flex:
        int32_t t_break
        float coefs[TOTAL_BAND_FLEX_NRT][FLEX_SCCD_NUM_C]
        short int obs[TOTAL_BAND_FLEX_NRT][DEFAULT_CONSE]
        short int obs_date_since1982[DEFAULT_CONSE]
        short int norm_cm[DEFAULT_CONSE]
        short int cm_angle[DEFAULT_CONSE]

cdef extern from "../../cxx/output.h":
    ctypedef struct output_nrtmodel_flex:
        short int t_start_since1982
        short int num_obs
        short int obs[TOTAL_BAND_FLEX_NRT][DEFAULT_CONSE]
        short int obs_date_since1982[DEFAULT_CONSE]
        float covariance[TOTAL_BAND_FLEX_NRT][64]
        float nrt_coefs[TOTAL_BAND_FLEX_NRT][FLEX_SCCD_NUM_C]
        float H[TOTAL_BAND_FLEX_NRT]
        uint32_t rmse_sum[TOTAL_BAND_FLEX_NRT]
        short int norm_cm;
        short int cm_angle;
        unsigned char anomaly_conse;

cdef extern from "../../cxx/output.h":
    ctypedef struct output_nrtqueue_flex:
        short int clry[TOTAL_BAND_FLEX_NRT]
        short int clrx_since1982

cdef extern from "../../cxx/cold.h":
    cdef int32_t cold(int64_t *buf_b, int64_t *buf_g, int64_t *buf_r, int64_t *buf_n, int64_t *buf_s1, int64_t *buf_s2,int64_t *buf_t, int64_t *fmask_buf, int64_t *valid_date_array, int32_t valid_num_scenes, int32_t pos, double tcg, int32_t conse, bool output_cm, int32_t starting_date, bool b_c2, Output_t *rec_cg,int32_t *num_fc, int32_t cm_output_interval, short int *cm_outputs, short int *cm_outputs_date, double gap_days, double lam);

cdef extern from "../../cxx/cold_flex.h":
    cdef int32_t cold_flex(int64_t *ts_stack, int64_t *fmask_buf, int64_t *valid_date_array, int nbands, int tmask_b1_index, int tmask_b1_index, int32_t valid_num_scenes, int32_t pos,double tcg, double max_tcg, int32_t conse, bool output_cm, int32_t starting_date, Output_t_flex *rec_cg, int32_t *num_fc, int32_t cm_output_interval, short int *cm_outputs, short int *cm_outputs_date, double gap_days, double lam);


cdef extern from "../../cxx/cold.h":
    cdef int32_t obcold_reconstruction_procedure(int64_t *buf_b, int64_t *buf_g, int64_t *buf_r,  int64_t *buf_n, int64_t *buf_s1,
    int64_t *buf_s2, int64_t *buf_t,  int64_t *fmask_buf, int64_t *valid_date_array, int32_t valid_num_scenes, int64_t *break_dates,
    int32_t break_date_len, int32_t pos, bool b_c2, int32_t conse, Output_t *rec_cg, int32_t *num_fc, double lam);



cdef extern from "../../cxx/s_ccd.h":
    cdef int32_t sccd(int64_t *buf_b, int64_t *buf_g, int64_t *buf_r, int64_t *buf_n, int64_t *buf_s1, int64_t *buf_s2, int64_t *buf_t, int64_t *fmask_buf, int64_t *valid_date_array, int32_t valid_num_scenes, double tcg, int32_t *num_fc, int32_t *nrt_mode, Output_sccd *rec_cg, output_nrtmodel *nrt_model, int32_t *num_nrt_queue, output_nrtqueue *nrt_queue, short int *min_rmse, int32_t conse, bool b_c2, bool output_anomaly, Output_sccd_anomaly *rec_cg_anomaly, int32_t *num_fc_anomaly, double anomaly_tcg, int anomaly_conse, double predictability_tcg,  bool b_output_state, double state_intervaldays, int32_t *n_state, int64_t *state_days, double *states_ensemble, bool fitting_coefs, double lam);


cdef extern from "../../cxx/s_ccd_flex.h":
    cdef int32_t sccd_flex(int64_t *ts_data, int64_t *fmask_buf, int64_t *valid_date_array, int nbands, int tmask_b1_index, int tmask_b2_index, int valid_num_scenes, double tcg, double max_t_cg, int32_t *num_fc, int32_t *nrt_mode, Output_sccd_flex *rec_cg, output_nrtmodel_flex *nrt_model, int32_t *num_obs_queue, output_nrtqueue_flex *obs_queue, short int *min_rmse, 
    int32_t conse, bool b_c2, bool output_anomaly, Output_sccd_anomaly_flex *rec_cg_anomaly,
    int32_t *num_fc_anomaly, double anomaly_tcg, int anomaly_conse, double predictability_tcg, bool b_output_state,double state_intervaldays, int32_t *n_state, int64_t *state_days, double *states_ensemble,bool fitting_coefs, double lam, int64_t n_coefs);


cdef Output_sccd t
cdef output_nrtqueue t2
cdef output_nrtmodel t3
cdef Output_sccd_anomaly t4


#cdef class SccdOutput:
#    cdef public int32_t position
#    cdef public np.ndarray rec_cg
#    cdef public int32_t nrt_mode
#    cdef public tuple nrt_model
#    cdef public np.ndarray nrt_queue
#    def __init__(self, position, rec_cg, nrt_mode, nrt_model, nrt_queue):
#        self.position = position
#        self.rec_cg = rec_cg
#        self.nrt_mode = nrt_mode
#        self.nrt_model = nrt_model
#        self.nrt_queue = nrt_queue
SccdOutput = namedtuple("SccdOutput", "position rec_cg min_rmse nrt_mode nrt_model nrt_queue")
SccdReccganomaly = namedtuple("SccdReccganomaly", "position rec_cg_anomaly")

def test_func():
    """
    This is a docstring
    """
    return None


cpdef _cold_detect(np.ndarray[np.int64_t, ndim=1, mode='c'] dates, np.ndarray[np.int64_t, ndim=1, mode='c'] ts_b,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_g, np.ndarray[np.int64_t, ndim=1, mode='c'] ts_r,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_n, np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s1,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s2, np.ndarray[np.int64_t, ndim=1, mode='c'] ts_t,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] qas, double t_cg = 15.0863, int32_t conse=6, int32_t pos=1, 
                   bint output_cm=False, int32_t starting_date=0, int32_t n_cm=0, int32_t cm_output_interval=0, bint b_c2=True,
                   double gap_days=365.25, double lam=20):
    """
    Helper function to do COLD algorithm.

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
        output_cm: bool, 'True' means outputting change magnitude and change magnitude dates, only for object-based COLD
        starting_date: the starting date of the whole dataset to enable reconstruct CM_date,
                    all pixels for a tile should have the same date, only for output_cm is True
        b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to the current low quality
        cm_output_interval: the temporal interval of outputting change magnitudes
        gap_days: define the day number of the gap year for i_dense
        Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands

        Returns
        ----------
        change records: the COLD outputs that characterizes each temporal segment
    """

    cdef int32_t valid_num_scenes = qas.shape[0]
    # allocate memory for rec_cg
    cdef int32_t num_fc = 0
    cdef Output_t t
    rec_cg = np.zeros(NUM_FC, dtype=reccg_dt)

    cdef int64_t [:] dates_view = dates
    cdef int64_t [:] ts_b_view = ts_b
    cdef int64_t [:] ts_g_view = ts_g
    cdef int64_t [:] ts_r_view = ts_r
    cdef int64_t [:] ts_n_view = ts_n
    cdef int64_t [:] ts_s1_view = ts_s1
    cdef int64_t [:] ts_s2_view = ts_s2
    cdef int64_t [:] ts_t_view = ts_t
    cdef int64_t [:] qas_view = qas
    cdef Output_t [:] rec_cg_view = rec_cg

    # cm_outputs and cm_outputs_date are for object-based cold
    if output_cm == True:
        if cm_output_interval == 0:
           cm_output_interval = 60
        if starting_date == 0:
           starting_date = dates[0]
        if n_cm == 0:
           n_cm = math.ceil((dates[valid_num_scenes-1] - starting_date + 1) / cm_output_interval) + 1
        cm_outputs = np.full(n_cm, -9999, dtype=np.short)
        cm_outputs_date = np.full(n_cm, -9999, dtype=np.short)
    # set the length to 1 to save memory, as they won't be assigned values
    else:
        cm_outputs = np.full(1, -9999, dtype=np.short)
        cm_outputs_date = np.full(1, -9999, dtype=np.short)
    cdef short [:] cm_outputs_view = cm_outputs  # memory view
    cdef short [:] cm_outputs_date_view = cm_outputs_date  # memory view

    result = cold(&ts_b_view[0], &ts_g_view[0], &ts_r_view[0], &ts_n_view[0], &ts_s1_view[0], &ts_s2_view[0], &ts_t_view[0],
                 &qas_view[0], &dates_view[0], valid_num_scenes, pos, t_cg, conse, output_cm,
                 starting_date, b_c2, &rec_cg_view[0], &num_fc, cm_output_interval, &cm_outputs_view[0], &cm_outputs_date_view[0],
                 gap_days, lam)
    if result != 0:
        raise RuntimeError("cold function fails for pos = {} ".format(pos))
    else:
        if num_fc <= 0:
            raise Exception("The COLD function has no change records outputted for pos = {} (possibly due to no enough clear observation)".format(pos))
        else:
            if output_cm == False:
                return rec_cg[:num_fc] # np.asarray uses also the buffer-protocol and is able to construct
                                                             # a dtype-object from cython's array
            else:  # for object-based COLD
                return [rec_cg[:num_fc], cm_outputs, cm_outputs_date]


cpdef _obcold_reconstruct(np.ndarray[np.int64_t, ndim=1, mode='c'] dates,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] ts_b,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] ts_g,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] ts_r,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] ts_n,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s1,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s2,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] ts_t,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] qas,
                          np.ndarray[np.int64_t, ndim=1, mode='c'] break_dates, int32_t pos=1,
                          int32_t conse=6, bint b_c2=True, double lam=20):
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
        conse: int, consecutive observation number (for calculating change magnitudes)
        b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to its current low quality
        Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands

        Returns
        ----------
        change records: the COLD outputs that characterizes each temporal segment
    """

    cdef int32_t valid_num_scenes = qas.shape[0]
    cdef int32_t break_date_len = break_dates.shape[0]
    # allocate memory for rec_cg
    cdef int32_t num_fc = 0
    cdef Output_t t
    rec_cg = np.zeros(NUM_FC, dtype=reccg_dt)

    cdef int64_t [:] dates_view = dates
    cdef int64_t [:] ts_b_view = ts_b
    cdef int64_t [:] ts_g_view = ts_g
    cdef int64_t [:] ts_r_view = ts_r
    cdef int64_t [:] ts_n_view = ts_n
    cdef int64_t [:] ts_s1_view = ts_s1
    cdef int64_t [:] ts_s2_view = ts_s2
    cdef int64_t [:] ts_t_view = ts_t
    cdef int64_t [:] qas_view = qas
    cdef int64_t [:] break_dates_view = break_dates
    cdef Output_t [:] rec_cg_view = rec_cg

    result = obcold_reconstruction_procedure(&ts_b_view[0], &ts_g_view[0], &ts_r_view[0], &ts_n_view[0], &ts_s1_view[0],
      &ts_s2_view[0], &ts_t_view[0], &qas_view[0], &dates_view[0], valid_num_scenes, &break_dates_view[0], break_date_len,
      pos, b_c2, conse, &rec_cg_view[0], &num_fc, lam)
    if result != 0:
        raise RuntimeError("cold function fails for pos = {} ".format(pos))
    else:
        if num_fc <= 0:
            raise Exception("The reconstruct function has no change records outputted for pos = {} (possibly due to no enough clear observation)".format(pos))
        else:
            return rec_cg[:num_fc] # np.asarray uses also the buffer-protocol and is able to construct a dtype-object from cython's array


cpdef _sccd_detect(np.ndarray[np.int64_t, ndim=1, mode='c'] dates,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_b,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_g,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_r,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_n,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s1,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s2,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_t,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] qas,
                   double t_cg = 15.0863, int32_t conse=6, int32_t pos=1, 
                   bint b_c2=True, bint output_anomaly=False, double anomaly_tcg=9.236, 
                   int anomaly_conse=3, double predictability_tcg=9.236, 
                   bint b_output_state=False, double state_intervaldays=1, bint fitting_coefs=False, double lam=20):
    """
    S-CCD processing. It is required to be done before near real time monitoring

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
       b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to its current low quality
       output_anomaly: bool, output anomaly break
       anomaly_tcg: the gate change magnitude threshold for defining anomaly
       anomaly_conse: the consecutive observation number for defining anomalies
       predictability_tcg: threshold for predicability test
       b_output_state: bool, if output intermediate state variables
       state_intervaldays: bool, what is the interval for outputting state
       fitting_coefs: if using fitting curve to output harmonic coefficients
       Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands
       Returns
       ----------
        namedtupe: SccdOutput
            change records: the S-CCD outputs that characterizes each temporal segment
            rec_cg:
            min_rmse
            int32_t nrt_mode,             /* O: 0 - void; 1 - monitor mode for standard; 2 - queue mode for standard;
                                        3 - monitor mode for snow; 4 - queue mode for snow */
            output_nrtmodel: nrt model if monitor mode, empty if queue mode
            output_nrtqueue: obs queue if queue mode, empty if monitor mode
    """
    if conse > DEFAULT_CONSE:
        raise RuntimeError("The inputted conse is longer than the maximum conse for S-CCD: {}".format(DEFAULT_CONSE))

    cdef int32_t valid_num_scenes = qas.shape[0]
    # allocate memory for rec_cg
    cdef int32_t num_fc = 0
    cdef int32_t num_nrt_queue = 0
    cdef int32_t num_fc_anomaly = 0
    rec_cg = np.zeros(NUM_FC_SCCD, dtype=sccd_dt)
    nrt_queue = np.zeros(NUM_NRT_QUEUE, dtype=nrtqueue_dt)
    nrt_model = np.zeros(1, dtype=nrtmodel_dt)
    rec_cg_anomaly = np.zeros(NUM_FC_SCCD, dtype=anomaly_dt)
    cdef int32_t nrt_mode = 0
    cdef int32_t n_state = 0
    
    # cdef int64_t ** state_ensemble = <int64_t **>malloc(max_n_states * NRT_BAND * 3)
    if output_anomaly == True and b_output_state == True:
        raise RuntimeError("output_anomaly and b_output_state cannot be set both True")

    if b_output_state == True:
        max_n_states = math.ceil((dates[-1] - dates[0]) / state_intervaldays)
        if max_n_states < 1:
            raise RuntimeError("Make sure that state_intervaldays must be larger than 0, and not higher than the total day number")
        # state_ensemble = np.full((max_n_states, NRT_BAND * 3), 0, dtype=np.double)
        state_ensemble = np.zeros(max_n_states * NRT_BAND * 3, dtype=np.double)
        state_days = np.zeros(max_n_states, dtype=np.int64)
    else:
        state_ensemble = np.zeros(1, dtype=np.double)
        state_days = np.zeros(1, dtype=np.int64)

    if dates[-1] - dates[0] < 365.25:
        raise RuntimeError("The input data length is smaller than 1 year for pos = {}".format(pos))

    # initiate minimum rmse
    min_rmse = np.full(NRT_BAND, 0, dtype=np.short)

    # memory view
    cdef int64_t [:] dates_view = dates
    cdef int64_t [:] ts_b_view = ts_b
    cdef int64_t [:] ts_g_view = ts_g
    cdef int64_t [:] ts_r_view = ts_r
    cdef int64_t [:] ts_n_view = ts_n
    cdef int64_t [:] ts_s1_view = ts_s1
    cdef int64_t [:] ts_s2_view = ts_s2
    cdef int64_t [:] ts_t_view = ts_t
    cdef int64_t [:] qas_view = qas
    cdef short [:] min_rmse_view = min_rmse

    cdef Output_sccd [:] rec_cg_view = rec_cg
    cdef output_nrtqueue [:] nrt_queue_view = nrt_queue
    cdef output_nrtmodel [:] nrt_model_view = nrt_model
    cdef Output_sccd_anomaly [:] rec_cg_anomaly_view = rec_cg_anomaly

    cdef int64_t [:] states_days_view = state_days
    cdef double [:] states_ensemble_view = state_ensemble


    result = sccd(&ts_b_view[0], &ts_g_view[0], &ts_r_view[0], &ts_n_view[0], &ts_s1_view[0], &ts_s2_view[0], &ts_t_view[0], &qas_view[0], &dates_view[0], valid_num_scenes, t_cg, &num_fc, &nrt_mode, &rec_cg_view[0], &nrt_model_view[0], &num_nrt_queue, &nrt_queue_view[0], &min_rmse_view[0], conse, b_c2, output_anomaly, &rec_cg_anomaly_view[0], &num_fc_anomaly, anomaly_tcg, anomaly_conse, predictability_tcg, b_output_state, state_intervaldays, &n_state, &states_days_view[0], &states_ensemble_view[0], fitting_coefs, lam)
    
    if result != 0:
        raise RuntimeError("S-CCD function fails for pos = {} ".format(pos))
    else:
        if num_fc > 0:
            output_rec_cg = rec_cg[:num_fc]
        else:
            output_rec_cg = np.array([])

        if output_anomaly == False:
            if b_output_state == False:
                if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
                    return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], np.array([]))
                if nrt_mode % 10 == 2 or nrt_mode == 4:  # queue mode
                    return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue[:num_nrt_queue])
                elif nrt_mode % 10 == 5:  # queue mode
                    return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], nrt_queue[:num_nrt_queue])
                elif nrt_mode == 0:  # void mode
                    return SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]),
                                    np.array([]))
                else:
                    raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))
            else:
                colnames = ['dates', 'blue_trend', 'green_trend', 'red_trend', 'nir_trend', 'swir1_trend', 'swir2_trend', 
                            'blue_annual', 'green_annual', 'red_annual', 'nir_annual', 'swir1_annual',  'swir2_annual', 'blue_semiannual', 
                            'green_semiannual', 'red_semiannual', 'nir_semiannual', 'swir1_semiannual', 'swir2_semiannual']
                state_ensemble = state_ensemble.reshape(-1, NRT_BAND * 3)
                state_all = pd.DataFrame(np.column_stack((state_days[0:n_state], state_ensemble[0:n_state,:])), columns=colnames)
                if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
                    return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], np.array([])), state_all]
                if nrt_mode % 10 == 2 or nrt_mode == 4:  # queue mode
                    return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue[:num_nrt_queue]), state_all]
                elif nrt_mode % 10 == 5:  # queue mode
                    return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], nrt_queue[:num_nrt_queue]), state_all]
                elif nrt_mode == 0:  # void mode
                    return [SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]),
                                    np.array([])), state_all]
                else:
                    raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))
        else:
            if num_fc_anomaly > 0:
                output_rec_cg_anomaly = rec_cg_anomaly[:num_fc_anomaly]
            else:
                output_rec_cg_anomaly = np.array([])

            if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
                return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode,
                                   nrt_model[0], np.array([])),
                                   SccdReccganomaly(pos, output_rec_cg_anomaly)]
            elif nrt_mode % 10 == 2 or nrt_mode == 4:
                return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue[:num_nrt_queue]),
                                    SccdReccganomaly(pos, output_rec_cg_anomaly)]
            elif nrt_mode % 10 == 5:  # queue mode
                return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], nrt_queue[:num_nrt_queue]),
                                   SccdReccganomaly(pos, output_rec_cg_anomaly)]
            elif nrt_mode == 0:  # void mode
                return [SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]),
                                   np.array([])), output_rec_cg_anomaly]
            else:
                raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))


cpdef _sccd_update(sccd_pack,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] dates,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_b,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_g,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_r,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_n,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s1,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_s2,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] ts_t,
                   np.ndarray[np.int64_t, ndim=1, mode='c'] qas,
                   double t_cg = 15.0863, int32_t conse=6, int32_t pos=1, bint b_c2=True,
                   double anomaly_tcg=9.236, double predictability_tcg=15.086, double lam=20):
    """
    SCCD online update for new observations

       Parameters
       ----------
       sccd_pack: a namedtuple of SccdOutput
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
       b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to its current low quality
       anomaly_tcg: the gate change magnitude threshold for defining anomaly
       Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands
       Returns
       ----------
       namedtupe: SccdOutput
            rec_cg: the S-CCD outputs that characterizes each temporal segment
            min_rmse
            int32_t nrt_mode,             /* O: 0 - void; 1 - monitor mode for standard; 2 - queue mode for standard;
                                            3 - monitor mode for snow; 4 - queue mode for snow  */
            nrt_model: nrt model if monitor mode, empty if queue mode
            nrt_queue: obs queue if queue mode, empty if monitor mode
    """

    cdef int32_t valid_num_scenes = qas.shape[0]
    # allocate memory for rec_cg
    # cdef int32_t num_fc = 0
    # cdef int32_t num_nrt_queue = 0
    cdef int32_t nrt_mode = sccd_pack.nrt_mode

    if nrt_mode != 0 and nrt_mode % 10 != 1 and nrt_mode % 10 != 2 and nrt_mode != 3 and nrt_mode != 4 and nrt_mode != 5:
        raise RuntimeError("Invalid nrt_node input {} for pos = {} ".format(nrt_mode, pos))

    cdef int32_t num_fc = len(sccd_pack.rec_cg)
    cdef int32_t num_nrt_queue = len(sccd_pack.nrt_queue)
    cdef int32_t num_fc_anomaly = 0
    cdef Output_sccd_anomaly* rec_cg_anomaly = <Output_sccd_anomaly*> PyMem_Malloc(sizeof(t4))
    state_ensemble = np.zeros(1, dtype=np.double)
    state_days = np.zeros(1, dtype=np.int64)
    cdef int32_t n_state = 0

    # grab inputs from the input
    rec_cg_new = np.empty(NUM_FC_SCCD, dtype=sccd_dt)
    if num_fc > 0:
        rec_cg_new[0:num_fc] = sccd_pack.rec_cg[0:num_fc]

    nrt_queue_new = np.empty(NUM_NRT_QUEUE, dtype=nrtqueue_dt)
    if num_nrt_queue > 0:
        nrt_queue_new[0:num_nrt_queue] = sccd_pack.nrt_queue[0:num_nrt_queue]

    # TO CHECK
    if nrt_mode % 10 == 1 or nrt_mode == 3 or nrt_mode % 10 == 5:
        nrt_model_new = np.zeros(1, dtype=nrtmodel_dt)
        nrt_model_new[0] = sccd_pack.nrt_model
    else:
        nrt_model_new = np.empty(1, dtype=nrtmodel_dt)

    min_rmse = sccd_pack.min_rmse

    # memory view
    cdef Output_sccd [:] rec_cg_view = rec_cg_new
    cdef output_nrtqueue [:] nrt_queue_view = nrt_queue_new
    cdef output_nrtmodel [:] nrt_model_view = nrt_model_new
    cdef short [:] min_rmse_view = min_rmse
    cdef int64_t [:] dates_view = dates
    cdef int64_t [:] ts_b_view = ts_b
    cdef int64_t [:] ts_g_view = ts_g
    cdef int64_t [:] ts_r_view = ts_r
    cdef int64_t [:] ts_n_view = ts_n
    cdef int64_t [:] ts_s1_view = ts_s1
    cdef int64_t [:] ts_s2_view = ts_s2
    cdef int64_t [:] ts_t_view = ts_t
    cdef int64_t [:] qas_view = qas

    
    cdef int64_t [:] states_days_view = state_days
    cdef double [:] states_ensemble_view = state_ensemble

    result = sccd(&ts_b_view[0], &ts_g_view[0], &ts_r_view[0], &ts_n_view[0], &ts_s1_view[0], &ts_s2_view[0],&ts_t_view[0], &qas_view[0], &dates_view[0], valid_num_scenes, t_cg, &num_fc, &nrt_mode, &rec_cg_view[0],&nrt_model_view[0], &num_nrt_queue, &nrt_queue_view[0], &min_rmse_view[0], conse, b_c2, False,rec_cg_anomaly, &num_fc_anomaly, anomaly_tcg, 3, predictability_tcg, False, 1, &n_state, &states_days_view[0], &states_ensemble_view[0], False, lam)

    PyMem_Free(rec_cg_anomaly)
    if result != 0:
        raise RuntimeError("sccd_update function fails for pos = {} ".format(pos))
    else:
        # sccd_pack_copy = None
        if num_fc > 0:
            output_rec_cg = rec_cg_new[0:num_fc]
        else:
            output_rec_cg = np.array([])

        if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
            return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode,
                              nrt_model_new[0], np.array([]))
        elif nrt_mode % 10 == 2 or nrt_mode == 4: # queue mode:
            return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue_new[0:num_nrt_queue])
        elif nrt_mode % 10 == 5:  # queue or m2q mode
            return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model_new[0], nrt_queue_new[0:num_nrt_queue])
        elif nrt_mode == 0:  # void mode
            return SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]), np.array([]))
        else:
            raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))


cpdef _cold_detect_flex(np.ndarray[np.int64_t, ndim=1, mode='c'] dates, np.ndarray[np.int64_t, ndim=1, mode='c'] ts_stack,
                        np.ndarray[np.int64_t, ndim=1, mode='c'] qas, int32_t valid_num_scenes, int32_t nbands,
                        double t_cg, double max_t_cg, int32_t conse=6, int32_t pos=1, bint output_cm=False, 
                        int32_t starting_date=0, int32_t n_cm=0, int32_t cm_output_interval=0, 
                        double gap_days=365.25, int32_t tmask_b1_index=1, int32_t tmask_b2_index=1, double lam=20):
    """
    Helper function to do COLD algorithm.

        Parameters
        ----------
        dates: 1d array of shape(observation numbers), list of ordinal dates
        ts_stack: 2d array of shape(observation numbers), time series of multispectral data.
        qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
        valid_num_scenes: the number of valid observations
        nbands: the number of inputted bands
        t_cg: threshold for change magnitude
        max_t_cg: threshold for identifying outlier
        pos: position id of the pixel
        conse: consecutive observation number
        output_cm: bool, 'True' means outputting change magnitude and change magnitude dates, only for object-based COLD
        starting_date: the starting date of the whole dataset to enable reconstruct CM_date,
                    all pixels for a tile should have the same date, only for output_cm is True
        cm_output_interval: the temporal interval of outputting change magnitudes
        gap_days: define the day number of the gap year for i_dense
        tmask_b1_index: the first band id for tmask
        tmask_b2_index: the second band id for tmask

        Returns
        ----------
        change records: the COLD outputs that characterizes each temporal segment
    """

    # allocate memory for rec_cg
    cdef int32_t num_fc = 0
    cdef Output_t_flex t
    rec_cg = np.zeros(NUM_FC, dtype=cold_dt_flex)

    cdef int64_t [:] dates_view = dates
    cdef int64_t [:] ts_stack_view = ts_stack
    cdef int64_t [:] qas_view = qas
    cdef Output_t_flex [:] rec_cg_view = rec_cg

    # cm_outputs and cm_outputs_date are for object-based cold
    if output_cm == True:
        if cm_output_interval == 0:
           cm_output_interval = 60
        if starting_date == 0:
           starting_date = dates[0]
        if n_cm == 0:
           n_cm = math.ceil((dates[valid_num_scenes-1] - starting_date + 1) / cm_output_interval) + 1
        cm_outputs = np.full(n_cm, -9999, dtype=np.short)
        cm_outputs_date = np.full(n_cm, -9999, dtype=np.short)
    # set the length to 1 to save memory, as they won't be assigned values
    else:
        cm_outputs = np.full(1, -9999, dtype=np.short)
        cm_outputs_date = np.full(1, -9999, dtype=np.short)
    cdef short [:] cm_outputs_view = cm_outputs  # memory view
    cdef short [:] cm_outputs_date_view = cm_outputs_date  # memory view
    result = cold_flex(&ts_stack_view[0], &qas_view[0], &dates_view[0], nbands, tmask_b1_index, tmask_b2_index, valid_num_scenes, pos, t_cg, max_t_cg, conse, output_cm, starting_date, 
                        &rec_cg_view[0], &num_fc, cm_output_interval, &cm_outputs_view[0], 
                        &cm_outputs_date_view[0], gap_days, lam)
    
    if result != 0:
        raise RuntimeError("cold function fails for pos = {} ".format(pos))
    else:
        if num_fc <= 0:
            raise Exception("The COLD function has no change records outputted for pos = {} (possibly due to no enough clear observation)".format(pos))
        else:
            rec_cg_new = _update_cold_reccg(rec_cg[:num_fc], nbands)
            if output_cm == False:
                return rec_cg_new # np.asarray uses also the buffer-protocol and is able to construct
                                                             # a dtype-object from cython's array
            else:  # for object-based COLD
                return [rec_cg_new, cm_outputs, cm_outputs_date]



cpdef _sccd_detect_flex(np.ndarray[np.int64_t, ndim=1, mode='c'] dates, np.ndarray[np.int64_t,      ndim=1, mode='c'] ts_stack,np.ndarray[np.int64_t, ndim=1, mode='c'] qas, int32_t valid_num_scenes, int32_t nbands, double t_cg, double max_t_cg, int32_t conse=6, int32_t pos=1, bint b_c2=True, bint output_anomaly=False, double anomaly_tcg=9.236, int anomaly_conse=3,double predictability_tcg=9.236, bint b_output_state=False, double state_intervaldays=1, int32_t tmask_b1_index=1, int32_t tmask_b2_index=1, bint fitting_coefs=False, double lam=20, bool trimodal=False):
    """
    Helper function to do COLD algorithm.

        Parameters
        ----------
        dates: 1d array of shape(observation numbers), list of ordinal dates
        ts_stack: 2d array of shape(observation numbers), time series of multispectral data.
        qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
        valid_num_scenes: the number of valid observations
        nbands: the number of inputted bands
        t_cg: threshold of change magnitude, default is chi2.ppf(0.99,5)
        max_t_cg: threshold for identifying outliers
        conse: consecutive observation number
        pos: position id of the pixel
        b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to its current low quality
        output_anomaly: indicate whether to output anomalies
        anomaly_tcg: the gate change magnitude threshold for defining anomalies
        anomaly_conse: the consecutive observation number for defining anomalies
        starting_date: the starting date of the whole dataset to enable reconstruct CM_date,
                    all pixels for a tile should have the same date, only for output_cm is True
        cm_output_interval: the temporal interval of outputting change magnitudes
        gap_days: define the day number of the gap year for i_dense
        tmask_b1_index: the first band id for tmask
        tmask_b2_index: the second band id for tmask
        lam: the lambda for lasso regression

        Returns
        ----------
        change records: the S-CCD outputs that characterizes each temporal segment
    """
    if conse > DEFAULT_CONSE:
        raise RuntimeError("The inputted conse is longer than the maximum conse for S-CCD: {}".format(DEFAULT_CONSE))

    # allocate memory for rec_cg
    cdef Output_t t
    cdef int64_t n_coefs

    # allocate memory for rec_cg
    cdef int32_t num_fc = 0
    cdef int32_t num_nrt_queue = 0
    cdef int32_t num_fc_anomaly = 0
    cdef int32_t nrt_mode = 0
    cdef int32_t n_state = 0
    rec_cg = np.zeros(NUM_FC_SCCD, dtype=sccd_dt_flex)
    nrt_queue = np.zeros(NUM_NRT_QUEUE, dtype=nrtqueue_dt_flex)
    nrt_model = np.zeros(1, dtype=nrtmodel_dt_flex)
    rec_cg_anomaly = np.zeros(NUM_FC_SCCD, dtype=anomaly_dt_flex)
    if trimodal == True:
        n_coefs = 8
    else:
        n_coefs = 6


    if b_output_state == True:
        max_n_states = math.ceil((dates[-1] - dates[0]) / state_intervaldays)
        if max_n_states < 1:
            raise RuntimeError("Make sure that state_intervaldays must be larger than 0, and not higher than the total day number")
        state_ensemble = np.zeros(int(max_n_states * nbands * n_coefs / 2), dtype=np.double)
        state_days = np.zeros(max_n_states, dtype=np.int64)
    else:
        state_ensemble = np.zeros(1, dtype=np.double)
        state_days = np.zeros(1, dtype=np.int64)

    if dates[-1] - dates[0] < 365.25:
        raise RuntimeError("The input data length is smaller than 1 year for pos = {}".format(pos))

    # initiate minimum rmse
    min_rmse = np.full(nbands, 0, dtype=np.short)

    # memory view
    cdef int64_t [:] dates_view = dates
    cdef int64_t [:] ts_stack_view = ts_stack
    cdef int64_t [:] qas_view = qas
    cdef short [:] min_rmse_view = min_rmse

    cdef Output_sccd_flex [:] rec_cg_view = rec_cg
    cdef output_nrtqueue_flex [:] nrt_queue_view = nrt_queue
    cdef output_nrtmodel_flex [:] nrt_model_view = nrt_model
    cdef Output_sccd_anomaly_flex [:] rec_cg_anomaly_view = rec_cg_anomaly

    cdef int64_t [:] states_days_view = state_days
    cdef double [:] states_ensemble_view = state_ensemble

    result = sccd_flex(&ts_stack_view[0], &qas_view[0], &dates_view[0], nbands, tmask_b1_index, tmask_b2_index, 
                        valid_num_scenes, t_cg, max_t_cg, &num_fc, &nrt_mode, &rec_cg_view[0],
                        &nrt_model_view[0], &num_nrt_queue, &nrt_queue_view[0], &min_rmse_view[0], 
                        conse, b_c2, output_anomaly, &rec_cg_anomaly_view[0], &num_fc_anomaly, 
                        anomaly_tcg, anomaly_conse, predictability_tcg, b_output_state, state_intervaldays, &n_state, &states_days_view[0], &states_ensemble_view[0], fitting_coefs, lam, n_coefs)

    if result != 0:
        raise RuntimeError("S-CCD function fails for pos = {} ".format(pos))
    else:
        if num_fc > 0:
            output_rec_cg = _update_sccd_reccg(rec_cg[:num_fc], nbands, n_coefs)
            # output_rec_cg = rec_cg[:num_fc]
        else:
            output_rec_cg = np.array([])

        nrt_model = _update_nrt_model(nrt_model, nbands, n_coefs)
        nrt_queue = _update_nrtqueue(nrt_queue, nbands)
        if output_anomaly == False:
            if b_output_state == False:
                if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
                    return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], np.array([]))
                if nrt_mode % 10 == 2 or nrt_mode == 4:  # queue mode
                    return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue[:num_nrt_queue])
                elif nrt_mode % 10 == 5:  # queue mode
                    return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], nrt_queue[:num_nrt_queue])
                elif nrt_mode == 0:  # void mode
                    return SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]),
                                    np.array([]))
                else:
                    raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))
            else:
                if trimodal == False:
                    colnames = ["dates"] + [f"b{i}_trend" for i in range(nbands)] + [f"b{i}_annual" for i in range(nbands)] + [f"b{i}_semiannual" for i in range(nbands)] 
                    state_ensemble = state_ensemble.reshape(-1, nbands * 3)
                    state_all = pd.DataFrame(np.column_stack((state_days[0:n_state], state_ensemble[0:n_state,:])), columns=colnames)
                else:
                    colnames = ["dates"] + [f"b{i}_trend" for i in range(nbands)] + [f"b{i}_annual" for i in range(nbands)] + [f"b{i}_semiannual" for i in range(nbands)] + [f"b{i}_trimodal" for i in range(nbands)] 
                    state_ensemble = state_ensemble.reshape(-1, nbands * 4)
                    state_all = pd.DataFrame(np.column_stack((state_days[0:n_state], state_ensemble[0:n_state,:])), columns=colnames)
                if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
                    return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], np.array([])), state_all]
                if nrt_mode % 10 == 2 or nrt_mode == 4:  # queue mode
                    return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue[:num_nrt_queue]), state_all]
                elif nrt_mode % 10 == 5:  # queue mode
                    return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], nrt_queue[:num_nrt_queue]), state_all]
                elif nrt_mode == 0:  # void mode
                    return [SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]),
                                    np.array([])), state_days[0:n_state], state_all]
                else:
                    raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))
        else:
            if num_fc_anomaly > 0:
                output_rec_cg_anomaly = rec_cg_anomaly[:num_fc_anomaly]
            else:
                output_rec_cg_anomaly = np.array([])
            output_rec_cg_anomaly = update_anomaly(output_rec_cg_anomaly, nbands, n_coefs)
            if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
                return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode,
                                   nrt_model[0], np.array([])),
                                   SccdReccganomaly(pos, output_rec_cg_anomaly)]
            elif nrt_mode % 10 == 2 or nrt_mode == 4:
                return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue[:num_nrt_queue]),
                                    SccdReccganomaly(pos, output_rec_cg_anomaly)]
            elif nrt_mode % 10 == 5:  # queue mode
                return [SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model[0], nrt_queue[:num_nrt_queue]),
                                   SccdReccganomaly(pos, output_rec_cg_anomaly)]
            elif nrt_mode == 0:  # void mode
                return [SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]),
                                   np.array([])), output_rec_cg_anomaly]
            else:
                raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))


cpdef _sccd_update_flex(sccd_pack,
                        np.ndarray[np.int64_t, ndim=1, mode='c'] dates,
                        np.ndarray[np.int64_t, ndim=1, mode='c'] ts_stack,
                        np.ndarray[np.int64_t, ndim=1, mode='c'] qas, int32_t valid_num_scenes, 
                        int32_t nbands, double t_cg, double max_t_cg, 
                        int32_t conse=6, int32_t pos=1, bint b_c2=True,
                        double anomaly_tcg=9.236, int anomaly_conse=3, 
                        double predictability_tcg=15.086, 
                        int32_t tmask_b1_index=1, int32_t tmask_b2_index=1, double lam=20, bool trimodal=False):
    """
    SCCD online update for new observations

       Parameters
       ----------
       sccd_pack: a namedtuple of SccdOutput
       dates: 1d array of shape(observation numbers), list of ordinal dates
       ts_stack: 2d array of shape(observation numbers), time series of multispectral data.
       qas: 1d array, the QA cfmask bands. '0' - clear; '1' - water; '2' - shadow; '3' - snow; '4' - cloud
       valid_num_scenes: the number of valid observations
       nbands: the number of inputted bands
       t_cg: threshold of change magnitude, default is chi2.ppf(0.99,5)
       pos: position id of the pixel
       conse: consecutive observation number
       b_c2: bool, a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band for valid pixel test due to its current low quality
       output_anomaly: indicate whether to output anomalies
       anomaly_tcg: the gate change magnitude threshold for defining anomalies
       anomaly_conse: the consecutive observations for defining anomalies
       tmask_b1_index: the first band id for tmask
       tmask_b2_index: the second band id for tmask
       Note that passing 2-d array to c as 2-d pointer does not work, so have to pass separate bands
       Returns
       ----------
       namedtupe: SccdOutput
            rec_cg: the S-CCD outputs that characterizes each temporal segment
            min_rmse
            int32_t nrt_mode,             /* O: 0 - void; 1 - monitor mode for standard; 2 - queue mode for standard;
                                            3 - monitor mode for snow; 4 - queue mode for snow  */
            nrt_model: nrt model if monitor mode, empty if queue mode
            nrt_queue: obs queue if queue mode, empty if monitor mode
    """

    # allocate memory for rec_cg
    # cdef int32_t num_fc = 0
    # cdef int32_t num_nrt_queue = 0
    cdef int32_t nrt_mode = sccd_pack.nrt_mode

    if nrt_mode != 0 and nrt_mode % 10 != 1 and nrt_mode % 10 != 2 and nrt_mode != 3 and nrt_mode != 4 and nrt_mode != 5:
        raise RuntimeError("Invalid nrt_node input {} for pos = {} ".format(nrt_mode, pos))

    cdef int64_t n_coefs
    if trimodal == True:
        n_coefs = 8
    else:
        n_coefs = 6

    # cdef int64_t n_coefs_old
    # cdef int64_t nbands_old

    cdef int32_t num_fc = len(sccd_pack.rec_cg)
    cdef int32_t num_nrt_queue = len(sccd_pack.nrt_queue)
    cdef int32_t num_fc_anomaly = 0
    cdef Output_sccd_anomaly_flex* rec_cg_anomaly = <Output_sccd_anomaly_flex*> PyMem_Malloc(sizeof(t4))
    state_ensemble = np.zeros(1, dtype=np.double)
    state_days = np.zeros(1, dtype=np.int64)
    cdef int32_t n_state = 0

    # grab inputs from the input
    rec_cg_new = np.empty(NUM_FC_SCCD, dtype=sccd_dt_flex)
    if num_fc > 0:
        rec_cg_new[0:num_fc] = sccd_pack.rec_cg[0:num_fc]

    nrt_queue_new = np.empty(NUM_NRT_QUEUE, dtype=nrtqueue_dt_flex)
    if num_nrt_queue > 0:
        nrt_queue_new[0:num_nrt_queue] = sccd_pack.nrt_queue[0:num_nrt_queue]

    # TO CHECK
    if nrt_mode % 10 == 1 or nrt_mode == 3 or nrt_mode % 10 == 5:
        nrt_model_new = np.zeros(1, dtype=nrtmodel_dt_flex)
        nrt_model_new[0] = sccd_pack.nrt_model
        if nbands != np.shape(sccd_pack.nrt_model['nrt_coefs'])[0]:
            raise RuntimeError("Unmatched bands for original and new SccdOutput for pos = {}: the original band number is {} ".format(nrt_mode, np.shape(sccd_pack.nrt_model['nrt_coefs'])[0]))
        if n_coefs != np.shape(sccd_pack.nrt_model['nrt_coefs'])[1]:
            raise RuntimeError("Unmatched harmonic coefficient number for original and new SccdOutput for pos = {}: the harmonic coefficient number is {} ".format(nrt_mode, np.shape(sccd_pack.nrt_model['nrt_coefs'])[1]))
    else:
        nrt_model_new = np.empty(1, dtype=nrtmodel_dt_flex)

    min_rmse = sccd_pack.min_rmse

    # memory view
    cdef Output_sccd_flex [:] rec_cg_view = rec_cg_new
    cdef output_nrtqueue_flex [:] nrt_queue_view = nrt_queue_new
    cdef output_nrtmodel_flex [:] nrt_model_view = nrt_model_new
    cdef short [:] min_rmse_view = min_rmse
    cdef int64_t [:] dates_view = dates
    cdef int64_t [:] ts_stack_view = ts_stack
    cdef int64_t [:] qas_view = qas

    cdef int64_t [:] states_days_view = state_days
    cdef double [:] states_ensemble_view = state_ensemble

    result = sccd_flex(&ts_stack_view[0], &qas_view[0], &dates_view[0], nbands, tmask_b1_index,     tmask_b2_index, valid_num_scenes, t_cg, max_t_cg, &num_fc, &nrt_mode, &rec_cg_view[0],&nrt_model_view[0], &num_nrt_queue, &nrt_queue_view[0], &min_rmse_view[0], conse, b_c2, False, rec_cg_anomaly, &num_fc_anomaly, anomaly_tcg, anomaly_conse, predictability_tcg, False, 1, &n_state, &states_days_view[0],&states_ensemble_view[0], False, lam, n_coefs)

    PyMem_Free(rec_cg_anomaly)
    if result != 0:
        raise RuntimeError("sccd_update function fails for pos = {} ".format(pos))
    else:
        # sccd_pack_copy = None
        if num_fc > 0:
            output_rec_cg = _update_sccd_reccg(rec_cg_new[:num_fc], nbands, n_coefs)
            # output_rec_cg = rec_cg[:num_fc]
        else:
            output_rec_cg = np.array([])
        
        nrt_model_new = _update_nrt_model(nrt_model_new, nbands, n_coefs)
        nrt_queue_new = _update_nrtqueue(nrt_queue_new, nbands)

        if nrt_mode % 10 == 1 or nrt_mode == 3:  # monitor mode
            return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode,
                              nrt_model_new[0], np.array([]))
        elif nrt_mode % 10 == 2 or nrt_mode == 4: # queue mode:
            return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, np.array([]), nrt_queue_new[0:num_nrt_queue])
        elif nrt_mode % 10 == 5:  # queue or m2q mode
            return SccdOutput(pos, output_rec_cg, min_rmse, nrt_mode, nrt_model_new[0], nrt_queue_new[0:num_nrt_queue])
        elif nrt_mode == 0:  # void mode
            return SccdOutput(pos, np.array([]), min_rmse, nrt_mode, np.array([]), np.array([]))
        else:
            raise RuntimeError("No correct nrt_mode (mode={}) returned for pos = {} ".format(nrt_mode, pos))