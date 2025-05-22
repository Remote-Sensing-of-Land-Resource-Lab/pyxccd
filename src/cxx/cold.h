#ifndef CCD_H
#define CCD_H
#include <stdint.h>
#include <stdbool.h>
#include "output.h"
// #include <xgboost/c_api.h>

int cold(
    int64_t *buf_b,             /* I:  Landsat blue spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_g,             /* I:  Landsat green spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_r,             /* I:  Landsat red spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_n,             /* I:  Landsat NIR spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s1,            /* I:  Landsat swir1 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s2,            /* I:  Landsat swir2 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_t,             /* I:  Landsat thermal spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,         /* I:  the time series of cfmask values. 0 - clear; 1 - water; 2 - shadow; 3 - snow; 4 - cloud  */
    int64_t *valid_date_array,  /* I:  valid date as matlab serial date form (counting from Jan 0, 0000). Note ordinal date in python is from (Jan 1th, 0001) */
    int valid_num_scenes,       /* I: number of valid scenes  */
    int pos,                    /* I: the position id of pixel */
    double tcg,                 /* I: threshold of change threshold  */
    int conse,                  /* I: consecutive observation number   */
    bool b_outputCM,            /* I: indicate if outputting change magnitudes for object-based cold, for cold only, it is the false */
    int starting_date,          /* I: (optional) the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile should have the same date, only for b_outputCM is True */
    bool b_c2,                  /* I: a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band due to the current low quality  */
    Output_t *rec_cg,           /* O: outputted structure for CCDC results    */
    int *num_fc,                /* O: number of fitting curves                   */
    int CM_OUTPUT_INTERVAL,     /* I: the interval of days of outputting change maganitude                   */
    short int *CM_outputs,      /* I/O: (optional) maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    short int *CM_outputs_date, /* I/O: (optional) dates for maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    double gap_days,            /* I: the day number of gap to define i_dense; it is useful for the cases that gap is in the middle of time series      */
    double lambda);

int stand_procedure(
    int valid_num_scenes,      /* I:  number of valid scenes  */
    int64_t *valid_date_array, /* I: valid date time series  */
    int64_t *buf_b,            /* I:  Landsat blue spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_g,            /* I:  Landsat green spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_r,            /* I:  Landsat red spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_n,            /* I:  Landsat NIR spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s1,           /* I:  Landsat swir1 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s2,           /* I:  Landsat swir2 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_t,            /* I:  Landsat thermal spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,        /* I:  mask-based time series  */
    int *id_range,
    double tcg,        /* I: threshold of change threshold  */
    int conse,         /* I: consecutive observation number   */
    bool b_outputCM,   /* I: indicate if outputting change magnitudes object-based cold*/
    int starting_date, /* I: the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile should have the same date, only for b_outputCM is True */
    Output_t *rec_cg,  /* O: outputted structure for CCDC results     */
    int *num_fc,       /* O: number of fitting curves                       */
    int CM_OUTPUT_INTERVAL,
    short int *CM_outputs,      /* I/O: maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    short int *CM_outputs_date, /* I/O: dates for maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    bool b_c2,
    double gap_days,
    double lambda);

int inefficientobs_procedure(
    int valid_num_scenes,      /* I:  number of scenes  */
    int64_t *valid_date_array, /* I: valid date time series  */
    int64_t *buf_b,            /* I:  Landsat blue spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_g,            /* I:  Landsat green spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_r,            /* I:  Landsat red spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_n,            /* I:  Landsat NIR spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s1,           /* I:  Landsat swir1 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s2,           /* I:  Landsat swir2 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_t,            /* I:  Landsat thermal spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,        /* I:  mask-based time series  */
    int *id_range,
    float sn_pct,
    Output_t *rec_cg,
    int *num_fc,
    bool b_c2,
    double lambda);

int obcold_reconstruction_procedure(
    int64_t *buf_b,            /* I:  Landsat blue spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_g,            /* I:  Landsat green spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_r,            /* I:  Landsat red spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_n,            /* I:  Landsat NIR spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s1,           /* I:  Landsat swir1 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s2,           /* I:  Landsat swir2 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_t,            /* I:  Landsat thermal spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,        /* I:  the time series of cfmask values. 0 - clear; 1 - water; 2 - shadow; 3 - snow; 4 - cloud  */
    int64_t *valid_date_array, /* I:  valid date as matlab serial date form (counting from Jan 0, 0000). Note ordinal date in python is from (Jan 1th, 0001) */
    int valid_num_scenes,      /* I: number of valid scenes  */
    int64_t *break_dates,      /*an array of break dates with a fixed length of num_year, '0' means no breaks */
    int break_date_len,        /*I: the number of focused years */
    int pos,                   /*I: the position of the pixel */
    bool b_c2,                 /* I: a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band due to the current low quality  */
    int conse,
    Output_t *rec_cg, /* O: Initialize NUM of Functional Curves    */
    int *num_fc,
    double lambda);

double angle_decaying(
    double input,
    double lowbound,
    double highbound);

// int stand_procedure_fixeddays(
//     int valid_num_scenes,      /* I:  number of valid scenes  */
//     int64_t *valid_date_array, /* I: valid date time series  */
//     int64_t *buf_b,            /* I:  Landsat blue spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
//     int64_t *buf_g,            /* I:  Landsat green spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
//     int64_t *buf_r,            /* I:  Landsat red spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
//     int64_t *buf_n,            /* I:  Landsat NIR spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
//     int64_t *buf_s1,           /* I:  Landsat swir1 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
//     int64_t *buf_s2,           /* I:  Landsat swir2 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
//     int64_t *buf_t,            /* I:  Landsat thermal spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
//     int64_t *fmask_buf,        /* I:  mask-based time series  */
//     int *id_range,
//     double tcg,        /* I: threshold of change threshold  */
//     int conse,         /* I: consecutive observation number   */
//     bool b_outputCM,   /* I: indicate if cold is running as the first step of object-based cold*/
//     int starting_date, /* I: the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile should have the same date, only for b_outputCM is True */
//     Output_t *rec_cg,  /* O: outputted structure for CCDC results     */
//     int *num_fc,       /* O: number of fitting curves                       */
//     int CM_OUTPUT_INTERVAL,
//     short int *CM_outputs,      /* I/O: maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
//     short int *CM_outputs_date, /* I/O: dates for maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
//     int min_days_conse);

#endif // CCD_H
