#ifndef COLDF_H
#define COLDF_H
#include <stdint.h>
#include <stdbool.h>
#include "output.h"
// #include <xgboost/c_api.h>

int cold_flex(
    int64_t *ts_data,           /* I:  multispectral time series reshaped into 1 column. Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,         /* I:  the time series of cfmask values. 0 - clear; 1 - water; 2 - shadow; 3 - snow; 4 - cloud  */
    int64_t *valid_date_array,  /* I:  valid date as python serial date form (counting from Jan 0, 0000). Note ordinal date in python is from (Jan 1th, 0001) */
    int nbands,                 /* I: input band number */
    int tmask_b1,               /* I: the band id used for tmask */
    int tmask_b2,               /* I: the band id used for tmask */
    int valid_num_scenes,       /* I: number of valid scenes  */
    int pos,                    /* I: the position id of pixel */
    double t_cg,                /* I: threshold for identfying breaks */
    double max_t_cg,            /* I: threshold for identfying outliers */
    int conse,                  /* I: consecutive observation number   */
    bool b_outputCM,            /* I: indicate if outputting change magnitudes for object-based cold, for cold only, it is the false */
    int starting_date,          /* I: (optional) the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile should have the same date, only for b_outputCM is True */
    Output_t_flex *rec_cg,      /* O: outputted structure for CCDC results    */
    int *num_fc,                /* O: number of fitting curves                   */
    int CM_OUTPUT_INTERVAL,     /* I: (optional) change magnitude output interval  */
    short int *CM_outputs,      /* I/O: (optional) maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    short int *CM_outputs_date, /* I/O: (optional) dates for maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    double gap_days,            /* I: the day number of gap to define i_dense; it is useful for the cases that gap is in the middle of time series      */
    double lambda);

int stand_procedure_flex(
    int valid_num_scenes,      /* I:  number of valid scenes  */
    int64_t *valid_date_array, /* I: valid date time series  */
    int64_t *ts_data,
    int64_t *fmask_buf, /* I:  mask-based time series  */
    int nbands,         /* I: input band number */
    int *id_range,
    double t_cg,           /* I: threshold for identfying breaks */
    double max_t_cg,       /* I: threshold for identfying outliers */
    int conse,             /* I: consecutive observation number   */
    bool b_outputCM,       /* I: indicate if cold is running as the first step of object-based cold*/
    int starting_date,     /* I: the starting date of the whole dataset to enable reconstruct CM_date, all pixels for a tile should have the same date, only for b_outputCM is True */
    Output_t_flex *rec_cg, /* O: outputted structure for CCDC results     */
    int *num_fc,           /* O: number of fitting curves                       */
    int CM_OUTPUT_INTERVAL,
    short int *CM_outputs,      /* I/O: maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    short int *CM_outputs_date, /* I/O: dates for maximum change magnitudes at every CM_OUTPUT_INTERVAL days, only for b_outputCM is True*/
    double gap_days,
    int tmask_b1, /* I: the band id used for tmask */
    int tmask_b2, /* I: the band id used for tmask */
    double lambda);

int inefficientobs_procedure_flex(
    int valid_num_scenes,      /* I:  number of scenes  */
    int64_t *valid_date_array, /* I: valid date time series  */
    int64_t *ts_data,
    int64_t *fmask_buf, /* I:  mask-based time series  */
    int nbands,         /* I: input band number */
    int *id_range,
    float sn_pct,
    Output_t_flex *rec_cg,
    int *num_fc,
    double lambda);

#endif // COLDF_H
