#ifndef CCD_STOCHASTIC_F
#define CCD_STOCHASTIC_F

#include <stdint.h>
#include "KFAS.h"
#include "output.h"

int sccd_flex(
    int64_t *ts_data,                           /* I:  multispectral time series reshaped into 1 column. Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,                         /* I:  mask-based time series              */
    int64_t *valid_date_array,                  /* I: valid date time series               */
    int nbands,                                 /* I: input band number */
    int tmask_b1,                               /* I: the band id used for tmask */
    int tmask_b2,                               /* I: the band id used for tmask */
    int valid_num_scenes,                       /* I: number of valid scenes under cfmask fill counts  */
    double tcg,                                 /* I: the change threshold  */
    double max_t_cg,                            /* I: threshold for identfying outliers */
    int *num_fc,                                /* O: number of fitting curves                       */
    int *nrt_mode,                              /* O: 2nd digit: 0 - void; 1 - monitor mode for standard; 2 - queue mode for standard; 3 - monitor mode for snow; 4 - queue mode for snow; 1st digit: 1 - predictability untest */
    Output_sccd_flex *rec_cg,                   /* O: historical change records for SCCD results    */
    output_nrtmodel_flex *nrt_model,            /* O: nrt model structure for SCCD results    */
    int *num_obs_queue,                         /* O: the number of multispectral observations    */
    output_nrtqueue_flex *obs_queue,            /* O: multispectral observations in queue    */
    short int *min_rmse,                        /* O: adjusted rmse for the pixel    */
    int conse,                                  /* I: consecutive observation number for change detection   */
    bool b_c2,                                  /* I: a temporal parameter to indicate if collection 2. C2 needs ignoring thermal band due to the current low quality  */
    bool b_pinpoint,                            /* I: output pinpoint break for training purpose  */
    Output_sccd_pinpoint_flex *rec_cg_pinpoint, /* O: historical change records for SCCD results    */
    int *num_fc_pinpoint,
    double gate_tcg,
    double predictability_tcg,
    bool b_output_state, /* I: indicate whether to output state  */
    double state_intervaldays,
    int *n_state,
    int64_t *state_days,
    double *states_ensemble, /* O: states records for blue band */
    bool b_fitting_coefs,
    double lambda);

int step1_ssm_initialize_flex(
    ssmodel_constants *instance, /* I/O: the outputted initial SSM model, we will assign H     */
    int *clrx,                   /* I: clear pixel curve in X direction (date)             */
    float *clry,                 /* I: clear pixel curve in Y direction (spectralbands)    */
    int stable_start,            /* I:  the start of the stable stage  */
    int stable_end,              /* I:  the start of the stable stage  */
    float **fit_cft,             /*I: the lasso coefficientis */
    gsl_matrix *cov_p,           /* I/O:  initial P1  */
    int i_b,                     /* I:  the band order */
    unsigned int *sum_square_vt, /* I/O:  the sum of predicted square of residuals  */
    int n_clr,
    bool b_coefs_records,
    int *n_coefs_records,
    nrt_coefs_records_flex *coefs_records,
    int nbands,
    double lambda);

int step1_cold_initialize_flex(
    int conse,           /* I: adjusted consecutive observation number               */
    short int *min_rmse, /* I: the adjusted RMS                        */
    int *n_clr,          /* I: number of clear observations                         */
    double tcg,          /* I: the threshold of change magnitude                       */
    double max_t_cg,
    int *i_dense,             /* I: used to count i for dense time point check          */
    int *num_curve,           /* I/O: the number of fitting curve                        */
    int *clrx,                /* I/O: clear pixel curve in X direction (date)             */
    float **clry,             /* I/O: clear pixel curve in Y direction (spectralbands)    */
    int *cur_i,               /* I/O: the current number of monitoring observation          */
    int *i_start,             /* I/O: the start number of current curve                   */
    Output_sccd_flex *rec_cg, /* I/O: records of change points info                    */
    int i_span_min,           /* I: the minimum value for i_span                    */
    int *prev_i_break,        /*I : the i_break of the last curve                    */
    float *rmse,              /* I/O: Root Mean Squared Error array used for initialized kalman filter model    */
    int nbands,
    int tmask_b1,
    int tmask_b2,
    double lambda);

int step2_KF_ChangeDetection_flex(
    ssmodel_constants *instance, /* I: ssm constant structure */
    int *clrx,                   /* I: dates   */
    float **clry,                /* I: observations   */
    int cur_i,                   /* I: the ith of observation to be processed   */
    int i_start,
    int *num_curve,              /* I: the number of curves   */
    int conse,                   /* I: the consecutive number of observations   */
    short int *min_rmse,         /* I: adjusted RMSE   */
    float tcg,                   /* I: the change threshold  */
    int *n_clr,                  /* I: the total observation of current observation queue  */
    gsl_matrix **cov_p,          /* I/O: covariance matrix */
    float **fit_cft,             /* I/O: state variables  */
    Output_sccd_flex *rec_cg,    /* I/O: the outputted S-CCD result structure   */
    unsigned int *sum_square_vt, /* I/O:  the sum of predicted square of residuals  */
    int *num_obs_processed,      /* I/O:  the number of current non-noise observations being processed */
    int t_start,
    bool b_pinpoint,
    Output_sccd_pinpoint_flex *rec_cg_pinpoint, /* O: historical change records for SCCD results    */
    int *num_fc_pinpoint,
    double gate_tcg,
    short int *norm_cm_scale100,
    short int *mean_angle_scale100,
    float *CM_outputs,
    float t_max_cg_sccd, /* I: the threshold of identifying outliers */
    bool b_coefs_records,
    int *n_coefs_records,
    nrt_coefs_records_flex *coefs_records,
    int nbands,
    bool b_fitting_coefs,
    double lambda);

/************************************************************************
FUNCTION: step3_processingend

PURPOSE:
Step 3 of S-CCD: processing the end of time series.
RETURN VALUE:
Type = int (SUCCESS OR FAILURE)

Programmer: Su Ye
**************************************************************************/
int step3_processing_end_flex(
    ssmodel_constants *instance,
    gsl_matrix **cov_p,
    float **fit_cft,
    int *clrx,
    float **clry,
    int cur_i,
    int *n_clr,
    int *nrt_mode,
    int i_start,
    int prev_i_break,                /* I: the i_break of the last curve*/
    output_nrtmodel_flex *nrt_model, /* I/O: the NRT change records */
    int *num_obs_queue,              /* O: the number of multispectral observations    */
    output_nrtqueue_flex *obs_queue, /* O: multispectral observations in queue    */
    unsigned *sum_square_vt,         /* I/O:  the sum of predicted square of residuals  */
    int num_obs_processed,
    int t_start,
    int conse,
    short int *min_rmse,
    double gate_tcg,
    bool change_detected,
    double predictability_tcg,
    int nbands,
    double lambda);

int sccd_snow_flex(
    int *clrx,    /* I: clear pixel curve in X direction (date)             */
    float **clry, /* I: clear pixel curve in Y direction (spectralbands)    */
    int n_clr,
    int *nrt_status,                 /* O: 1 - monitor mode; 2 - queue mode    */
    output_nrtmodel_flex *nrt_model, /* O: nrt records    */
    int *num_obs_queue,              /* O: the number of multispectral observations    */
    output_nrtqueue_flex *obs_queue, /* O: multispectral observations in queue    */
    bool b_coefs_records,
    int *n_coefs_records,
    nrt_coefs_records_flex *coefs_records,
    int nbands,
    double lambda);

int sccd_standard_flex(
    int *clrx,    /* I: clear pixel curve in X direction (date)             */
    float **clry, /* I: clear pixel curve in Y direction (spectralbands)    */
    int *n_clr,
    double tcg, /* I:  threshold of change magnitude   */
    double max_t_cg,
    Output_sccd_flex *rec_cg,        /* O: offline change records */
    int *num_fc,                     /* O: intialize NUM of Functional Curves    */
    int *nrt_mode,                   /* O: 1 - monitor mode; 2 - queue mode    */
    output_nrtmodel_flex *nrt_model, /* O: nrt records    */
    int *num_obs_queue,              /* O: the number of multispectral observations    */
    output_nrtqueue_flex *obs_queue, /* O: multispectral observations in queue    */
    short int *min_rmse,             /* O: adjusted rmse for the pixel    */
    int conse,
    bool b_pinpoint,
    Output_sccd_pinpoint_flex *rec_cg_pinpoint, /* O: historical change records for SCCD results    */
    int *num_fc_pinpoint,
    double gate_tcg,
    double predictability_tcg,
    bool b_coefs_records,
    int *n_coefs_records,
    nrt_coefs_records_flex *coefs_records,
    int nbands,
    int tmask_b1,
    int tmask_b2,
    bool b_fitting_coefs,
    double lambda);
#endif // CCD_STOCHASTIC_F