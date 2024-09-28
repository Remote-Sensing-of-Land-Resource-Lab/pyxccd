#ifndef OUTPUT_H
#define OUTPUT_H
#include <stdbool.h>
#include "defines.h"
#include "input.h"

typedef struct
{
    int row;
    int col;
} Position_t;

/* Structure for the 'output' data type */
typedef struct
{
    int t_start;        /* time when series model gets started */
    int t_end;          /* time when series model gets ended */
    int t_break;        /* time when the first break (change) is observed */
    int pos;            /* the location of each time series model */
    int num_obs;        /* the number of "good" observations used for model
                           estimation */
    short int category; /* the quality of the model estimation (what model
                     is used, what process is used) */
    /*
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
    8: model has 7 coefs + 1 const*/
    short int change_prob; /* the probability of a pixel that have undergone
                              change (between 0 and 100) */
    float coefs[TOTAL_IMAGE_BANDS][NUM_COEFFS];
    /*  coefficients for each time series model for each
        spectral band*/
    float rmse[TOTAL_IMAGE_BANDS];
    /*  RMSE for each time series model for each
        spectral band*/
    float magnitude[TOTAL_IMAGE_BANDS]; /* the magnitude of change (difference between model
                                   prediction and observation for each spectral band)*/
} Output_t;

typedef struct
{
    int t_start;        /* time when series model gets started */
    int t_end;          /* time when series model gets ended */
    int t_break;        /* time when the first break (change) is observed */
    int pos;            /* the location of each time series model */
    int num_obs;        /* the number of "good" observations used for model
                           estimation */
    short int category; /* the quality of the model estimation (what model
                     is used, what process is used) */
    /*
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
    8: model has 7 coefs + 1 const*/
    short int change_prob; /* the probability of a pixel that have undergone
                              change (between 0 and 100) */
    float coefs[TOTAL_IMAGE_BANDS_FLEX][NUM_COEFFS];
    /*  coefficients for each time series model for each
        spectral band*/
    float rmse[TOTAL_IMAGE_BANDS_FLEX];
    /*  RMSE for each time series model for each
        spectral band*/
    float magnitude[TOTAL_IMAGE_BANDS_FLEX]; /* the magnitude of change (difference between model
                                   prediction and observation for each spectral band)*/
} Output_t_flex;

typedef struct
{
    int t_start; /* time when series model gets started */
    int t_break; /* time when the first break (change) is observed */
    // float change_prob;     /* the probability of a pixel that have undergone */
    /* change (between 0 and 100) */
    int num_obs; /* the number of "good" observations used for model
                    estimation */
    float coefs[TOTAL_IMAGE_BANDS_SCCD][SCCD_NUM_C];
    /*  coefficients for each time series model for each
        spectral band*/
    float rmse[TOTAL_IMAGE_BANDS_SCCD];

    float magnitude[TOTAL_IMAGE_BANDS_SCCD]; /* the magnitude of change (difference between model
                                   prediction and observation for each spectral band)*/

} Output_sccd;

typedef struct
{
    int t_break; /* time when the first break (change) is observed */
    float coefs[TOTAL_IMAGE_BANDS_SCCD][SCCD_NUM_C];
    short int obs[TOTAL_IMAGE_BANDS_SCCD][DEFAULT_CONSE_SCCD];
    short int obs_date_since1982[DEFAULT_CONSE_SCCD];
    short int norm_cm[DEFAULT_CONSE_SCCD];
    short int cm_angle[DEFAULT_CONSE_SCCD];
} Output_sccd_pinpoint;

typedef struct
{
    short int t_start_since1982; /* dates (Julian dates - JULIAN_DATE_LAST_DAY_1972) when series model gets started */
    short int num_obs;
    short int obs[TOTAL_IMAGE_BANDS_SCCD][DEFAULT_CONSE_SCCD];                   /* the last observations, d=(TOTAL_IMAGE_BANDS, conse - 1) */
    short int obs_date_since1982[DEFAULT_CONSE_SCCD];                            /* dates (Julian dates - JULIAN_DATE_LAST_DAY_1972) the for observations, d=(TOTAL_IMAGE_BANDS, conse - 1) */
    float covariance[TOTAL_IMAGE_BANDS_SCCD][DEFAULT_N_STATE * DEFAULT_N_STATE]; /* covariance matrix,  d=(TOTAL_IMAGE_BANDS, SCCD_NUM_C * SCCD_NUM_C), the corresponding date is the first element of conse_obs_date */
    float nrt_coefs[TOTAL_IMAGE_BANDS_SCCD][SCCD_NUM_C];                         /* state matrix, d=(TOTAL_IMAGE_BANDS, SCCD_NUM_C)  */
    float H[TOTAL_IMAGE_BANDS_SCCD];                                             /*  observation noice, d=TOTAL_IMAGE_BANDS   */
    unsigned int rmse_sum[TOTAL_IMAGE_BANDS_SCCD];
    short int norm_cm;
    short int cm_angle;
    unsigned char conse_last;
} output_nrtmodel;

typedef struct
{
    short int clry[TOTAL_IMAGE_BANDS_SCCD];
    short int clrx_since1982;
} output_nrtqueue;

typedef struct
{
    float nrt_coefs[TOTAL_IMAGE_BANDS_SCCD][SCCD_NUM_C];
    int clrx;
} nrt_coefs_records;

typedef struct
{
    float nrt_coefs[TOTAL_IMAGE_BANDS_SCCD][HALF_DEFAULT_N_STATE];
    int clrx;
} nrt_state_records;

/* The below are sccd flexible mode*/
typedef struct
{
    int t_start; /* time when series model gets started */
    int t_break; /* time when the first break (change) is observed */
    // float change_prob;     /* the probability of a pixel that have undergone */
    /* change (between 0 and 100) */
    int num_obs; /* the number of "good" observations used for model
                    estimation */
    float coefs[TOTAL_IMAGE_BANDS_FLEX_SCCD][SCCD_NUM_C];
    /*  coefficients for each time series model for each
        spectral band*/
    float rmse[TOTAL_IMAGE_BANDS_FLEX_SCCD];

    float magnitude[TOTAL_IMAGE_BANDS_FLEX_SCCD]; /* the magnitude of change (difference between model
                                   prediction and observation for each spectral band)*/

} Output_sccd_flex;

typedef struct
{
    int t_break; /* time when the first break (change) is observed */
    float coefs[TOTAL_IMAGE_BANDS_FLEX_SCCD][SCCD_NUM_C];
    short int obs[TOTAL_IMAGE_BANDS_FLEX_SCCD][DEFAULT_CONSE_SCCD];
    short int obs_date_since1982[DEFAULT_CONSE_SCCD];
    short int norm_cm[DEFAULT_CONSE_SCCD];
    short int cm_angle[DEFAULT_CONSE_SCCD];
} Output_sccd_pinpoint_flex;

typedef struct
{
    short int t_start_since1982; /* dates (Julian dates - JULIAN_DATE_LAST_DAY_1972) when series model gets started */
    short int num_obs;
    short int obs[TOTAL_IMAGE_BANDS_FLEX_SCCD][DEFAULT_CONSE_SCCD];                   /* the last observations, d=(TOTAL_IMAGE_BANDS, conse - 1) */
    short int obs_date_since1982[DEFAULT_CONSE_SCCD];                                 /* dates (Julian dates - JULIAN_DATE_LAST_DAY_1972) the for observations, d=(TOTAL_IMAGE_BANDS, conse - 1) */
    float covariance[TOTAL_IMAGE_BANDS_FLEX_SCCD][DEFAULT_N_STATE * DEFAULT_N_STATE]; /* covariance matrix,  d=(TOTAL_IMAGE_BANDS, SCCD_NUM_C * SCCD_NUM_C), the corresponding date is the first element of conse_obs_date */
    float nrt_coefs[TOTAL_IMAGE_BANDS_FLEX_SCCD][SCCD_NUM_C];                         /* state matrix, d=(TOTAL_IMAGE_BANDS, SCCD_NUM_C)  */
    float H[TOTAL_IMAGE_BANDS_FLEX_SCCD];                                             /*  observation noice, d=TOTAL_IMAGE_BANDS   */
    unsigned int rmse_sum[TOTAL_IMAGE_BANDS_FLEX_SCCD];
    short int norm_cm;
    short int cm_angle;
    unsigned char conse_last;
} output_nrtmodel_flex;

typedef struct
{
    short int clry[TOTAL_IMAGE_BANDS_FLEX_SCCD];
    short int clrx_since1982;
} output_nrtqueue_flex;

typedef struct
{
    float nrt_coefs[TOTAL_IMAGE_BANDS_FLEX_SCCD][SCCD_NUM_C];
    int clrx;
} nrt_coefs_records_flex;

typedef struct
{
    float nrt_coefs[TOTAL_IMAGE_BANDS_FLEX_SCCD][HALF_DEFAULT_N_STATE];
    int clrx;
} nrt_state_records_flex;

/* SY 09242018 */
int firstDegradationYear(
    Output_t *t,      /*I: outputted structure */
    int num_fc,       /*I: the number of inputted models */
    int n_lassoband,  /*I:the number of lasso band to focus on*/
    bool updirection, /*I: updirection or downdirection for the inputted band */
    int *dyear        /*O: outputed degradation year */
);

int write_raw_binary(
    FILE *rb_fptr, /* I: pointer to the raw binary file */
    int nlines,    /* I: number of lines to write to the file */
    int nsamps,    /* I: number of samples to write to the file */
    int size,      /* I: number of bytes per pixel (ex. sizeof(uint8)) */
    int *img_array /* I: array of nlines * nsamps * size to be written
                          to the raw binary file */
);

void write_envi_header(
    FILE *fptr,        /* I: pointer to the raw binary file */
    Input_meta_t *meta /* I: saved header file info */
);

int write_output_binary(
    FILE *fptr, /* I: pointer to the binary file */
    Output_t t  /* I: outputted structure     */
);

int write_output_binary_sccd(
    FILE *fptr,   /* I: pointer to the binary file */
    Output_sccd t /* I: outputted structure     */
);

#endif
