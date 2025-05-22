/* this is an effort to consolidate all defines.  Previously, they    */
/* scattered throughout .c and/or .h files, and not always included   */
/* and/or avilable everywhere or where needed.  Also, some reduncancy */
/* and conflicts existed.                                             */

/* from ccdc.c */
#define TOTAL_IMAGE_BANDS 7           /* Number of image bands, for loops.   */
#define TOTAL_IMAGE_BANDS_FLEX 10     /* Number of image bands for flexibale mode.   */
#define TOTAL_IMAGE_BANDS_FLEX_SCCD 8 /* Number of image bands for flexibale mode.   */
#define TOTAL_IMAGE_BANDS_SCCD 6      /* Number of image bands, for loops.   */
#define TOTAL_INDICES 0               /* Number of image bands, for loops. */
#define TOTAL_BANDS 8                 /* Total image plus mask bands, for loops.  HLS */
#define MIN_NUM_C 4                   /* Minimum number of coefficients           */
#define MID_NUM_C 6                   /* Mid-point number of coefficients         */
#define SCCD_MAX_NUM_C 6              /* Maximum number of coefficients           */
#define SCCD_NUM_C 6                  /* Mid-point number of coefficients         */
#define MAX_NUM_C 8                   /* Maximum number of coefficients           */
#define N_TIMES 3                     /* number of clear observations/coefficients*/
#define NUM_YEARS 365.25              /* average number of days per year          */
#define NUM_FC 40                     /*  the maximum number of fitting curve to be outputted */
#define T_CONST 4.42                  /* Threshold for cloud, shadow, and snow detection */
                                      /* Old T_CONST = 4.89 SY 11/14/2018         */
#define MIN_YEARS 1                   /* minimum year for model intialization     */
#define T_SN 0.75                     /* no change detection for permanent snow pixels */
#define T_CLR 0.25                    /* Fmask fails threshold                    */
#define T_MAX_CG_reconstruct 15.0863
#define T_MAX_CG 30.8562 /* chi-square inversed T_max_cg (1e-5) for    */
                         /* last step noise removal                    */
                         /*  SY 11/19/2018: change from 35.8882 to 30.8562)                  */
#define T_MAX_CG_SCCD 25.7448
// #define T_MIN_CG_SCCD 9.236
// #define T_MAX_CG_SCCD 30.8562
// #define T_MAX_CG_SCCD 20.515

/* threshold (degree) of mean included angle                          */
#define NSIGN 45

/* from 2darray.c */
/* Define a unique (i.e. random) value that can be used to verify a pointer
   points to an LSRD_2D_ARRAY. This is used to verify the operation succeeds to
   get an LSRD_2D_ARRAY pointer from a row pointer. */
#define SIGNATURE 0x326589ab

/* Given an address returned by the allocate routine, get a pointer to the
   entire structure. */
#define GET_ARRAY_STRUCTURE_FROM_PTR(ptr) \
   ((LSRD_2D_ARRAY *)((char *)(ptr) - offsetof(LSRD_2D_ARRAY, memory_block)))

/* from input.c */
// #define TOTAL_IMAGE_BANDS 7

/* from misc.c */
/* 12-31-1972 is 720258 in julian day since year 0001 */
/* matlab use 720624 since year 0000 */
#define ORDINAL_DATE_LAST_DAY_1972 720258
#define ORDINAL_LANDSAT4_LAUNCH 723742 /* 7-16-1982*/
#define ORDINAL_DATE_1982_1_1 723546   /* 7-16-1982*/
#define LANDSAT_START_YEAR 1973
#define LEAP_YEAR_DAYS 366
#define NON_LEAP_YEAR_DAYS 365
#define AVE_DAYS_IN_A_YEAR 365.25
#define ROBUST_COEFFS 5
#define LASSO_COEFFS 8
// #define SCCD_COEFFS 6
// #define TOTAL_IMAGE_BANDS 7

/* from input.h */
/* possible cfmask values */
#define CFMASK_CLEAR 0
#define CFMASK_WATER 1
#define CFMASK_SHADOW 2
#define CFMASK_SNOW 3
#define CFMASK_CLOUD 4
#define CFMASK_FILL 255
#define IMAGE_FILL -9999

#define CFMASK_BAND 7

/* from output.h */
#define FILL_VALUE 255
#define NUM_COEFFS 8
#define NUM_BANDS 7

/* from s_ccd.c */
#define ANNUAL_STATE 1
#define SEMIANNUAL_STATE 2

/* step3_KF_ChangeDetection RETURN_VALUE*/
#define REGULAREND 0
#define CHANGEDETECTED 1
#define FALSECHANGE 2

/* for ini_mode in sccd */
#define SIMPLE 1
#define CONSE_END 6

#define UPDATE_FREQ 1
// #define CONSE 6           /* No. of CONSEquential pixels 4 bldg. model*/
#define LASSO_MIN 6 /* No. of CONSEquential pixels 4 bldg. model*/

#define INITIAL_P_RATIO 0.25
#define SCCD_T_CONST 4.42

#define DEFAULT_N_STATE 6
#define HALF_DEFAULT_N_STATE 3
#define NUM_LASSO_BANDS 5 /* Number of bands for Least Absolute Shrinkage */
                          /* and Selection Operator LASSO regressions */
#define DEFAULT_COLD_TCG 15.0863

#define INI_Q00 0.25 /* MATLAB default */
#define INI_P 1000000

/* from ccd.c detection method*/
// #define CCD 1
#define COLD 1
#define SCCD 2
#define OBCOLD 3 // COLD with outputting CM magnitudes
#define OBCOLD_RECONSTRUCT 4
#define SCCDONLINE 5
#define COLD_FLEX 6
#define SCCD_FLEX 7

#define SKIP_PERCENTAGE 0.03

#define TIFF_FORMAT 1
#define ENVI_FORMAT 2

#define NA_VALUE -9999

#define N_FEATURE 7
#define DEFAULT_PROBABILITY 0.99
// #define DEFAULT_CONSE 6
#define DEFAULT_CONSE_SCCD 8
#define JULY1ST_DOY 183
#define SLOPE_SCALE 10000 // the parameter to guarantee the precision for converting scale double slope to float slope

/* pixel status for NRT */
#define NRT_VOID 0
#define NRT_MONITOR_STANDARD 1
#define NRT_QUEUE_STANDARD 2
#define NRT_MONITOR_SNOW 3
#define NRT_QUEUE_SNOW 4
#define NRT_MONITOR2QUEUE 5
#define MAX_OBS_QUEUE 240 /* the maximum number of observations in queue (for NRT) */

#define LANDSAT45_TM 4
#define LANDSAT7_ETM 5
#define LANDSAT8_OLI 8

#define KFAS_TOL 1E-14
#define KFAS_LIK_C 0.5 * log(8.0 * atan(1.0))
#define SLOPE_SS_SCALE 10000000
#define PINPOINT_CONSE 3
#define MAX_SHORT 32767
#define CORRECT_RATIO_PREDICTABILITY 0.5
#define STATUS_DELEY_DAYS 15
#define MAX_VALID_OBS 10000
#define DEFAULT_lambda 20
// #define RECENT_CHANGE_LAST_DAY 15
