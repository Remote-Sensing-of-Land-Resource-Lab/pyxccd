#ifndef MISC_H
#define MISC_H
#include <stdbool.h>

int adjust_median_variogram(
    int *clrx,                 /* I: dates                                          */
    float **array,             /* I: input array                                    */
    int dim1_len,              /* I: dimension 1 length in input array              */
    int dim2_start,            /* I: dimension 2 start index                        */
    int dim2_end,              /* I: dimension 2 end index                          */
    float *date_vario,         /* O: outputted median variogran for dates           */
    float *max_neighdate_diff, /*O: maximum difference for two neighbor times       */
    float *output_array,       /* O: output array                                   */
    int option                 /* I: option for median variogram: 1 - normal; 2 - adjust (PYCCD version) */
);

double chi2inv(
    double P,
    unsigned int dim);

bool checkbit(
    int packedint,
    int offset);

int auto_mask(
    int *clrx,
    float **clry,
    int start,
    int end,
    float years,
    float t_b1,
    float t_b2,
    float n_t,
    int *bl_ids,
    int tmask_b1, /* I: the band id used for tmask */
    int tmask_b2  /* I: the band id used for tmask */
);

int auto_ts_fit(
    int *clrx,
    float **clry,
    int band_index,
    int lasso_band_index, /* this variable is the legacy of experiments on modeling only lasso bands. Zhe wanted to keep all bands still. So pass i_b to this variable*/
    int start,
    int end,
    int df,
    double **coefs,
    double *rmse,
    double **v_dif,
    float lam);

int auto_ts_predict(
    int *clrx,
    double **coefs,
    int df,
    int lasso_band_index,
    int start,
    int end,
    double *pred_y);

int auto_ts_predict_float(
    int *clrx,
    float **coefs,
    int df,
    int lasso_band_index,
    int start,
    int end,
    float *pred_y);

void update_cft(
    int i_span,
    int n_times,
    int min_num_c,
    int mid_num_c,
    int max_num_c,
    int num_c,
    int *update_number_c);

void split_directory_scenename(
    const char *filename, /* I: Name of scene with path to split    */
    char *directory,      /* O: Directory portion of file name      */
    char *scene_name      /* O: Scene name portion of the file name */
);

void rmse_from_square_root_mean(
    float **array,  /* I: input array                      */
    float fit_cft,  /* I: input fit_cft value              */
    int dim1_index, /* I: dimension 1 index in input array */
    int dim2_len,   /* I: dimension 2 length               */
    float *rmse     /* O: output rmse                      */
);

void partial_square_root_mean(
    float **array,   /* I: input array                         */
    int dim1_index,  /* I: 1st dimension index                 */
    int start,       /* I: number of start elements in 1st dim */
    int end,         /* I: number of end elements in 1st dim   */
    float **fit_ctf, /* I:                                     */
    float *rmse      /* O: output rmse value                   */
);

void matlab_2d_array_mean(
    float **array,     /* I: input array                         */
    int dim1_index,    /* I: 1st dimension index                 */
    int dim2_len,      /* I: number of input elements in 2nd dim */
    float *output_mean /* O: output norm value                   */
);

void matlab_2d_float_median(
    float **array,       /* I: input array                         */
    int dim1_index,      /* I: 1st dimension index                 */
    int dim2_len,        /* I: number of input elements in 2nd dim */
    float *output_median /* O: output norm value                   */
);

void matlab_2d_double_median(
    double **array,       /* I: input array */
    int dim1_index,       /* I: 1st dimension index */
    int dim2_len,         /* I: number of input elements in 2nd dim */
    double *output_median /* O: output norm value */
);

void matlab_2d_partial_mean(
    float **array,     /* I: input array                         */
    int dim1_index,    /* I: 1st dimension index                 */
    int start,         /* I: number of start elements in 2nd dim */
    int end,           /* I: number of end elements in 2nd dim   */
    float *output_mean /* O: output norm value                   */
);

void matlab_float_2d_partial_median(
    float **array,       /* I: input array                         */
    int dim1_index,      /* I: 1st dimension index                 */
    int start,           /* I: number of start elements in 2nd dim */
    int end,             /* I: number of end elements in 2nd dim   */
    float *output_median /* O: output norm value                   */
);

void matlab_double_2d_partial_median(
    double **array,       /* I: input array                         */
    int dim1_index,       /* I: 1st dimension index                 */
    int start,            /* I: number of start elements in 2nd dim */
    int end,              /* I: number of end elements in 2nd dim   */
    double *output_median /* O: output norm value                   */
);

void matlab_2d_partial_square_mean(
    float **array,     /* I: input array                         */
    int dim1_index,    /* I: 1st dimension index                 */
    int start,         /* I: number of start elements in 2nd dim */
    int end,           /* I: number of end elements in 2nd dim   */
    float *output_mean /* O: output norm value                   */
);

void matlab_2d_array_norm(
    double **array,     /* I: input array                         */
    int dim1_index,     /* I: 1st dimension index                 */
    int dim2_len,       /* I: number of input elements in 2nd dim */
    double *output_norm /* O: output norm value                   */
);

void matlab_2d_array_norm_float(
    float **array,     /* I: input array                                   */
    int dim1_index,    /* I: 1st dimension index                           */
    int dim2_len,      /* I: number of input elements in 2nd dim           */
    float *output_norm /* O: output norm value                             */
);

void array_1d_mean(
    float *array, /* I: input array */
    int start,    /* I: number of input elements in 2nd dim */
    int end,
    float *output_mean /* O: output norm value */
);

void get_ids_length(
    int *id_array, /* I: input array */
    int start,     /* I: array start index */
    int end,       /* I: array end index */
    int *id_len    /* O: number of non-zero number in the array */
);

void matlab_unique(
    int *clrx,
    float **clry,
    int nums,
    int *new_nums);

extern void elnet_(

    // input:

    int *ka,     //   ka = algorithm flag
                 //      ka=1 => covariance updating algorithm
                 //      ka=2 => naive algorithm
    float *parm, //   parm = penalty member index (0 <= parm <= 1)
                 //        = 0.0 => ridge
                 //        = 1.0 => lasso
    int *no,     //   no = number of observations
    int *ni,     //   ni = number of predictor variables
    float *x,    //   x[ni][no] = predictor data matrix flat file (overwritten)
    float *y,    //   y[no] = response vector (overwritten)
    float *w,    //   w[no]= observation weights (overwritten)
    int *jd,     //   jd(jd(1)+1) = predictor variable deletion flag
                 //      jd(1) = 0  => use all variables
                 //      jd(1) != 0 => do not use variables jd(2)...jd(jd(1)+1)
    float *vp,   //   vp(ni) = relative penalties for each predictor variable
                 //      vp(j) = 0 => jth variable unpenalized
    // double cl[][2],	//   cl(2,ni) = interval constraints on coefficient values (overwritten)
    //       cl(1,j) = lower bound for jth coefficient value (<= 0.0)
    //       cl(2,j) = upper bound for jth coefficient value (>= 0.0)
    int *ne,      //   ne = maximum number of variables allowed to enter largest model
                  //        (stopping criterion)
    int *nx,      //   nx = maximum number of variables allowed to enter all models
                  //        along path (memory allocation, nx > ne).
    int *nlam,    //   nlam = (maximum) number of lamda values
    float *flmin, //   flmin = user control of lamda values (>=0)
                  //      flmin < 1.0 => minimum lamda = flmin*(largest lamda value)
                  //      flmin >= 1.0 => use supplied lamda values (see below)
    float *ulam,  //   ulam(nlam) = user supplied lamda values (ignored if flmin < 1.0)
    float *thr,   //   thr = convergence threshold for each lamda solution.
                  //      iterations stop when the maximum reduction in the criterion value
                  //      as a result of each parameter update over a single pass
                  //      is less than thr times the null criterion value.
                  //      (suggested value, thr=1.0e-5)
    int *isd,     //   isd = predictor variable standarization flag:
                  //      isd = 0 => regression on original predictor variables
                  //      isd = 1 => regression on standardized predictor variables
                  //      Note: output solutions always reference original
                  //            variables locations and scales.
                  // int *intr,		//   intr = intercept flag
                  //      intr = 0/1 => don't/do include intercept in model
    // int *maxit,		//   maxit = maximum allowed number of passes over the data for all lambda
    //       values (suggested values, maxit = 100000)

    // output:

    int *lmu,   //   lmu = actual number of lamda values (solutions)
    float *a0,  //   a0(lmu) = intercept values for each solution
    float *ca,  //   ca(nx,lmu) = compressed coefficient values for each solution
    int *ia,    //   ia(nx) = pointers to compressed coefficients
    int *nin,   //   nin(lmu) = number of compressed coefficients for each solution
    float *rsq, //   rsq(lmu) = R**2 values for each solution
    float *alm, //   alm(lmu) = lamda values corresponding to each solution
    int *nlp,   //   nlp = actual number of passes over the data for all lamda values
    int *jerr   //   jerr = error flag:
                //      jerr  = 0 => no error
                //      jerr > 0 => fatal error - no output returned
                //         jerr < 7777 => memory allocation error
                //         jerr = 7777 => all used predictors have zero variance
                //         jerr = 10000 => maxval(vp) <= 0.0
                //      jerr < 0 => non fatal error - partial output:
                //         Solutions for larger lamdas (1:(k-1)) returned.
                //         jerr = -k => convergence for kth lamda value not reached
                //            after maxit (see above) iterations.
                //         jerr = -10000-k => number of non zero coefficients along path
                //            exceeds nx (see above) at kth lamda value.
);

extern void spelnet_(

    // input:

    int *ka,      //   ka = algorithm flag
                  //      ka=1 => covariance updating algorithm
                  //      ka=2 => naive algorithm
    double *parm, //   parm = penalty member index (0 <= parm <= 1)
                  //        = 0.0 => ridge
                  //        = 1.0 => lasso
    int *no,      //   no = number of observations
    int *ni,      //   ni = number of predictor variables
    double *x,    //   x[ni][no] = predictor data matrix flat file (overwritten)
    double *y,    //   y[no] = response vector (overwritten)
    double *w,    //   w[no]= observation weights (overwritten)
    int *jd,      //   jd(jd(1)+1) = predictor variable deletion flag
                  //      jd(1) = 0  => use all variables
                  //      jd(1) != 0 => do not use variables jd(2)...jd(jd(1)+1)
    double *vp,   //   vp(ni) = relative penalties for each predictor variable
                  //      vp(j) = 0 => jth variable unpenalized
    // double cl[][2],	//   cl(2,ni) = interval constraints on coefficient values (overwritten)
    //       cl(1,j) = lower bound for jth coefficient value (<= 0.0)
    //       cl(2,j) = upper bound for jth coefficient value (>= 0.0)
    int *ne,       //   ne = maximum number of variables allowed to enter largest model
                   //        (stopping criterion)
    int *nx,       //   nx = maximum number of variables allowed to enter all models
                   //        along path (memory allocation, nx > ne).
    int *nlam,     //   nlam = (maximum) number of lamda values
    double *flmin, //   flmin = user control of lamda values (>=0)
                   //      flmin < 1.0 => minimum lamda = flmin*(largest lamda value)
                   //      flmin >= 1.0 => use supplied lamda values (see below)
    double *ulam,  //   ulam(nlam) = user supplied lamda values (ignored if flmin < 1.0)
    double *thr,   //   thr = convergence threshold for each lamda solution.
                   //      iterations stop when the maximum reduction in the criterion value
                   //      as a result of each parameter update over a single pass
                   //      is less than thr times the null criterion value.
                   //      (suggested value, thr=1.0e-5)
    int *isd,      //   isd = predictor variable standarization flag:
                   //      isd = 0 => regression on original predictor variables
                   //      isd = 1 => regression on standardized predictor variables
                   //      Note: output solutions always reference original
                   //            variables locations and scales.
    // int *intr,		//   intr = intercept flag
    //       intr = 0/1 => don't/do include intercept in model
    // int *maxit,		//   maxit = maximum allowed number of passes over the data for all lambda
    //       values (suggested values, maxit = 100000)

    // output:

    int *lmu,    //   lmu = actual number of lamda values (solutions)
    double *a0,  //   a0(lmu) = intercept values for each solution
    double *ca,  //   ca(nx,lmu) = compressed coefficient values for each solution
    int *ia,     //   ia(nx) = pointers to compressed coefficients
    int *nin,    //   nin(lmu) = number of compressed coefficients for each solution
    double *rsq, //   rsq(lmu) = R**2 values for each solution
    double *alm, //   alm(lmu) = lamda values corresponding to each solution
    int *nlp,    //   nlp = actual number of passes over the data for all lamda values
    int *jerr    //   jerr = error flag:
                 //      jerr  = 0 => no error
                 //      jerr > 0 => fatal error - no output returned
                 //         jerr < 7777 => memory allocation error
                 //         jerr = 7777 => all used predictors have zero variance
                 //         jerr = 10000 => maxval(vp) <= 0.0
                 //      jerr < 0 => non fatal error - partial output:
                 //         Solutions for larger lamdas (1:(k-1)) returned.
                 //         jerr = -k => convergence for kth lamda value not reached
                 //            after maxit (see above) iterations.
                 //         jerr = -10000-k => number of non zero coefficients along path
                 //            exceeds nx (see above) at kth lamda value.
);

/*--------------------------------------------------------------------
c uncompress coefficient vectors for all solutions:
c
c call solns(ni,nx,lmu,ca,ia,nin,b)
c
c input:
c
c    ni,nx = input to elnet
c    lmu,ca,ia,nin = output from elnet
c
c output:
c
c    b(ni,lmu) = all elnet returned solutions in uncompressed format
----------------------------------------------------------------------*/
// extern int solns_(
//     int *ni,            //   ni = number of predictor variables
//     int *nx,            //   nx = maximum number of variables allowed to enter all models
//     int *lmu,           //   lmu = actual number of lamda values (solutions)
//     double *ca,         //   ca(nx,lmu) = compressed coefficient values for each solution
//     int *ia,            //   ia(nx) = pointers to compressed coefficients
//     int *nin,           //   nin(lmu) = number of compressed coefficients for each solution
//     double *b           //   b(ni,lmu) = compressed coefficient values for each solution
//);

extern int c_glmnet(
    int no,      // number of observations (no)
    int ni,      // number of predictor variables (ni)
    float *x,    // input matrix, x[ni][no]
    float *y,    // response vaiable, of dimentions (no)
    int nlam,    // number of lambda values
    float *ulam, // value of lambda values, of dimentions (nlam)
    float parm,  // the alpha variable

    int *lmu,                // lmu = actual number of lamda values (solutions)
    double cfs[nlam][ni + 1] // results = cfs[lmu][ni + 1]
);

void max_array_int(
    int a[],
    int num_elements,
    int *max_element,
    int *max_loc);

int partition_2d_float(
    float arr[],
    float *brr[],
    int left,
    int right,
    int band_num);

void quick_sort_2d_float(
    float arr[],
    float *brr[],
    int left,
    int right,
    int band_numb);

void quick_sort_2d_double(
    double arr[],
    double *brr[],
    int left,
    int right,
    int band_num);

double normalCDF(double u);

double normalQuantile(double p);

double MeanAngl(
    double **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,   // input: the number of lasso band
    int i_count      // input: the number of consecutive observations
);

float MediumAngl(
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count     // input: the number of consecutive observations
);

float angl_scatter_measure(
    float *med_diff,
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count,    // input: the number of consecutive observations
    int *lasso_bands);

int singleband_variogram(
    float *array,    /* I: input array                                    */
    int i_start,     /* I: dimension 2 start index                        */
    int i_end,       /* I: dimension 2 end index                          */
    float *variogram /* O: outputted median variogran for dates           */
);

int singleband_minvariogram(
    int *clrx,
    float *array,    /* I: input array                                    */
    int i_start,     /* I: dimension 2 start index                        */
    int i_end,       /* I: dimension 2 end index                          */
    float *variogram /* O: outputted median variogran for dates           */
);

int yearmonth2doy(
    int year,
    int month,
    int day);

int qabitval(
    int packedint);

float MeanAngl_float(
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count     // input: the number of consecutive observations
);

float MeanAngl_float_selective(
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count,    // input: the number of consecutive observations
    int *lasso_bands);

int auto_ts_fit_float(
    int *clrx,
    float **clry,
    int band_index,
    int lasso_band_index, /* this variable is the legacy of experiments on modeling only lasso bands. Zhe wanted to keep all bands still. So pass i_b to this variable*/
    int start,
    int end,
    int df,
    float **coefs,
    float *rmse,
    float **v_dif,
    float lam);

int auto_ts_fit_sccd(
    int *clrx,
    float **clry,
    int band_index,
    int lasso_band_index, /* this variable is the legacy of experiments on modeling only lasso bands. Zhe wanted to keep all bands still. So pass i_b to this variable*/
    int start,
    int end,
    int df,
    float **coefs,
    float *rmse,
    float **v_dif,
    float lam);

void usage();

// Internally defined routines
double X2(int nu, double pr);

#endif // MISC_H
