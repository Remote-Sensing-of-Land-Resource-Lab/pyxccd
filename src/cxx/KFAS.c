/* c translation for KFAS R package */
#include <math.h>
#include "KFAS.h"
#include "lbfgs.h"
#include "const.h"
#include "defines.h"

/* non-diffuse filtering for prediction */
void filter1step_missingobs(
    gsl_vector *zt,  /*I */
    double ht,       /*I */
    gsl_matrix *tt,  /*I */
    gsl_matrix *rqr, /*I */
    double *ft,      /*O*/
    gsl_matrix *pt,  /*I/O*/
    gsl_vector *kt,  /*I/O*/
    int m            /*I*/
)
{
    gsl_matrix *mm;

    // memcpy(zt_sub, &zt[1], 2*sizeof(*a));
    mm = gsl_matrix_alloc(m, m);
    /* mm = tt*pt*/
    gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, pt, tt, 0.0, mm);

    /* pt = mm * tt^T */
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mm, tt, 0.0, pt);
    gsl_matrix_add(pt, rqr);

    gsl_matrix_free(mm);

    /* force negative to be zero */
    //    for(k1 = 0; k1 < m; k1++){
    //        for(k2 = 0; k2 < m; k2++){
    //            if (gsl_matrix_get(pt, k1, k2) < 0){
    //                gsl_matrix_set(pt, k1, k2, 0.0);
    //            }
    //        }
    //    }
    return;
}

void filter1step_validobs(
    float yt,        /*I */
    gsl_vector *zt,  /*I */
    float *ht,       /*I */
    gsl_matrix *tt,  /*I */
    gsl_matrix *rqr, /*I */
    gsl_vector *at,  /*I/O*/
    gsl_matrix *pt,  /*I/O*/
    double *vt,      /*I/O*/
    double *ft,      /*I/O*/
    gsl_vector *kt,  /*I/O*/
    int m,
    gsl_vector *att)
{
    double finv;
    double tmp;
    double p;
    gsl_vector *ahelp;
    gsl_matrix *mm;

    ahelp = gsl_vector_alloc(m);
    mm = gsl_matrix_alloc(m, m);

    /*kt = zt*pt*/
    gsl_blas_dsymv(CblasUpper, 1.0, pt,
                   zt, 0.0, kt);

    /*ft = kt *ztt + ht*/
    gsl_blas_ddot(zt, kt, &p);
    *ft = p + *ht;

    gsl_blas_ddot(zt, at, &tmp);
    *vt = yt - tmp;

    if (*ft > KFAS_TOL)
    {
        finv = 1.0 / *ft;
        gsl_blas_daxpy((*vt) * finv, kt, at);
        gsl_blas_dsyr(CblasUpper, -finv,
                      kt, pt);
        // Joseph's form covariance update
        // pt = [I - k * z] * pt * [I - k * z]' + k * HT\ * kt
    }
    else
    {
        *ft = 0.0;
    }

    gsl_vector_memcpy(att, at);

    gsl_blas_dgemv(CblasNoTrans, 1.0, tt, at, 0.0, ahelp);

    gsl_vector_memcpy(at, ahelp);

    /* mm = tt*pt*/
    gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, pt, tt, 0.0, mm);
    /* pt = mm * tt^T */
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mm, tt, 0.0, pt);

    gsl_matrix_add(pt, rqr);

    //     // force to be non negative
    //    for(k1 = 0; k1 < m; k1++){
    //        for(k2 = 0; k2 < m; k2++){
    //            if (gsl_matrix_get(pt, k1, k2) < 0){
    //                gsl_matrix_set(pt, k1, k2, 0.0);
    //            }
    //        }
    //    }
    gsl_vector_free(ahelp);
    gsl_matrix_free(mm);

    return;
}

void dfilter1step_validobs(
    double yt,        /*I */
    gsl_vector *zt,   /*I */
    double ht,        /*I */
    gsl_matrix *tt,   /*I */
    gsl_matrix *rqr,  /*I */
    gsl_vector *at,   /*I/O*/
    gsl_matrix *pt,   /*I/O*/
    double *vt,       /*I/O*/
    double *ft,       /*I/O*/
    gsl_vector *kt,   /*I/O*/
    gsl_matrix *pinf, /*I/O*/
    double *finf,     /*I/O*/
    gsl_vector *kinf, /*I/O*/
    int *rankp,       /*I/O*/
    int m,
    bool fast_mode)
{
    double finv;
    double tmp;
    double p;
    gsl_vector *ahelp;
    gsl_matrix *mm;

    ahelp = gsl_vector_alloc(m);
    mm = gsl_matrix_alloc(m, m);

    /*kt = zt*pt*/
    gsl_blas_dsymv(CblasUpper, 1.0, pt,
                   zt, 0.0, kt);

    /*ft = kt *ztt + ht*/
    gsl_blas_ddot(zt, kt, &p);
    *ft = p + ht;

    /*finf = zt*pinf*zt'*/
    gsl_blas_dsymv(CblasUpper, 1.0, pinf,
                   zt, 0.0, kinf);
    gsl_blas_ddot(zt, kinf, finf);

    gsl_blas_ddot(zt, at, &tmp);
    *vt = yt - tmp;

    if (*finf > KFAS_TOL)
    {
        finv = 1.0 / *finf;
        gsl_blas_daxpy((*vt) * finv, kinf, at);
        // at = at + (*vt) * finv * kt;
        gsl_blas_dsyr(CblasUpper, *ft * pow(finv, 2),
                      kinf, pt);
        gsl_blas_dsyr2(CblasUpper, -finv, kt, kinf, pt);
        gsl_blas_dsyr(CblasUpper, -finv, kinf, pinf);
        *rankp = *rankp - 1;
    }
    else
    {
        *finf = 0.0;
        if (*ft > KFAS_TOL)
        {
            finv = 1.0 / *ft;
            gsl_blas_daxpy((*vt) * finv, kt, at);
            gsl_blas_dsyr(CblasUpper, -finv, kt, pt);
        }
    }

    if (*ft < KFAS_TOL)
        *ft = 0.0;

    if (*rankp == 0)
    {
        gsl_vector_free(ahelp);
        gsl_matrix_free(mm);
        return;
    }

    /* ahelp = 1.0 * tt* at */
    gsl_blas_dgemv(CblasNoTrans, 1.0, tt, at, 0.0, ahelp);

    gsl_vector_memcpy(at, ahelp);

    /* mm = tt*pt*/
    gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, pt, tt, 0.0, mm);

    /* pt = mm * tt^T */
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mm, tt, 0.0, pt);
    gsl_matrix_add(pt, rqr);

    /* mm = pinf*tt */
    gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, pinf, tt, 0.0, mm);

    /* pinf = mm * tt^T */
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mm, tt, 0.0, pinf);

    gsl_vector_free(ahelp);
    gsl_matrix_free(mm);

    return;
}

void dfilter1step_missingobs(
    gsl_vector *zt,   /*I */
    double ht,        /*I */
    gsl_matrix *tt,   /*I */
    gsl_matrix *rqr,  /*I */
    gsl_vector *at,   /*I/O: */
    double *ft,       /*O*/
    gsl_matrix *pt,   /*I/O*/
    gsl_vector *kt,   /*I/O*/
    double *finf,     /*I/O*/
    gsl_matrix *pinf, /*I/O*/
    gsl_vector *kinf, /*I/O*/
    int m,            /*I*/
    bool fast_mode)
{
    gsl_vector *ahelp;
    gsl_matrix *mm;

    ahelp = gsl_vector_alloc(m);
    mm = gsl_matrix_alloc(m, m);
    // memcpy(zt_sub, &zt[1], 2*sizeof(*a));

    if (fast_mode == FALSE)
    {
        /* at = 1.0 * tt* at */
        gsl_blas_dgemv(CblasNoTrans, 1.0, tt, at, 0.0, ahelp);

        gsl_vector_memcpy(at, ahelp);
    }

    /* mm = tt*pt*/
    gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, pt, tt, 0.0, mm);

    /* pt = mm * tt^T */
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mm, tt, 0.0, pt);

    gsl_matrix_add(pt, rqr);

    /* mm = pinf*tt */
    gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, pinf, tt, 0.0, mm);

    /* pinf = mm * tt^T */
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mm, tt, 0.0, pinf);

    gsl_vector_free(ahelp);
    gsl_matrix_free(mm);
    return;
}

// convert harmonic coefficients to state value aat
void fit_cft2vec_a(
    float *fit_cft,     /* I: harmonic coefficients  */
    gsl_vector *next_a, /* I: state values    */
    int cur_date,       /* I: current date          */
    int m,              /* I: the number of states   */
    int structure       /*I: structure indicatore */
)
{
    double w = TWO_PI / AVE_DAYS_IN_A_YEAR;
    int count_m = 2;
    gsl_vector_set(next_a, 0, fit_cft[0] + cur_date * (double)fit_cft[1] / SLOPE_SCALE);
    gsl_vector_set(next_a, 1, (double)fit_cft[1] / SLOPE_SCALE);
    if (count_m < m)
    {
        if (structure % 10 == 1)
        {
            gsl_vector_set(next_a, count_m, (double)(fit_cft[2] * cos((double)cur_date * w) + fit_cft[3] * sin((double)cur_date * w)));
            gsl_vector_set(next_a, count_m + 1, (double)(-fit_cft[2] * sin((double)cur_date * w) + fit_cft[3] * cos((double)cur_date * w)));
            count_m = count_m + 2;
        }
    }

    if (count_m < m)
    {
        if (structure / 10 == 1)
        {
            gsl_vector_set(next_a, count_m, (double)(fit_cft[4] * cos((double)cur_date * 2 * w) + fit_cft[5] * sin((double)cur_date * 2 * w)));
            gsl_vector_set(next_a, count_m + 1, (double)(-fit_cft[4] * sin((double)cur_date * 2 * w) + fit_cft[5] * cos((double)cur_date * 2 * w)));
            count_m = count_m + 2;
        }
    }

    if (count_m < m)
    {
        if (structure / 100 == 1)
        {
            gsl_vector_set(next_a, count_m, (double)(fit_cft[6] * cos((double)cur_date * 3 * w) + fit_cft[7] * sin((double)cur_date * 3 * w)));
            gsl_vector_set(next_a, count_m + 1, (double)(-fit_cft[6] * sin((double)cur_date * 3 * w) + fit_cft[7] * cos((double)cur_date * 3 * w)));
            count_m = count_m + 2;
        }
    }
}

/*****************************************************************
 *      convert a to harmonic coefficients
 *****************************************************************/
void vec_a2fit_cft(
    gsl_vector *next_a,
    float *fit_cft,
    int cur_date,
    int m,
    int structure)
{
    double w = TWO_PI / AVE_DAYS_IN_A_YEAR;
    int count_m = 2;
    int i;
    for (i = 0; i < SCCD_MAX_NUM_C; i++)
        fit_cft[i] = 0;

    fit_cft[0] = gsl_vector_get(next_a, 0) - gsl_vector_get(next_a, 1) * cur_date;
    // no slope scenario: m= 2, 4, 6, 8
    fit_cft[1] = (float)(gsl_vector_get(next_a, 1) * SLOPE_SCALE);

    if (count_m < m)
    {
        if (structure % 10 == 1)
        {
            fit_cft[2] = cos((double)cur_date * w) * gsl_vector_get(next_a, count_m) -
                         sin((double)cur_date * w) * gsl_vector_get(next_a, count_m + 1);
            fit_cft[3] = cos((double)cur_date * w) * gsl_vector_get(next_a, count_m + 1) +
                         sin((double)cur_date * w) * gsl_vector_get(next_a, count_m);
            count_m = count_m + 2;
        }
    }

    if (count_m < m)
    {
        if (structure / 10 == 1)
        {
            fit_cft[4] = cos((double)cur_date * 2 * w) * gsl_vector_get(next_a, count_m) -
                         sin((double)cur_date * 2 * w) * gsl_vector_get(next_a, count_m + 1);
            fit_cft[5] = cos((double)cur_date * 2 * w) * gsl_vector_get(next_a, count_m + 1) +
                         sin((double)cur_date * 2 * w) * gsl_vector_get(next_a, count_m);
            count_m = count_m + 2;
        }
    }

    if (count_m < m)
    {
        if (structure / 100 == 1)
        {
            fit_cft[6] = cos((double)cur_date * 3 * w) * gsl_vector_get(next_a, count_m) -
                         sin((double)cur_date * 3 * w) * gsl_vector_get(next_a, count_m + 1);
            fit_cft[7] = cos((double)cur_date * 3 * w) * gsl_vector_get(next_a, count_m + 1) +
                         sin((double)cur_date * 3 * w) * gsl_vector_get(next_a, count_m);
            count_m = count_m + 2;
        }
    }
    // printf("cos((double)cur_date * 2 * w) = %f\n", cos((double)cur_date * 2 * w));
}

/******************************************************************************
Date        Programmer       Reason
--------    ---------------  -------------------------------------
02/14/2021   Su Ye           create elements of ssm instance and give them default values
******************************************************************************/
int initialize_ssmconstants(
    int n_state,
    float rmse,
    double base_value,
    ssmodel_constants *instance)
{
    int i, j;
    instance->m = n_state;

    // it is a three-digit indicator, 11 meaning 'semi + annual cycle'
    instance->structure = 11;

    /* alloc memory for each element*/

    /*
               level      trend      cycle     cycle*       cycle     cycle*      cycle     cycle*
        level      1       1      0.00000000 0.00000000  0.00000000 0.00000000  0.00000000  0.00000000
        trend      0       1      0.00000000 0.00000000  0.00000000 0.00000000  0.00000000  0.00000000
        cycle      0       0      0.99985204 0.01720158  0.00000000 0.00000000  0.00000000  0.00000000
        cycle*     0       0     -0.01720158 0.99985204  0.00000000 0.00000000  0.00000000  0.00000000
        cycle      0       0      0.00000000 0.00000000  0.99940821 0.03439806  0.00000000  0.00000000
        cycle*     0       0      0.00000000 0.00000000 -0.03439806 0.99940821  0.00000000  0.00000000
        cycle      0       0      0.00000000 0.00000000  0.00000000 0.00000000  0.99866864  0.05158437
        cycle*     0       0      0.00000000 0.00000000  0.00000000 0.00000000 -0.05158437  0.99866864

     */
    /* initialize t */
    for (i = 0; i < instance->m; i++)
    {
        for (j = 0; j < instance->m; j++)
        {
            if ((i == 0) && (j == 0))
            {
                gsl_matrix_set(instance->T, i, j, 1.0);
                continue;
            }

            if ((i == 0) && (j == 1))
            {
                gsl_matrix_set(instance->T, i, j, 1.0);
                continue;
            }

            if ((i == 1) && (j == 1))
            {
                gsl_matrix_set(instance->T, i, j, 1.0);
                continue;
            }

            if ((i == 2) && (j == 2))
            {
                gsl_matrix_set(instance->T, i, j, cos((double)TWO_PI / (double)NUM_YEARS));
                continue;
            }

            if ((i == 2) && (j == 3))
            {
                gsl_matrix_set(instance->T, i, j, sin((double)TWO_PI / (double)NUM_YEARS));
                continue;
            }

            if ((i == 3) && (j == 3))
            {
                gsl_matrix_set(instance->T, i, j, cos((double)TWO_PI / (double)NUM_YEARS));
                continue;
            }

            if ((i == 3) && (j == 2))
            {
                gsl_matrix_set(instance->T, i, j, -sin((double)TWO_PI / (double)NUM_YEARS));
                continue;
            }

            if ((i == 4) && (j == 4))
            {
                gsl_matrix_set(instance->T, i, j, cos((double)TWO_PI / (double)NUM_YEARS * 2.0));
                continue;
            }

            if ((i == 5) && (j == 5))
            {
                gsl_matrix_set(instance->T, i, j, cos((double)TWO_PI / (double)NUM_YEARS * 2.0));
                continue;
            }

            if ((i == 4) && (j == 5))
            {
                gsl_matrix_set(instance->T, i, j, sin((double)TWO_PI / (double)NUM_YEARS * 2.0));
                continue;
            }

            if ((i == 5) && (j == 4))
            {
                gsl_matrix_set(instance->T, i, j, -sin((double)TWO_PI / (double)NUM_YEARS * 2.0));
                continue;
            }

            if (instance->m == 8)
            {
                if ((i == 6) && (j == 6))
                {
                    gsl_matrix_set(instance->T, i, j, cos((double)TWO_PI / (double)NUM_YEARS * 3.0));
                    continue;
                }

                if ((i == 7) && (j == 7))
                {
                    gsl_matrix_set(instance->T, i, j, cos((double)TWO_PI / (double)NUM_YEARS * 3.0));
                    continue;
                }

                if ((i == 6) && (j == 7))
                {
                    gsl_matrix_set(instance->T, i, j, sin((double)TWO_PI / (double)NUM_YEARS * 3.0));
                    continue;
                }

                if ((i == 7) && (j == 6))
                {
                    gsl_matrix_set(instance->T, i, j, -sin((double)TWO_PI / (double)NUM_YEARS * 3.0));
                    continue;
                }
            }
        }
    }

    /*   initialize Z     */
    if (instance->m == 6) // the default
    {
        gsl_vector_set(instance->Z, 0, 1.0);
        gsl_vector_set(instance->Z, 1, 0.0);
        gsl_vector_set(instance->Z, 2, 1.0);
        gsl_vector_set(instance->Z, 3, 0.0);
        gsl_vector_set(instance->Z, 4, 1.0);
        gsl_vector_set(instance->Z, 5, 0.0);
    }
    else if (instance->m == 1)
    {
        /*   initialize Z     */
        gsl_vector_set(instance->Z, 0, 1.0);
    }
    else if (instance->m == 3)
    {
        /*   initialize Z     */
        gsl_vector_set(instance->Z, 0, 1.0);
        gsl_vector_set(instance->Z, 1, 1.0);
        gsl_vector_set(instance->Z, 2, 0.0);
    }
    else if (instance->m == 5)
    {
        /*   initialize Z     */
        gsl_vector_set(instance->Z, 0, 1.0);
        gsl_vector_set(instance->Z, 1, 1.0);
        gsl_vector_set(instance->Z, 2, 0.0);
        gsl_vector_set(instance->Z, 3, 1.0);
        gsl_vector_set(instance->Z, 4, 0.0);
    }
    else if (instance->m == 7)
    {
        /*   initialize Z     */
        gsl_vector_set(instance->Z, 0, 1.0);
        gsl_vector_set(instance->Z, 1, 1.0);
        gsl_vector_set(instance->Z, 2, 0.0);
        gsl_vector_set(instance->Z, 3, 1.0);
        gsl_vector_set(instance->Z, 4, 0.0);
        gsl_vector_set(instance->Z, 5, 1.0);
        gsl_vector_set(instance->Z, 6, 0.0);
    }

    /*   initialize Q     */
    // double ini_q00 = pow(base_value, 2) / 10000.0 / AVE_DAYS_IN_A_YEAR;
    double ini_q00 = INI_Q00;
    for (i = 0; i < instance->m; i++)
        if (i == 1)
            gsl_matrix_set(instance->Q, i, i, ini_q00 / SLOPE_SS_SCALE);
        else
            gsl_matrix_set(instance->Q, i, i, ini_q00);

    instance->H = rmse;
    return SUCCESS;
}

/***********************************************************
 * calculate initial p based on at
 * *********************************************************/
float caculate_ini_p(
    int m,
    gsl_vector *ini_a,
    gsl_vector *z)
{
    /* initialize p based on a intensity*/
    double a_intensity;
    double z_sum = 0;
    int k;
    if (m == 1)
        a_intensity = gsl_vector_get(ini_a, 0);
    else if (m == 3)
        a_intensity = gsl_vector_get(ini_a, 0) + gsl_vector_get(ini_a, 1);
    else if (m == 5)
        a_intensity = gsl_vector_get(ini_a, 0) + gsl_vector_get(ini_a, 1) + gsl_vector_get(ini_a, 3);
    else if (m == 7)
        a_intensity = gsl_vector_get(ini_a, 0) + gsl_vector_get(ini_a, 1) + gsl_vector_get(ini_a, 3) + gsl_vector_get(ini_a, 5);
    else if (m == 6)
        a_intensity = gsl_vector_get(ini_a, 0) + gsl_vector_get(ini_a, 2) + gsl_vector_get(ini_a, 4);
    else if (m == 8)
        a_intensity = gsl_vector_get(ini_a, 0) + gsl_vector_get(ini_a, 2) + gsl_vector_get(ini_a, 4) + gsl_vector_get(ini_a, 6);

    // calculate z_sum
    for (k = 0; k < m; k++)
        z_sum = z_sum + gsl_vector_get(z, k);

    return pow((float)a_intensity, 2) / z_sum;
}

double compute_f(
    gsl_matrix *P,
    ssmodel_constants instance)
{
    gsl_vector *kt_tmp;
    double ft_tmp;
    kt_tmp = gsl_vector_alloc(instance.m);
    gsl_blas_dsymv(CblasUpper, 1.0, P,
                   instance.Z, 0.0, kt_tmp);

    /* ft = kt *ztt + ht */
    gsl_blas_ddot(instance.Z, kt_tmp, &ft_tmp);
    ft_tmp = ft_tmp + instance.H;
    gsl_vector_free(kt_tmp);
    return ft_tmp;
}

int KF_ts_predict_conse(
    ssmodel_constants *instance, /* i: the inputted ssm instance   */
    int *clrx,                   /* i: the inputted dates   */
    gsl_matrix *P_ini,           /* i: a m x m matrix of the covariance matrix for pred_start */
    float **fit_cft,             /* i: a m vector of a for pred_start */
    int pred_start,              /* i: close, included for prediction */
    int pred_end,                /* i: close, included for prediction */
    int i_b,
    int cur_i,
    float *pred_y,   /* O: the predicted obs values */
    float *pred_y_f, /*O: the predicted f (RMSE) values */
    bool b_foutput)
{
    int i;
    gsl_vector *kt;
    gsl_vector *kt_tmp;
    double ft_tmp;
    //    gsl_matrix* pt;         /*  A m x m matrix containing the covariance matrix for last_pred_loc */
    //    gsl_vector* at;
    gsl_matrix *mm;
    if (b_foutput == TRUE)
    {
        mm = gsl_matrix_alloc(instance->m, instance->m);
        kt = gsl_vector_alloc(instance->m);
        kt_tmp = gsl_vector_alloc(instance->m);
    }
    double w = TWO_PI / AVE_DAYS_IN_A_YEAR;
    double w2 = 2 * w;

    /* make copy so that recursion won't really change values*/
    //    gsl_matrix_memcpy(pt, P_ini);
    //    gsl_vector_memcpy(at, at_ini);

    /* the predict from clrx[pred_start] to clrx[pred_end] as if they are all missing obs */
    for (i = pred_start; i < pred_end + 1; i++)
    {
        /* this loop predicts every values between two observation dates */
        //        for (j = 0; j < clrx[i + 1] - clrx[i]; j++) /* predict ith observation */
        //        {
        //            /**for observation date, we need make predication
        //             * but without updating KF parameters */
        //            if (j == 0)
        //            {
        /* predict y without updating */
        // gsl_blas_ddot(instance->Z, at_ini, &yt_tmp);
        //                printf("fit_cft[0][0]: %f\n", fit_cft[i_b][0]);
        //                printf("fit_cft[0][1]: %f\n", fit_cft[i_b][1]);
        //                printf("fit_cft[0][2]: %f\n", fit_cft[i_b][2]);
        //                printf("fit_cft[0][3]: %f\n", fit_cft[i_b][3]);
        //                printf("fit_cft[0][4]: %f\n", fit_cft[i_b][4]);
        //                printf("fit_cft[0][5]: %f\n", fit_cft[i_b][5]);
        pred_y[i - pred_start] = fit_cft[i_b][0] + fit_cft[i_b][1] * (float)clrx[i] / SLOPE_SCALE + fit_cft[i_b][2] * cos((float)clrx[i] * w) + fit_cft[i_b][3] * sin((float)clrx[i] * w) + fit_cft[i_b][4] * cos((float)clrx[i] * w2) + fit_cft[i_b][5] * sin((float)clrx[i] * w2);
        if (b_foutput == TRUE)
        {
            /* kt = pt*zt */
            gsl_blas_dsymv(CblasUpper, 1.0, P_ini,
                           instance->Z, 0.0, kt_tmp);

            /* ft = kt *ztt + ht */
            gsl_blas_ddot(instance->Z, kt_tmp, &ft_tmp);
            ft_tmp = ft_tmp + instance->H;

            pred_y_f[i - pred_start] = (float)ft_tmp;
        }
        else
            pred_y_f[i - pred_start] = 0;

        // printf("ft for %d is %f: \n", clrx[i] + j, ft_tmp);

        /* update fit_cft using new at*/
        //                if (b_fastmode == TRUE)
        //                    vec_a2fit_cft(at_ini, fit_cft, clrx[i]);
        //            }

        if (b_foutput == TRUE)
        {
            filter1step_missingobs(instance->Z, instance->H, instance->T,
                                   instance->Q, &ft_tmp, P_ini, kt, instance->m);

            //                gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, P_ini, instance->T, 0.0, mm);

            //                /* pt = mm * tt^T */
            //                gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, P_ini, instance->T, 0.0, P_ini);
        }

        // printf("ft for %d is %f: \n", clrx[i] + j, ft_tmp);

        //        }
    }

    if (b_foutput == TRUE)
    {
        gsl_vector_free(kt);
        //    gsl_matrix_free(pt);
        //    gsl_vector_free(at);
        gsl_vector_free(kt_tmp);
        gsl_matrix_free(mm);
    }
    return SUCCESS;
}

void filter1step_missingobs_fast(
    gsl_vector *zt,  /*I */
    double ht,       /*I */
    gsl_matrix *rqr, /*I */
    double *ft,      /*O*/
    gsl_matrix *pt,  /*I/O*/
    gsl_vector *kt,  /*I/O*/
    int m,           /*I*/
    double gap_days)
{
    gsl_matrix *mm;
    gsl_matrix *tt; /*I */
    gsl_matrix *mrqr;
    int i, j;
    tt = gsl_matrix_calloc(DEFAULT_N_STATE, DEFAULT_N_STATE);
    mrqr = gsl_matrix_calloc(DEFAULT_N_STATE, DEFAULT_N_STATE);
    mm = gsl_matrix_alloc(m, m);
    // memcpy(zt_sub, &zt[1], 2*sizeof(*a));
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < m; j++)
        {
            if ((i == 0) && (j == 0))
            {
                gsl_matrix_set(tt, i, j, 1.0);
                continue;
            }

            if ((i == 0) && (j == 1))
            {
                gsl_matrix_set(tt, i, j, 1.0 * gap_days);
                continue;
            }

            if ((i == 1) && (j == 1))
            {
                gsl_matrix_set(tt, i, j, 1.0);
                continue;
            }

            if ((i == 2) && (j == 2))
            {
                gsl_matrix_set(tt, i, j, cos((double)TWO_PI * gap_days / (double)NUM_YEARS));
                continue;
            }

            if ((i == 2) && (j == 3))
            {
                gsl_matrix_set(tt, i, j, sin((double)TWO_PI * gap_days / (double)NUM_YEARS));
                continue;
            }

            if ((i == 3) && (j == 3))
            {
                gsl_matrix_set(tt, i, j, cos((double)TWO_PI * gap_days / (double)NUM_YEARS));
                continue;
            }

            if ((i == 3) && (j == 2))
            {
                gsl_matrix_set(tt, i, j, -sin((double)TWO_PI * gap_days / (double)NUM_YEARS));
                continue;
            }

            if ((i == 4) && (j == 4))
            {
                gsl_matrix_set(tt, i, j, cos((double)TWO_PI * gap_days / (double)NUM_YEARS * 2.0));
                continue;
            }

            if ((i == 5) && (j == 5))
            {
                gsl_matrix_set(tt, i, j, cos((double)TWO_PI * gap_days / (double)NUM_YEARS * 2.0));
                continue;
            }

            if ((i == 4) && (j == 5))
            {
                gsl_matrix_set(tt, i, j, sin((double)TWO_PI * gap_days / (double)NUM_YEARS * 2.0));
                continue;
            }

            if ((i == 5) && (j == 4))
            {
                gsl_matrix_set(tt, i, j, -sin((double)TWO_PI * gap_days / (double)NUM_YEARS * 2.0));
                continue;
            }
        }
    }

    // gsl_blas_dscal(gap_days, mrqr);
    for (i = 0; i < m; i++)
        gsl_matrix_set(mrqr, i, i, gsl_matrix_get(rqr, i, i) * gap_days);

    // gsl_blas_daxpy(gap_days, rqr, mrqr);
    /* mm = tt*pt*/
    gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, pt, tt, 0.0, mm);

    /* pt = mm * tt^T */
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mm, tt, 0.0, pt);
    gsl_matrix_add(pt, mrqr);

    gsl_matrix_free(mm);
    gsl_matrix_free(tt);
    gsl_matrix_free(mrqr);

    /* force negative to be zero */
    //    for(k1 = 0; k1 < m; k1++){
    //        for(k2 = 0; k2 < m; k2++){
    //            if (gsl_matrix_get(pt, k1, k2) < 0){
    //                gsl_matrix_set(pt, k1, k2, 0.0);
    //            }
    //        }
    //    }
    return;
}

int KF_ts_filter_regular(
    ssmodel_constants *instance, /* i: the inputted ssm instance   */
    int *clrx,                   /* i: the inputted dates   */
    float *clry,                 /* i: the inputted observations   */
    gsl_matrix *cov_p,           /* i/O: m x m matrix of the covariance matrix for pred_start */
    float **fit_cft,             /* i/O: m vector of a for pred_start */
    int cur_i,                   /* i: the current i */
    int i_b,                     /* i: the band number */
    double *vt,                  /* I/O: predicted residuals */
    bool steady)
{
    int j;
    gsl_vector *kt;
    gsl_vector *att;
    gsl_vector *state_a; // filtered states
    // double vt;
    double ft;

    kt = gsl_vector_alloc(instance->m);
    att = gsl_vector_alloc(instance->m);
    state_a = gsl_vector_alloc(instance->m);
    /* this loop predicts every missing values between two observation dates */
    // for (j = 0; j < clrx[cur_i + 1] - clrx[cur_i]; j++) /* predict ith observation */
    // {
    //     if (0 == j)
    //     {
    //         /* first get at from fit_cft*/
    //         fit_cft2vec_a(fit_cft[i_b], state_a, clrx[cur_i], instance->m, instance->structure);
    //         //            if(i_b == 3)
    //         //                printf("i = %d: %f, %f, %f, %f\n", clrx[cur_i], gsl_vector_get(state_a, 0), gsl_vector_get(state_a, 1),
    //         //                       gsl_vector_get(state_a, 2), gsl_vector_get(state_a, 3));
    //         //            if(i_b == 3)
    //         //                printf("i = %d: %f, %f, %f, %f\n", clrx[cur_i], fit_cft[i_b][0], fit_cft[i_b][1],
    //         //                       fit_cft[i_b][2], fit_cft[i_b][3]);
    //         /* predicts valid obs values between two observation dates */
    //         filter1step_validobs(clry[cur_i], instance->Z, &instance->H, instance->T, instance->Q,
    //                              state_a, cov_p, vt, &ft, kt, instance->m, att);
    //         /* update fit_cft using new at*/
    //         vec_a2fit_cft(state_a, fit_cft[i_b], clrx[cur_i] + 1, instance->m, instance->structure);
    //         //            if(i_b == 3)
    //         //                printf("i = %d: %f, %f, %f, %f\n", clrx[cur_i],fit_cft[i_b][0], fit_cft[i_b][1],
    //         //                        fit_cft[i_b][2], fit_cft[i_b][3]);
    //         // printf("rmse for %d time point for band %d: %f\n", cur_i, i_band, rmse[i_b][cur_i]);
    //     }
    //     else
    //     {
    //         if (FALSE == steady)
    //         {
    //             filter1step_missingobs(instance->Z, instance->H, instance->T, instance->Q,
    //                                    &ft, cov_p, kt, instance->m);
    //         }
    //     }
    //     // printf("ft for %d is %f: \n", clrx[cur_i] + j, ft);
    // }
    fit_cft2vec_a(fit_cft[i_b], state_a, clrx[cur_i], instance->m, instance->structure);
    /* predicts valid obs values between two observation dates */
    filter1step_validobs(clry[cur_i], instance->Z, &instance->H, instance->T, instance->Q,
                         state_a, cov_p, vt, &ft, kt, instance->m, att);
    /* update fit_cft using new at*/
    vec_a2fit_cft(state_a, fit_cft[i_b], clrx[cur_i] + 1, instance->m, instance->structure);

    filter1step_missingobs_fast(instance->Z, instance->H, instance->Q, &ft, cov_p, kt, instance->m, (double)(clrx[cur_i + 1] - clrx[cur_i] - 1));

    gsl_vector_free(kt);
    gsl_vector_free(att);
    gsl_vector_free(state_a);

    return SUCCESS;
}

int KF_ts_filter_falsechange(
    ssmodel_constants *instance, /* i: the inputted ssm instance   */
    int *clrx,                   /* i: the inputted dates   */
    gsl_matrix *cov_p,           /* i/O: a m x m matrix of the covariance matrix for pred_start */
    int cur_i)
{
    int j;
    gsl_vector *kt;
    double ft;

    kt = gsl_vector_alloc(instance->m);

    /* this loop predicts every missing values between two observation dates */
    // for (j = 0; j < clrx[cur_i + 1] - clrx[cur_i]; j++) /* predict ith observation */
    // {
    //     filter1step_missingobs(instance->Z, instance->H, instance->T, instance->Q,
    //                            &ft, cov_p, kt, instance->m);
    // }
    filter1step_missingobs_fast(instance->Z, instance->H, instance->Q, &ft, cov_p, kt, instance->m, (double)(clrx[cur_i + 1] - clrx[cur_i]));
    gsl_vector_free(kt);

    return SUCCESS;
}

// void vmmin(int n0, double *b, double *Fmin, optimfn fminfn, optimgr fmingr,
//       int maxit, int trace, int *mask,
//       double abstol, double reltol, int nREPORT, void *ex,
//       int *fncount, int *grcount, int *fail)
//{
//     bool accpoint, enough;
//     double *g, *t, *X, *c, **B;
//     int   count, funcount, gradcount;
//     double f, gradproj;
//     int   i, j, ilast, iter = 0;
//     double s, steplength;
//     double D1, D2;
//     int   n, *l;

//    if (maxit <= 0) {
//    *fail = 0;
//    *Fmin = fminfn(n0, b, ex);
//    *fncount = *grcount = 0;
//    return;
//    }

//    if (nREPORT <= 0)
//    error(_("REPORT must be > 0 (method = \"BFGS\")"));
//    l = (int *) R_alloc(n0, sizeof(int));
//    n = 0;
//    for (i = 0; i < n0; i++) if (mask[i]) l[n++] = i;
//    g = vect(n0);
//    t = vect(n);
//    X = vect(n);
//    c = vect(n);
//    B = Lmatrix(n);
//    f = fminfn(n0, b, ex);
//    if (!R_FINITE(f))
//    error(_("initial value in 'vmmin' is not finite"));
//    if (trace) Rprintf("initial  value %f \n", f);
//    *Fmin = f;
//    funcount = gradcount = 1;
//    fmingr(n0, b, g, ex);
//    iter++;
//    ilast = gradcount;

//    do {
//    if (ilast == gradcount) {
//        for (i = 0; i < n; i++) {
//        for (j = 0; j < i; j++) B[i][j] = 0.0;
//        B[i][i] = 1.0;
//        }
//    }
//    for (i = 0; i < n; i++) {
//        X[i] = b[l[i]];
//        c[i] = g[l[i]];
//    }
//    gradproj = 0.0;
//    for (i = 0; i < n; i++) {
//        s = 0.0;
//        for (j = 0; j <= i; j++) s -= B[i][j] * g[l[j]];
//        for (j = i + 1; j < n; j++) s -= B[j][i] * g[l[j]];
//        t[i] = s;
//        gradproj += s * g[l[i]];
//    }

//    if (gradproj < 0.0) {	/* search direction is downhill */
//        steplength = 1.0;
//        accpoint = FALSE;
//        do {
//        count = 0;
//        for (i = 0; i < n; i++) {
//            b[l[i]] = X[i] + steplength * t[i];
//            if (reltest + X[i] == reltest + b[l[i]]) /* no change */
//            count++;
//        }
//        if (count < n) {
//            f = fminfn(n0, b, ex);
//            funcount++;
//            accpoint = R_FINITE(f) &&
//            (f <= *Fmin + gradproj * steplength * acctol);
//            if (!accpoint) {
//            steplength *= stepredn;
//            }
//        }
//        } while (!(count == n || accpoint));
//        enough = (f > abstol) &&
//        fabs(f - *Fmin) > reltol * (fabs(*Fmin) + reltol);
//        /* stop if value if small or if relative change is low */
//        if (!enough) {
//        count = n;
//        *Fmin = f;
//        }
//        if (count < n) {/* making progress */
//        *Fmin = f;
//        fmingr(n0, b, g, ex);
//        gradcount++;
//        iter++;
//        D1 = 0.0;
//        for (i = 0; i < n; i++) {
//            t[i] = steplength * t[i];
//            c[i] = g[l[i]] - c[i];
//            D1 += t[i] * c[i];
//        }
//        if (D1 > 0) {
//            D2 = 0.0;
//            for (i = 0; i < n; i++) {
//            s = 0.0;
//            for (j = 0; j <= i; j++)
//                s += B[i][j] * c[j];
//            for (j = i + 1; j < n; j++)
//                s += B[j][i] * c[j];
//            X[i] = s;
//            D2 += s * c[i];
//            }
//            D2 = 1.0 + D2 / D1;
//            for (i = 0; i < n; i++) {
//            for (j = 0; j <= i; j++)
//                B[i][j] += (D2 * t[i] * t[j]
//                    - X[i] * t[j] - t[i] * X[j]) / D1;
//            }
//        } else {	/* D1 < 0 */
//            ilast = gradcount;
//        }
//        } else {	/* no progress */
//        if (ilast < gradcount) {
//            count = 0;
//            ilast = gradcount;
//        }
//        }
//    } else {		/* uphill search */
//        count = 0;
//        if (ilast == gradcount) count = n;
//        else ilast = gradcount;
//        /* Resets unless has just been reset */
//    }
//    if (trace && (iter % nREPORT == 0))
//        Rprintf("iter%4d value %f\n", iter, f);
//    if (iter >= maxit) break;
//    if (gradcount - ilast > 2 * n)
//        ilast = gradcount;	/* periodic restart */
//    } while (count != n || ilast != gradcount);
//    if (trace) {
//    Rprintf("final  value %f \n", *Fmin);
//    if (iter < maxit) Rprintf("converged\n");
//    else Rprintf("stopped after %i iterations\n", iter);
//    }
//    *fail = (iter < maxit) ? 0 : 1;
//    *fncount = funcount;
//    *grcount = gradcount;
//}
