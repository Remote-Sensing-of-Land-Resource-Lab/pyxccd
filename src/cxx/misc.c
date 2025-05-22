#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <dirent.h>
// #include <fnmatch.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include "2d_array.h"
#include "const.h"
#include "utilities.h"
#include "defines.h"
#include "cold.h"
#include "misc.h"

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_fit.h>

// Externally defined routines
double Chi_Square_Distribution(double x, int n);
double Chi_Square_Density(double x, int n);

/******************************************************************************
MODULE:  update_cft

PURPOSE:  determine the number of coefficient use in the time series model

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/23/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void update_cft(
    int i_span,
    int n_times,
    int min_num_c,
    int mid_num_c,
    int max_num_c,
    int num_c,
    int *update_number_c)
{
    /* start with 4 coefficients model */
    if (i_span < mid_num_c * n_times)
    {
        *update_number_c = min(min_num_c, num_c);
    }
    /* start with 6 coefficients model */
    else if (i_span < max_num_c * n_times)
    {
        *update_number_c = min(mid_num_c, num_c);
    }
    /* start with 8 coefficients model */
    else
    {
        *update_number_c = min(max_num_c, num_c);
    }
}

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
)
{
    int i, j, k, s;                               /* loop indecies                                     */
    float *var;                                   /* pointer for allocation variable memory            */
    int dim2_len = dim2_end - dim2_start + 1;     /* perhaps should get defined  */
    char FUNC_NAME[] = "adjust_median_variogram"; /* for error messages             */
    int *unique_diff;                             /* date difference value, used to calculate majority*/
    int *unique_diff_freq;
    int unique_diff_count;
    int majority_date_diff = 0;
    int step;
    float diff;
    int max_freq;
    int max_loc;
    int m;

    //    for (i = 0; i < dim2_len; i++)
    //    {
    //        for (j = 0; j < TOTAL_IMAGE_BANDS; j++)
    //        {
    //           printf("%f\n", (float)array[j][i]);
    //        }

    //    }
    if (dim2_len == 0)
    {
        for (i = 0; i < dim1_len; i++)
        {
            output_array[i] = 0;
            *date_vario = 0;
            return (SUCCESS);
        }
    }
    if (dim2_len == 1)
    {
        for (i = 0; i < dim1_len; i++)
        {
            output_array[i] = array[i][dim2_start];
            *date_vario = clrx[dim2_start];
            return (SUCCESS);
        }
    }

    var = (float *)malloc((dim2_len - 1) * sizeof(float));
    if (var == NULL)
    {
        RETURN_ERROR("Allocating var memory", FUNC_NAME, ERROR);
    }

    for (j = 0; j < dim2_len - 1; j++)
    {
        var[j] = abs(clrx[j + dim2_start + 1] - clrx[j + dim2_start]);

        // printf("%d, %d, %f\n", clrx[j + dim2_start + 1], clrx[j + dim2_start], var[j]);
    }
    quick_sort_float(var, 0, dim2_len - 2);
    m = (dim2_len - 1) / 2;
    if ((dim2_len - 1) % 2 == 0)
        *date_vario = (var[m - 1] + var[m]) / 2.0;
    else
        *date_vario = var[m];

    *max_neighdate_diff = var[dim2_len - 2];

    if (option == 1)
    {

        for (i = 0; i < dim1_len; i++)
        {
            for (j = dim2_start; j < dim2_end; j++)
            {
                var[j - dim2_start] = abs(array[i][j + 1] - array[i][j]);
                // printf("%d var for band %d: %f\n", j, i+1, (float)var[j]);
            }
            quick_sort_float(var, 0, dim2_len - 2);
            //            for (j = 0; j < dim2_end; j++)
            //            {
            //               printf("%f\n", var[j]);
            //            }
            m = (dim2_len - 1) / 2;
            if ((dim2_len - 1) % 2 == 0)
            {
                // printf("%f\n", var[m-1]);
                // printf("%f\n", var[m]);
                output_array[i] = (var[m - 1] + var[m]) / 2.0;
            }
            else
                output_array[i] = var[m];
        }
    }
    else if (option == 2)
    {
        unique_diff = (int *)malloc((dim2_len - 1) * sizeof(int));
        unique_diff_freq = (int *)malloc((dim2_len - 1) * sizeof(int));

        step = 1;

        while (majority_date_diff < 31)
        {
            unique_diff_count = 0;
            majority_date_diff = 0;
            for (k = 0; k < dim2_len - 1; k++)
            {
                unique_diff[k] = 0;
                unique_diff_freq[k] = 0;
            }
            k = dim2_start;
            for (j = dim2_start; j < dim2_end + 1 - step; j++)
            {
                diff = abs(clrx[j + step] - clrx[j]);
                if (unique_diff_count == 0)
                {
                    unique_diff[0] = diff;
                    unique_diff_freq[0]++;
                    unique_diff_count++;
                }
                else
                {
                    for (s = 0; s < unique_diff_count; s++)
                    {
                        if (diff == unique_diff[s])
                        {
                            unique_diff_freq[s]++;
                            break;
                        }
                    }
                    if (s == unique_diff_count)
                    {
                        unique_diff[unique_diff_count] = diff;
                        unique_diff_freq[unique_diff_count]++;
                        unique_diff_count++;
                    }
                }
            }
            //        printf("\n");
            //        for (s = 0; s < unique_diff_count; s++)
            //        {
            //                 printf("%d\n", unique_diff_freq[s]);
            //                 printf("%d\n", unique_diff[s]);
            //        }
            max_array_int(unique_diff_freq, unique_diff_count, &max_freq, &max_loc);
            step++;
            majority_date_diff = unique_diff[max_loc];
        }

        for (i = 0; i < dim1_len; i++)
        {

            /*This method differentiates from the standard calculate_variogram in that
            it attempts to only use observations that are greater than 30 days apart.*/

            // printf("%\n");
            s = 0;
            for (j = dim2_start; j < dim2_end + 1 - step + 1; j++)
            {
                if (clrx[j + step - 1] - clrx[j] > 30)
                {
                    var[s] = (float)abs(array[i][j + step - 1] - array[i][j]);
                    // printf("%d\n", clrx[j+step - 1] - clrx[j]);
                    s++;
                }
                //            printf("%f\n", array[i][j]);
                //            printf("%f\n", array[i][j+1]);
                //            printf("%f\n", var[j]);
            }

            quick_sort_float(var, 0, s - 1);

            m = s / 2;
            if (s % 2 == 0)
            {
                output_array[i] = (var[m - 1] + var[m]) / 2.0;
            }
            else
                output_array[i] = var[m];

            // printf("%d\n", output_array[i]);
        }

        free(unique_diff);
        free(unique_diff_freq);
    }
    else if (option == 3)
    {
        int j_count;
        for (i = 0; i < dim1_len; i++)
        {
            j_count = 0;
            for (j = dim2_start; j < dim2_end; j++)
            {
                if (clrx[j + 1] - clrx[j] < 91)
                {
                    var[j_count] = fabsf(array[i][j + 1] - array[i][j]) / abs(clrx[j + 1] - clrx[j]);
                    j_count++;
                }

                // printf("%d var for band %d: %f\n", j, i+1, (float)var[j]);
            }
            quick_sort_float(var, 0, j_count - 1);
            //            for (j = 0; j < dim2_end; j++)
            //            {
            //               printf("%f\n", var[j]);
            //            }
            m = j_count / 2;
            if (j_count % 2 == 0)
            {
                // printf("%f\n", var[m-1]);
                // printf("%f\n", var[m]);
                output_array[i] = (var[m - 1] + var[m]) / 2.0 * (clrx[dim2_end] - clrx[dim2_start]) / (dim2_len - 1);
            }
            else
                output_array[i] = var[m] * (clrx[dim2_end] - clrx[dim2_start]) / (dim2_len - 1);
        }
    }
    else if (option == 4)
    {
        int j_count;
        for (i = 0; i < dim1_len; i++)
        {
            j_count = 0;
            for (j = dim2_start; j < dim2_end; j++)
            {
                if (clrx[j + 1] - clrx[j] < *date_vario * 3)
                {
                    var[j_count] = fabsf(array[i][j + 1] - array[i][j]);
                    j_count++;
                }

                // printf("%d var for band %d: %f\n", j, i+1, (float)var[j]);
            }
            quick_sort_float(var, 0, j_count - 1);
            //            for (j = 0; j < dim2_end; j++)
            //            {
            //               printf("%f\n", var[j]);
            //            }
            m = j_count / 2;
            if (j_count % 2 == 0)
            {
                // printf("%f\n", var[m-1]);
                // printf("%f\n", var[m]);
                output_array[i] = (var[m - 1] + var[m]) / 2.0;
            }
            else
                output_array[i] = var[m];
        }
    }
    free(var);

    return (SUCCESS);
}

/******************************************************************************
NAME:           split_directory_scenename

PURPOSE:
Split the specified filename into a directory, a root file name, and an
extension.  The last character of the directory path will be a '/'.

RETURN: NONE

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/16/2016   Song Guo         Original development
******************************************************************************/
#include <string.h>
#include <limits.h> /* For PATH_MAX                                */
// #include "error.h"
#include "input.h"
#include "const.h"

void split_directory_scenename(
    const char *filename, /* I: Name of scene with path to split         */
    char *directory,      /* O: Directory portion of file name           */
    char *scene_name      /* O: Scene name portion of the file name      */
)
{
    char file_name[PATH_MAX]; /* Local copy of filename                      */
    char *ptr = NULL;         /* String pointer                              */

    /******************************************************************/
    /*                                                                */
    /* Make a local copy of filename so it is not destroyed */
    /*                                                                */
    /******************************************************************/

    strcpy(file_name, filename);

    /******************************************************************/
    /*                                                                */
    /* Check for a directory path                                     */
    /* Find ending '/'                                                */
    /*                                                                */
    /******************************************************************/

    ptr = (char *)strrchr(file_name, '/');
    if (ptr != NULL)
    {
        *(ptr++) = '\0';
        strcpy(directory, file_name);
    }
    else
        strcpy(directory, "");

    /******************************************************************/
    /*                                                                */
    /* Obtain the scene name                                          */
    /*                                                                */
    /******************************************************************/

    strcpy(scene_name, ptr);

    return;
}

/******************************************************************************
MODULE:  rmse_from_square_root_mean

PURPOSE:  simulate matlab calculate rmse from square root mean

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
6/26/2015   Song Guo         Original Development

******************************************************************************/
void rmse_from_square_root_mean(
    float **array,  /* I: input array */
    float fit_cft,  /* I: input fit_cft value */
    int dim1_index, /* I: dimension 1 index in input array */
    int dim2_len,   /* I: dimension 2 length */
    float *rmse     /* O: output rmse */
)
{
    int i;
    float sum = 0.0;

    for (i = 0; i < dim2_len; i++)
    {
        sum += (array[dim1_index][i] - fit_cft) *
               (array[dim1_index][i] - fit_cft);
    }
    *rmse = sqrt(sum / dim2_len);
}

/******************************************************************************
MODULE:  partition_int

PURPOSE:  partition used for the quick_sort_int routine

RETURN VALUE:
Type = int
Value           Description
-----           -----------
i               partitioned value

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/23/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
int partition_int(int arr[], int left, int right)
{
    int i = left, j = right;
    int tmp;
    int pivot = arr[(left + right) / 2];

    while (i <= j)
    {
        while (arr[i] < pivot)
        {
            i++;
        }
        while (arr[j] > pivot)
        {
            j--;
        }
        if (i <= j)
        {
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    }

    return i;
}

/******************************************************************************
MODULE:  quick_sort_int

PURPOSE:  sort the scene_list based on integer yeardoy

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/23/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void quick_sort_int(int arr[], int left, int right)
{
    int index = partition_int(arr, left, right);

    if (left < index - 1)
    {
        quick_sort_int(arr, left, index - 1);
    }
    if (index < right)
    {
        quick_sort_int(arr, index, right);
    }
}

/******************************************************************************
MODULE:  partition_2d_float

PURPOSE:  partition used for the quick_sort routine

RETURN VALUE:
Type = int
Value           Description
-----           -----------
i               partitioned value

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/21/2016   Song Guo         Original Development

NOTES:
******************************************************************************/
int partition_2d_float(float arr[], float *brr[], int left, int right, int band_num)
{
    int i = left, j = right;
    float tmp;
    float tmp2[TOTAL_IMAGE_BANDS];
    float pivot = arr[(left + right) / 2];
    int b;

    while (i <= j)
    {
        while (arr[i] < pivot)
        {
            i++;
        }
        while (arr[j] > pivot)
        {
            j--;
        }
        if (i <= j)
        {
            tmp = arr[i];
            for (b = 0; b < band_num; b++)
                tmp2[b] = brr[b][i];
            arr[i] = arr[j];
            for (b = 0; b < band_num; b++)
                brr[b][i] = brr[b][j];
            arr[j] = tmp;
            for (b = 0; b < band_num; b++)
                brr[b][j] = tmp2[b];
            i++;
            j--;
        }
    }

    return i;
}

/******************************************************************************
MODULE:  partition_2d_double

PURPOSE:  partition used for the quick_sort routine

RETURN VALUE:
Type = int
Value           Description
-----           -----------
i               partitioned value

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
09/22/2019   Su Ye          Original Development

NOTES:
******************************************************************************/
int partition_2d_double(double arr[], double *brr[], int left, int right, int band_num)
{
    int i = left, j = right;
    double tmp;
    double tmp2[TOTAL_IMAGE_BANDS];
    double pivot = arr[(left + right) / 2];
    int b;

    while (i <= j)
    {
        while (arr[i] < pivot)
        {
            i++;
        }
        while (arr[j] > pivot)
        {
            j--;
        }
        if (i <= j)
        {
            tmp = arr[i];
            for (b = 0; b < band_num; b++)
                tmp2[b] = brr[b][i];
            arr[i] = arr[j];
            for (b = 0; b < band_num; b++)
                brr[b][i] = brr[b][j];
            arr[j] = tmp;
            for (b = 0; b < band_num; b++)
                brr[b][j] = tmp2[b];
            i++;
            j--;
        }
    }

    return i;
}
/******************************************************************************
MODULE:  quick_sort_2d_float

PURPOSE:  sort the scene_list & sdate based on yeardoy string

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/21/2016   Song Guo         Original Development

NOTES:
******************************************************************************/
void quick_sort_2d_float(float arr[], float *brr[], int left, int right, int band_num)
{
    int index = partition_2d_float(arr, brr, left, right, band_num);

    if (left < index - 1)
    {
        quick_sort_2d_float(arr, brr, left, index - 1, band_num);
    }
    if (index < right)
    {
        quick_sort_2d_float(arr, brr, index, right, band_num);
    }
}

/******************************************************************************
MODULE:  quick_sort_2d_double

PURPOSE:  sort the scene_list & sdate based on yeardoy string

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
09/22/2019   Su Ye         Original Development

NOTES:
******************************************************************************/
void quick_sort_2d_double(double arr[], double *brr[], int left, int right, int band_num)
{
    int index = partition_2d_double(arr, brr, left, right, band_num);

    if (left < index - 1)
    {
        quick_sort_2d_double(arr, brr, left, index - 1, band_num);
    }
    if (index < right)
    {
        quick_sort_2d_double(arr, brr, index, right, band_num);
    }
}

/******************************************************************************
MODULE:  partial_square_root_mean

PURPOSE:  simulate square root mean function of paritail of a 2d array

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/9/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void partial_square_root_mean(
    float **array,   /* I: input array */
    int dim1_index,  /* I: 1st dimension index */
    int start,       /* I: number of start elements in 2nd dim */
    int end,         /* I: number of end elements in 2nd dim */
    float **fit_ctf, /* I: */
    float *rmse      /* O: output rmse value */
)
{
    int i;
    float sum = 0;
    float rmse_square;

    for (i = start; i <= end; i++)
    {
        sum += ((array[dim1_index][i] - fit_ctf[dim1_index][0]) *
                (array[dim1_index][i] - fit_ctf[dim1_index][0]));
    }
    rmse_square = sum / (float)(end - start + 1);
    *rmse = sqrt(rmse_square);
}

/******************************************************************************
MODULE:  matlab_2d_array_norm

PURPOSE:  simulate matlab norm function for 1 dimension in 2d array cases only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/9/2015   Song Guo         Original Development

NOTES:
******************************************************************************/

void matlab_2d_array_norm(
    double **array,     /* I: input array                                   */
    int dim1_index,     /* I: 1st dimension index                           */
    int dim2_len,       /* I: number of input elements in 2nd dim           */
    double *output_norm /* O: output norm value                             */
)
{
    int i;
    double sum = 0.0;

    for (i = 0; i < dim2_len; i++)
    {
        sum += array[dim1_index][i] * array[dim1_index][i];
    }
    *output_norm = sqrt(sum);
}

/******************************************************************************
MODULE:  matlab_2d_array_norm

PURPOSE:  simulate matlab norm function for 1 dimension in 2d array cases only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/9/2015   Song Guo         Original Development

NOTES:
******************************************************************************/

void matlab_2d_array_norm_float(
    float **array,     /* I: input array                                   */
    int dim1_index,    /* I: 1st dimension index                           */
    int dim2_len,      /* I: number of input elements in 2nd dim           */
    float *output_norm /* O: output norm value                             */
)
{
    int i;
    float sum = 0.0;

    for (i = 0; i < dim2_len; i++)
    {
        sum += array[dim1_index][i] * array[dim1_index][i];
    }
    *output_norm = sqrtf(sum);
}

void matlab_1d_array_norm(
    float *array,      /* I: input array                                   */
    int dim_len,       /* I: number of input elements in 2nd dim           */
    float *output_norm /* O: output norm value                             */
)
{
    int i;
    float sum = 0.0;

    for (i = 0; i < dim_len; i++)
    {
        sum += array[i] * array[i];
    }
    *output_norm = sqrt(sum);
}

/******************************************************************************
MODULE:  matlab_2d_float_median

PURPOSE:  simulate matlab median function for 1 dimesion in 2d array float point
          number case only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
6/26/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void matlab_2d_float_median(
    float **array,       /* I: input array */
    int dim1_index,      /* I: 1st dimension index */
    int dim2_len,        /* I: number of input elements in 2nd dim */
    float *output_median /* O: output norm value */
)
{
    int m = dim2_len / 2;

    if (dim2_len % 2 == 0)
    {
        *output_median = (array[dim1_index][m - 1] + array[dim1_index][m]) / 2.0;
    }
    else
    {
        *output_median = array[dim1_index][m];
    }
}

/******************************************************************************
MODULE:  matlab_2d_double_median

PURPOSE:  simulate matlab median function for 1 dimesion in 2d array float point
          number case only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
6/26/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void matlab_2d_double_median(
    double **array,       /* I: input array */
    int dim1_index,       /* I: 1st dimension index */
    int dim2_len,         /* I: number of input elements in 2nd dim */
    double *output_median /* O: output norm value */
)
{
    int m = dim2_len / 2;

    if (dim2_len % 2 == 0)
    {
        *output_median = (array[dim1_index][m - 1] + array[dim1_index][m]) / 2.0;
    }
    else
    {
        *output_median = array[dim1_index][m];
    }
}

/******************************************************************************
MODULE:  matlab_2d_array_mean

PURPOSE:  simulate matlab mean function for 1 dimension in 2d array cases only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/9/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void matlab_2d_array_mean(
    float **array,     /* I: input array */
    int dim1_index,    /* I: 1st dimension index */
    int dim2_len,      /* I: number of input elements in 2nd dim */
    float *output_mean /* O: output norm value */
)
{
    int i;
    float sum = 0.0;

    for (i = 0; i < dim2_len; i++)
    {
        sum += array[dim1_index][i];
    }
    *output_mean = sum / (float)(dim2_len);
}

/******************************************************************************
MODULE:  array_1d_mean

PURPOSE:  compute mean for 1 dimension only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/25/2018   Su Ye         Original Development

NOTES:
******************************************************************************/
void array_1d_mean(
    float *array, /* I: input array */
    int start,    /* I: number of input elements in 2nd dim */
    int end,
    float *output_mean /* O: output norm value */
)
{
    int i;
    double sum = 0.0;

    for (i = start; i < end + 1; i++)
    {
        // printf("%d, %f\n", i, array[i]);
        sum += (double)array[i];
    }
    *output_mean = (float)(sum / (double)(end - start + 1));
}

/******************************************************************************
MODULE:  matlab_float_2d_partial_median

PURPOSE: simulate matlab mean function for partial part of 1 dimesion in 2d
         array cases only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/9/2015   Song Guo         Original Development
11/19/2018  Su Ye           Fixed a bug

NOTES:
******************************************************************************/

void matlab_float_2d_partial_median(
    float **array,       /* I: input array */
    int dim1_index,      /* I: 1st dimension index */
    int start,           /* I: number of start elements in 2nd dim */
    int end,             /* I: number of end elements in 2nd dim */
    float *output_median /* O: output median value */
)
{
    int m = (end + start + 1) / 2;

    if ((end - start + 1) % 2 == 0)
    {
        *output_median = (array[dim1_index][m - 1] + array[dim1_index][m]) / 2.0;
    }
    else
    {
        *output_median = array[dim1_index][m];
    }
}

/******************************************************************************
MODULE:  matlab_float_2d_partial_median

PURPOSE: simulate matlab mean function for partial part of 1 dimesion in 2d
         array cases only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
09/22/2019  Su Ye            Original development

NOTES:
******************************************************************************/

void matlab_double_2d_partial_median(
    double **array,       /* I: input array */
    int dim1_index,       /* I: 1st dimension index */
    int start,            /* I: number of start elements in 2nd dim */
    int end,              /* I: number of end elements in 2nd dim */
    double *output_median /* O: output median value */
)
{
    int m = (end + start + 1) / 2;

    if ((end - start + 1) % 2 == 0)
    {
        *output_median = (array[dim1_index][m - 1] + array[dim1_index][m]) / 2.0;
    }
    else
    {
        *output_median = array[dim1_index][m];
    }
}

/******************************************************************************
MODULE:  matlab_2d_partial_mean

PURPOSE:  simulate matlab mean function for partial part of 1 dimesion in 2d
          array cases only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/9/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void matlab_2d_partial_mean(
    float **array,     /* I: input array */
    int dim1_index,    /* I: 1st dimension index */
    int start,         /* I: number of start elements in 1st dim */
    int end,           /* I: number of end elements in 1st dim */
    float *output_mean /* O: output norm value */
)
{
    int i;
    float sum = 0.0;

    for (i = start; i <= end; i++)
    {
        sum += array[dim1_index][i];
    }
    *output_mean = sum / (float)(end - start + 1);
}

/******************************************************************************
MODULE:  matlab_2d_partial_square_mean

PURPOSE:  simulate matlab square mean function for partial part of 1 dimension
          in 2d array cases only

RETURN VALUE:
Type = void
Value           Description
-----           -----------


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/9/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void matlab_2d_partial_square_mean(
    float **array,     /* I: input array */
    int dim1_index,    /* I: 1st dimension index */
    int start,         /* I: number of start elements in 2nd dim */
    int end,           /* I: number of end elements in 2nd dim */
    float *output_mean /* O: output norm value */
)
{
    int i;
    float sum = 0;

    for (i = start; i <= end; i++)
    {
        sum += array[dim1_index][i] * array[dim1_index][i];
    }
    *output_mean = sum / (float)(end - start + 1);
}

/******************************************************************************
MODULE:  get_ids_length

PURPOSE:  get total number of non-zero elements of id array

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
2/11/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void get_ids_length(
    int *id_array, /* I: input array */
    int start,     /* I: array start index */
    int end,       /* I: array end index */
    int *id_len    /* O: number of non-zero number in the array */
)
{
    int i;
    int length = 0;

    for (i = start; i <= end; i++)
    {
        if (id_array[i] != 0)
        {
            length++;
        }
    }

    *id_len = length;
}

/******************************************************************************
MODULE:  matlab_unique

PURPOSE:  use the first value only for identical date data

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
 1/11/2016  Song Guo         Original Development

 NOTES:
******************************************************************************/

void matlab_unique(
    int *clrx,
    float **clry,
    int nums,
    int *new_nums)
{
    int k, b;
    int k_new = 0;

    for (k = 1, k_new = 1; k < nums; k++)
    {
        if (clrx[k] == clrx[k - 1])
        {
            continue;
        }
        clrx[k_new] = clrx[k];
        for (b = 0; b < TOTAL_IMAGE_BANDS; b++)
            clry[b][k_new] = clry[b][k];
        k_new++;
    }
    *new_nums = k_new;
}

/******************************************************************************
MODULE:  dofit

PURPOSE: Declare data type and allocate memory and do multiple linear robust
         fit used for auto_robust_fit

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
3/5/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void dofit(const gsl_multifit_robust_type *T,
           const gsl_matrix *X, const gsl_vector *y,
           gsl_vector *c, gsl_matrix *cov)
{
    gsl_multifit_robust_workspace *work = gsl_multifit_robust_alloc(T, X->size1, X->size2);
    gsl_multifit_robust(X, y, c, cov, work);
    gsl_multifit_robust_free(work);
    // work = NULL; // SY 03242019
}

/******************************************************************************
MODULE:  dofit_ls

PURPOSE: Declare data type and allocate memory and do multiple linear least-square
         fit used for auto_robust_fit

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
7/20/2019    Su Ye         Original Development

NOTES:
******************************************************************************/
void dofit_ls(const gsl_matrix *X, const gsl_vector *y,
              gsl_vector *c, gsl_matrix *cov)
{
    double chisq;
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(X->size1, X->size2);
    gsl_multifit_linear(X, y, c, cov, &chisq, work);
    gsl_multifit_linear_free(work);
    // work = NULL; // SY 03242019
}

/******************************************************************************
MODULE:  auto_robust_fit

PURPOSE:  Robust fit for one band

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
3/5/2015   Song Guo         Original Development

NOTES:
******************************************************************************/
void auto_robust_fit(
    double **clrx,
    float **clry,
    int nums,
    int start,
    int band_index,
    double *coefs)
{
    int i, j;
    const int p = 5; /* linear fit */
    gsl_matrix *x, *cov;
    gsl_vector *y, *c;

    /******************************************************************/
    /*                                                                */
    /* Defines the inputs/outputs for robust fitting                  */
    /*                                                                */
    /******************************************************************/

    x = gsl_matrix_alloc(nums, p);
    y = gsl_vector_alloc(nums);

    c = gsl_vector_alloc(p);
    cov = gsl_matrix_alloc(p, p);

    /******************************************************************/
    /*                                                                */
    /* construct design matrix x for linear fit                       */
    /*                                                                */
    /******************************************************************/

    for (i = 0; i < nums; ++i)
    {
        for (j = 0; j < p; j++)
        {
            if (j == 0)
            {
                gsl_matrix_set(x, i, j, 1.0);
            }
            else
            {
                gsl_matrix_set(x, i, j, clrx[i][j - 1]);
            }
        }
        gsl_vector_set(y, i, (double)clry[band_index][i + start]);
    }

    /******************************************************************/
    /*                                                                */
    /* perform robust fit                                             */
    /*                                                                */
    /******************************************************************/

    dofit(gsl_multifit_robust_bisquare, x, y, c, cov);

    for (j = 0; j < (int)c->size; j++)
    {
        coefs[j] = gsl_vector_get(c, j);
    }

    /******************************************************************/
    /*                                                                */
    /* Free the memories                                              */
    /*                                                                */
    /******************************************************************/

    gsl_matrix_free(x);
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
}

/******************************************************************************
MODULE:  auto_ls_fit

PURPOSE:  Least-sqaure fit for one band

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
7/20/2019   Su Ye         Original Development

NOTES:
******************************************************************************/
void auto_ls_fit(
    float **clrx,
    float **clry,
    int nums,
    int start,
    int band_index,
    float **coefs)
{
    int i, j;
    const int p = 5; /* 4 harmonic coefficients +1 intercept, no slope */
    gsl_matrix *x, *cov;
    gsl_vector *y, *c;

    /******************************************************************/
    /*                                                                */
    /* Defines the inputs/outputs for robust fitting                  */
    /*                                                                */
    /******************************************************************/

    x = gsl_matrix_alloc(nums, p);
    y = gsl_vector_alloc(nums);

    c = gsl_vector_alloc(p);
    cov = gsl_matrix_alloc(p, p);

    /******************************************************************/
    /*                                                                */
    /* construct design matrix x for linear fit                       */
    /*                                                                */
    /******************************************************************/

    for (i = 0; i < nums; i++)
    {
        for (j = 0; j < p; j++)
        {
            if (j == 0)
            {
                gsl_matrix_set(x, i, j, 1.0);
            }
            else
            {
                gsl_matrix_set(x, i, j, clrx[i][j - 1]);
            }
        }
        gsl_vector_set(y, i, clry[band_index][i + start]);
    }

    /******************************************************************/
    /*                                                                */
    /* perform least-square fit                                       */
    /*                                                                */
    /******************************************************************/

    dofit_ls(x, y, c, cov);

    for (j = 0; j < (int)c->size; j++)
    {
        coefs[band_index][j] = gsl_vector_get(c, j);
    }

    /******************************************************************/
    /*                                                                */
    /* Free the memories                                              */
    /*                                                                */
    /******************************************************************/

    gsl_matrix_free(x);
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
}

/******************************************************************************
MODULE:  auto_mask

PURPOSE:  Multitemporal cloud, cloud shadow, & snow masks (global version)

RETURN VALUE:
Type = int
ERROR error out due to memory allocation
SUCCESS no error encounted

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
3/5/2015    Song Guo         Original Development
20160104    Song Guo         Numerous bug fixes.
20160513    Brian Davis      Added SUCCESS argument to int return.
20240812    Su Ye            Add band number as options
NOTES:
******************************************************************************/
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
)
{
    char FUNC_NAME[] = "auto_mask";
    int year;
    double w, w2;
    int i;
    double **x;
    double pred_b2, pred_b5;
    int nums;
    double coefs[ROBUST_COEFFS];
    double coefs2[ROBUST_COEFFS];

    nums = end - start + 1;
    /* Allocate memory */
    x = (double **)allocate_2d_array(nums, ROBUST_COEFFS - 1, sizeof(double));
    if (x == NULL)
    {
        RETURN_ERROR("ERROR allocating x memory", FUNC_NAME, ERROR);
    }

    year = (int)(ceil(years));
    w = TWO_PI / AVE_DAYS_IN_A_YEAR;
    w2 = w / (double)year;

    for (i = 0; i < nums; i++)
    {
        x[i][0] = cos(w * (double)clrx[i + start]);
        x[i][1] = sin(w * (double)clrx[i + start]);
        x[i][2] = cos(w2 * (double)clrx[i + start]);
        x[i][3] = sin(w2 * (double)clrx[i + start]);
    }

    /******************************************************************/
    /*                                                                */
    /* Do robust fitting for band 2 */
    /*                                                                */
    /******************************************************************/

    auto_robust_fit(x, clry, nums, start, tmask_b1 - 1, coefs);

    /******************************************************************/
    /*                                                                */
    /* Do robust fitting for band 5 */
    /*                                                                */
    /******************************************************************/

    auto_robust_fit(x, clry, nums, start, tmask_b2 - 1, coefs2);

    /******************************************************************/
    /*                                                                */
    /* predict band 2 * band 5 refs, bl_ids value of 0 is clear and   */
    /* 1 otherwise                                                    */
    /*                                                                */
    /******************************************************************/

    for (i = 0; i < nums; i++)
    {
        pred_b2 = coefs[0] + coefs[1] * cos((double)clrx[i + start] * w) +
                  coefs[2] * sin((double)clrx[i + start] * w) + coefs[3] * cos((double)clrx[i + start] * w2) + coefs[4] * sin((double)clrx[i + start] * w2);
        pred_b5 = coefs2[0] + coefs2[1] * cos((double)clrx[i + start] * w) +
                  coefs2[2] * sin((double)clrx[i + start] * w) + coefs2[3] * cos((double)clrx[i + start] * w2) + coefs2[4] * sin((double)clrx[i + start] * w2);
        if ((fabs(clry[tmask_b1 - 1][i + start] - pred_b2) > (n_t * t_b1)) ||
            (fabs(clry[tmask_b2 - 1][i + start] - pred_b5) > (n_t * t_b2)))
        {
            // printf("%f, %f\n", clry[1][i+start],clry[1][i+start]-pred_b2);
            bl_ids[i] = 1;
        }
        else
        {
            // printf("%f, %f\n", clry[1][i+start],clry[1][i+start]-pred_b2);
            bl_ids[i] = 0;
        }
    }

    /* Free allocated memory */
    if (free_2d_array((void **)x) != SUCCESS)
    {
        RETURN_ERROR("Freeing memory: x\n", FUNC_NAME, ERROR);
    }

    return (SUCCESS);
}

/******************************************************************************
MODULE:  auto_ts_predict

PURPOSE:  Using lasso regression fitting coefficients to predict new values

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
3/5/2015    Song Guo         Original Development
20160104    Song Guo         Numerous bug fixes.
09192018    Su Ye            Add lasso band inex
NOTES:
******************************************************************************/
int auto_ts_predict(
    int *clrx,
    double **coefs,
    int df,
    int lasso_band_index,
    int start,
    int end,
    double *pred_y)
{
    char FUNC_NAME[] = "auto_ts_predict";
    int i;
    int nums = end - start + 1;
    double w, w2, w3;
    w = TWO_PI / AVE_DAYS_IN_A_YEAR;
    w2 = 2.0 * w;
    w3 = 3.0 * w;

    for (i = 0; i < nums; i++)
    {
        if (df == 2)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] *
                                                         (double)clrx[i + start];
        }
        else if (df == 4)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * (double)clrx[i + start] + coefs[lasso_band_index][2] * cos((double)clrx[i + start] * w) + coefs[lasso_band_index][3] * sin((double)clrx[i + start] * w);
        }
        else if (df == 5)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * cos((double)clrx[i + start] * w) + coefs[lasso_band_index][2] * sin((double)clrx[i + start] * w) + coefs[lasso_band_index][3] * cos((double)clrx[i + start] * w2) + coefs[lasso_band_index][4] * sin((double)clrx[i + start] * w2);
        }
        else if (df == 6)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * (double)clrx[i + start] + coefs[lasso_band_index][2] * cos((double)clrx[i + start] * w) + coefs[lasso_band_index][3] * sin((double)clrx[i + start] * w) + coefs[lasso_band_index][4] * cos((double)clrx[i + start] * w2) + coefs[lasso_band_index][5] * sin((double)clrx[i + start] * w2);
        }
        else if (df == 8)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * (double)clrx[i + start] + coefs[lasso_band_index][2] * cos((double)clrx[i + start] * w) + coefs[lasso_band_index][3] * sin((double)clrx[i + start] * w) + coefs[lasso_band_index][4] * cos((double)clrx[i + start] * w2) + coefs[lasso_band_index][5] * sin((double)clrx[i + start] * w2) + coefs[lasso_band_index][6] * cos((double)clrx[i + start] * w3) + coefs[lasso_band_index][7] * sin((double)clrx[i + start] * w3);
        }
        else
        {
            RETURN_ERROR("Unsupported df number", FUNC_NAME, ERROR);
        }
    }

    return (SUCCESS);
}

int auto_ts_predict_float(
    int *clrx,
    float **coefs,
    int df,
    int lasso_band_index,
    int start,
    int end,
    float *pred_y)
{
    char FUNC_NAME[] = "auto_ts_predict_float";
    int i;
    int nums = end - start + 1;
    float w, w2, w3;
    w = TWO_PI / AVE_DAYS_IN_A_YEAR;
    w2 = 2 * w;
    w3 = 3 * w;

    for (i = 0; i < nums; i++)
    {
        if (df == 2)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] *
                                                         (float)clrx[i + start] / SLOPE_SCALE;
        }
        else if (df == 4)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * (float)clrx[i + start] / SLOPE_SCALE + coefs[lasso_band_index][2] * cosf((float)clrx[i + start] * w) + coefs[lasso_band_index][3] * sinf((float)clrx[i + start] * w);
        }
        else if (df == 5)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * cosf((float)clrx[i + start] * w) + coefs[lasso_band_index][2] * sinf((float)clrx[i + start] * w) + coefs[lasso_band_index][3] * cosf((float)clrx[i + start] * w2) + coefs[lasso_band_index][4] * sinf((float)clrx[i + start] * w2);
        }
        else if (df == 6)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * (float)clrx[i + start] / SLOPE_SCALE + coefs[lasso_band_index][2] * cosf((float)clrx[i + start] * w) + coefs[lasso_band_index][3] * sinf((float)clrx[i + start] * w) + coefs[lasso_band_index][4] * cosf((float)clrx[i + start] * w2) + coefs[lasso_band_index][5] * sinf((float)clrx[i + start] * w2);
        }
        else if (df == 8)
        {
            pred_y[i] = coefs[lasso_band_index][0] + coefs[lasso_band_index][1] * (float)clrx[i + start] / SLOPE_SCALE + coefs[lasso_band_index][2] * cosf((float)clrx[i + start] * w) + coefs[lasso_band_index][3] * sinf((float)clrx[i + start] * w) + coefs[lasso_band_index][4] * cosf((float)clrx[i + start] * w2) + coefs[lasso_band_index][5] * sinf((float)clrx[i + start] * w2) + coefs[lasso_band_index][6] * cosf((float)clrx[i + start] * w3) + coefs[lasso_band_index][7] * sinf((float)clrx[i + start] * w3);
        }
        else
        {
            RETURN_ERROR("Unsupported df number", FUNC_NAME, ERROR);
        }
    }

    return (SUCCESS);
}

/******************************************************************************
MODULE:  auto_ts_fit

PURPOSE:  Lasso regression fitting with full outputs

RETURN VALUE:
Type = int
ERROR error in allocating memories
SUCCESS no error encounted

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
11/23/2015  Song Guo         Original Development
20151203    Brian Davis      Merge of versions.
20160104    Song Guo         Numerous bug fixes.
20160517    Brian Davis      Removed temporary hack of initializing lmu to 1.
                             There now seem to be no data-dependent crashes
                             with lmu initialized to some strange value(s).
                             Put df and nums correctly into error message
                             sprintf when malloc of y fails.
                             Incorporated fix to initialize coefs to 0.0.

NOTES:
******************************************************************************/
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
    float lam)
{
    char FUNC_NAME[] = "auto_ts_fit";
    char errmsg[MAX_STR_LEN];
    double w;
    int i, j;
    float **x;
    float *y;
    int status;
    double *yhat;
    double v_dif_norm = 0.0;
    int nums = 0;
    int nlam = 1; // number of lambda

    float alpha = 1.0;
    int lmu;
    double cfs[nlam][df];
    nums = end - start + 1;

    w = TWO_PI / 365.25;

    // float ulam[1] = {
    //     20,
    // }; // SY lambda = 20
    float ulam[1];
    ulam[0] = lam; // SY lambda = 20
    /* Allocate memory */
    if (df == 2 || df == 4 || df == 5 || df == 6 || df == 8)
    {
        x = (float **)allocate_2d_array(df - 1, nums, sizeof(float));
        if (x == NULL)
        {
            sprintf(errmsg, "Allocating x memory for %d - 1 times %d size of double",
                    df, nums);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        y = malloc(nums * sizeof(float));
        if (y == NULL)
        {
            sprintf(errmsg, "Allocating y memory %d %d", df, nums);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }
    }
    else
    {
        RETURN_ERROR("Unsupported df value", FUNC_NAME, ERROR);
    }

    yhat = (double *)malloc(nums * sizeof(double));
    if (yhat == NULL)
    {
        RETURN_ERROR("Allocating yhat memory", FUNC_NAME, ERROR);
    }

    switch (df)
    {

    case 2:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                coefs[lasso_band_index][j] = cfs[i][j];
            }
        }
        break;

    case 4:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = (float)cos(w * (float)clrx[i + start]);

            x[2][i] = (float)sin(w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];

            // printf("%f, %f, %f, %f\n", x[0][i], x[1][i], x[2][i], y[i]);
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                coefs[lasso_band_index][j] = cfs[i][j];
                // printf("%d band %d cof: %f\n", band_index, j, coefs[lasso_band_index][j]);
            }
        }
        break;

    case 5:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)cos(w * (float)clrx[i + start]);

            x[1][i] = (float)sin(w * (float)clrx[i + start]);
            x[2][i] = (float)cos(2.0 * w * (float)clrx[i + start]);
            x[3][i] = (float)sin(2.0 * w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];

            // printf("%f, %f, %f, %f\n", x[0][i], x[1][i], x[2][i], y[i]);
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                coefs[lasso_band_index][j] = cfs[i][j];
                // printf("%d band %d cof: %f\n", band_index, j, coefs[lasso_band_index][j]);
            }
        }
        break;

    case 6:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = (float)cos(w * (float)clrx[i + start]);
            x[2][i] = (float)sin(w * (float)clrx[i + start]);
            x[3][i] = (float)cos(2.0 * w * (float)clrx[i + start]);
            x[4][i] = (float)sin(2.0 * w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                coefs[lasso_band_index][j] = cfs[i][j];
            }
        }
        break;

    case 8:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = (float)cos(w * (float)clrx[i + start]);
            x[2][i] = (float)sin(w * (float)clrx[i + start]);
            x[3][i] = (float)cos(2.0 * w * (float)clrx[i + start]);
            x[4][i] = (float)sin(2.0 * w * (float)clrx[i + start]);
            x[5][i] = (float)cos(3.0 * w * (float)clrx[i + start]);
            x[6][i] = (float)sin(3.0 * w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                coefs[lasso_band_index][j] = cfs[i][j];
            }
        }
        break;
    }

    /* predict lasso model results */
    if (df == 2 || df == 4 || df == 5 || df == 6 || df == 8)
    {
        auto_ts_predict(clrx, coefs, df, lasso_band_index, start, end, yhat);
        for (i = 0; i < nums; i++)
        {
            v_dif[lasso_band_index][i] = (double)clry[band_index][i + start] - yhat[i];
        }
        matlab_2d_array_norm(v_dif, lasso_band_index, nums, &v_dif_norm);
        *rmse = v_dif_norm / sqrt((double)(nums - df));
        //*rmse = v_dif_norm / sqrt((double)(nums)); //SY 08182019
    }

    /* Free allocated memory */
    free(yhat);
    if (df == 2 || df == 4 || df == 5 || df == 6 || df == 8)
    {
        if (free_2d_array((void **)x) != SUCCESS)
        {
            RETURN_ERROR("Freeing memory: x\n", FUNC_NAME, ERROR);
        }
        free(y);
    }

    return (SUCCESS);
}

/******************************************************************************
MODULE:  auto_ts_fit_sccd

PURPOSE:  Lasso regression fitting with full outputs for sccd

RETURN VALUE:
Type = int
ERROR error in allocating memories
SUCCESS no error encounted

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
11/23/2015  Song Guo         Original Development
20151203    Brian Davis      Merge of versions.
20160104    Song Guo         Numerous bug fixes.
20160517    Brian Davis      Removed temporary hack of initializing lmu to 1.
                             There now seem to be no data-dependent crashes
                             with lmu initialized to some strange value(s).
                             Put df and nums correctly into error message
                             sprintf when malloc of y fails.
                             Incorporated fix to initialize coefs to 0.0.

NOTES:
******************************************************************************/
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
    float lam)
{
    char FUNC_NAME[] = "auto_ts_fit_sccd";
    char errmsg[MAX_STR_LEN];
    float w;
    int i, j;
    float **x;
    float *y;
    int status;
    float *yhat;
    float v_dif_norm = 0.0;
    int nums = 0;
    int nlam = 1; // number of lambda

    float alpha = 1.0;
    int lmu;
    double cfs[nlam][df];
    nums = end - start + 1;

    w = TWO_PI / 365.25;

    float ulam[1];
    ulam[0] = lam;

    /* Allocate memory */
    if (df == 2 || df == 4 || df == 5 || df == 6 || df == 8)
    {
        x = (float **)allocate_2d_array(df - 1, nums, sizeof(float));
        if (x == NULL)
        {
            sprintf(errmsg, "Allocating x memory for %d - 1 times %d size of float",
                    df, nums);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        y = malloc(nums * sizeof(float));
        if (y == NULL)
        {
            sprintf(errmsg, "Allocating y memory %d %d", df, nums);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }
    }
    else
    {
        RETURN_ERROR("Unsupported df value", FUNC_NAME, ERROR);
    }

    yhat = (float *)malloc(nums * sizeof(float));
    if (yhat == NULL)
    {
        RETURN_ERROR("Allocating yhat memory", FUNC_NAME, ERROR);
    }

    switch (df)
    {

    case 2:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < SCCD_NUM_C; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                if (j == 1)
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j] * SLOPE_SCALE);
                }
                else
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j]);
                }
            }
        }
        break;

    case 4:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = (float)cos(w * (float)clrx[i + start]);

            x[2][i] = (float)sin(w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];

            // printf("%f, %f, %f, %f\n", x[0][i], x[1][i], x[2][i], y[i]);
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < SCCD_NUM_C; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                if (j == 1)
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j] * SLOPE_SCALE);
                }
                else
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j]);
                }
            }
        }
        break;
    case 6:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = (float)cos(w * (float)clrx[i + start]);
            x[2][i] = (float)sin(w * (float)clrx[i + start]);
            x[3][i] = (float)cos(2.0 * w * (float)clrx[i + start]);
            x[4][i] = (float)sin(2.0 * w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < SCCD_NUM_C; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                if (j == 1)
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j] * SLOPE_SCALE);
                }
                else
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j]);
                }
            }
        }
        break;
    }

    /* predict lasso model results */
    if (df == 2 || df == 4 || df == 6)
    {
        auto_ts_predict_float(clrx, coefs, df, lasso_band_index, start, end, yhat);
        for (i = 0; i < nums; i++)
        {
            v_dif[lasso_band_index][i] = (float)clry[band_index][i + start] - yhat[i];
        }
        matlab_2d_array_norm_float(v_dif, lasso_band_index, nums, &v_dif_norm);
        *rmse = v_dif_norm / sqrt((float)(nums - df));
        //*rmse = v_dif_norm / sqrt((double)(nums)); //SY 08182019
    }

    /* Free allocated memory */
    free(yhat);
    if (df == 2 || df == 4 || df == 5 || df == 6)
    {
        if (free_2d_array((void **)x) != SUCCESS)
        {
            RETURN_ERROR("Freeing memory: x\n", FUNC_NAME, ERROR);
        }
        free(y);
    }

    return (SUCCESS);
}

/******************************************************************************
MODULE:  auto_ts_fit_float

PURPOSE:  Lasso regression fitting with full outputs

RETURN VALUE:
Type = int
ERROR error in allocating memories
SUCCESS no error encounted

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
11/23/2015  Song Guo         Original Development
20151203    Brian Davis      Merge of versions.
20160104    Song Guo         Numerous bug fixes.
20160517    Brian Davis      Removed temporary hack of initializing lmu to 1.
                             There now seem to be no data-dependent crashes
                             with lmu initialized to some strange value(s).
                             Put df and nums correctly into error message
                             sprintf when malloc of y fails.
                             Incorporated fix to initialize coefs to 0.0.

NOTES:
******************************************************************************/
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
    float lam)
{
    char FUNC_NAME[] = "auto_ts_fit_float";
    char errmsg[MAX_STR_LEN];
    float w;
    int i, j;
    float **x;
    float *y;
    int status;
    float *yhat;
    float v_dif_norm = 0.0;
    int nums = 0;
    int nlam = 1; // number of lambda

    float alpha = 1.0;
    int lmu;
    double cfs[nlam][df];
    nums = end - start + 1;

    w = TWO_PI / 365.25;

    float ulam[1];
    ulam[0] = lam; // SY lambda = 20

    /* Allocate memory */
    if (df == 2 || df == 4 || df == 5 || df == 6 || df == 8)
    {
        x = (float **)allocate_2d_array(df - 1, nums, sizeof(float));
        if (x == NULL)
        {
            sprintf(errmsg, "Allocating x memory for %d - 1 times %d size of float",
                    df, nums);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        y = malloc(nums * sizeof(float));
        if (y == NULL)
        {
            sprintf(errmsg, "Allocating y memory %d %d", df, nums);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }
    }
    else
    {
        RETURN_ERROR("Unsupported df value", FUNC_NAME, ERROR);
    }

    yhat = (float *)malloc(nums * sizeof(float));
    if (yhat == NULL)
    {
        RETURN_ERROR("Allocating yhat memory", FUNC_NAME, ERROR);
    }

    switch (df)
    {

    case 2:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                if (j == 1)
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j] * SLOPE_SCALE);
                }
                else
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j]);
                }
            }
        }
        break;

    case 4:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = cosf(w * (float)clrx[i + start]);

            x[2][i] = sinf(w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];

            // printf("%f, %f, %f, %f\n", x[0][i], x[1][i], x[2][i], y[i]);
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                if (j == 1)
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j] * SLOPE_SCALE);
                }
                else
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j]);
                }
            }
        }
        break;

    case 5:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = cosf(w * (float)clrx[i + start]);

            x[1][i] = sinf(w * (float)clrx[i + start]);
            x[2][i] = cosf(2 * w * (float)clrx[i + start]);
            x[3][i] = sinf(2 * w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];

            // printf("%f, %f, %f, %f\n", x[0][i], x[1][i], x[2][i], y[i]);
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR)
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                coefs[lasso_band_index][j] = cfs[i][j];
                // printf("%d band %d cof: %f\n", band_index, j, coefs[lasso_band_index][j]);
            }
        }
        break;

    case 6:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = cosf(w * (float)clrx[i + start]);
            x[2][i] = sinf(w * (float)clrx[i + start]);
            x[3][i] = cosf(2 * w * (float)clrx[i + start]);
            x[4][i] = sinf(2 * w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                if (j == 1)
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j] * SLOPE_SCALE);
                }
                else
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j]);
                }
            }
        }
        break;

    case 8:
        for (i = 0; i < nums; i++)
        {
            x[0][i] = (float)clrx[i + start];
            x[1][i] = cosf(w * (float)clrx[i + start]);
            x[2][i] = sinf(w * (float)clrx[i + start]);
            x[3][i] = cosf(2 * w * (float)clrx[i + start]);
            x[4][i] = sinf(2 * w * (float)clrx[i + start]);
            x[5][i] = cosf(3 * w * (float)clrx[i + start]);
            x[6][i] = sinf(3 * w * (float)clrx[i + start]);
            y[i] = (float)clry[band_index][i + start];
        }

        status = c_glmnet(nums, df - 1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
        if (status != SUCCESS)
        {
            sprintf(errmsg, "Calling c_glmnet when df = %d", df);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR)
        }

        for (i = 0; i < LASSO_COEFFS; i++)
            coefs[lasso_band_index][i] = 0.0;

        for (i = 0; i < lmu; i++)
        {
            for (j = 0; j < df; j++)
            {
                if (j == 1)
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j] * SLOPE_SCALE);
                }
                else
                {
                    coefs[lasso_band_index][j] = (float)(cfs[i][j]);
                }
            }
        }
        break;
    }

    /* predict lasso model results */
    if (df == 2 || df == 4 || df == 5 || df == 6 || df == 8)
    {
        auto_ts_predict_float(clrx, coefs, df, lasso_band_index, start, end, yhat);
        for (i = 0; i < nums; i++)
        {
            v_dif[lasso_band_index][i] = (float)clry[band_index][i + start] - yhat[i];
        }
        matlab_2d_array_norm_float(v_dif, lasso_band_index, nums, &v_dif_norm);
        *rmse = v_dif_norm / sqrtf((float)(nums - df));
        //*rmse = v_dif_norm / sqrt((double)(nums)); //SY 08182019
    }

    /* Free allocated memory */
    free(yhat);
    if (df == 2 || df == 4 || df == 5 || df == 6 || df == 8)
    {
        if (free_2d_array((void **)x) != SUCCESS)
        {
            RETURN_ERROR("Freeing memory: x\n", FUNC_NAME, ERROR);
        }
        free(y);
    }

    return (SUCCESS);
}

/******************************************************************************
MODULE:  auto_ts_ft_ls

PURPOSE:  Least-sqaure fit for one band

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
7/20/2019   Su Ye         Original Development

NOTES:
******************************************************************************/
// int auto_ts_fit_ls
//(
//     int *clrx,
//     float **clry,
//     int band_index,
//     int start,
//     int end,
//     int df,
//     float **coefs,
//     float *rmse,
//     float **v_dif
//)
//{
//     char FUNC_NAME[] = "auto_ts_fit_ls";
//     char errmsg[MAX_STR_LEN];
//     float w;
//     int i;
//     float **x;
//     float *y;
//     float *yhat;
//     float v_dif_norm = 0.0;
//     int nums = 0.0;

//    nums = end - start + 1;

//    w = TWO_PI / AVE_DAYS_IN_A_YEAR;

//    /* Allocate memory */
//    if (df == 5)
//    {
//        x = (float **)allocate_2d_array(nums, df - 1,sizeof(float));
//        if (x == NULL)
//        {
//            sprintf(errmsg, "Allocating x memory for %d - 1 times %d size of double",
//                    df, nums);
//            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
//        }
//    }
//    else
//    {
//        RETURN_ERROR("Unsupported df value", FUNC_NAME, ERROR);
//    }

//    yhat = (float *)malloc(nums * sizeof(float));
//    if (yhat == NULL)
//    {
//        RETURN_ERROR("Allocating yhat memory", FUNC_NAME, ERROR);
//    }

//    switch (df)
//    {
//       case 5:
//            for (i = 0; i < nums; i++)
//            {
//                x[i][0]= (float)cos(w * (float)clrx[i+start]);

//                x[i][1] = (float)sin(w * (float)clrx[i+start]);
//                x[i][2] = (float)cos(2 * w * (float)clrx[i+start]);

//                x[i][3]= (float)sin(2 * w * (float)clrx[i+start]);

//                // printf("%f, %f, %f, %f\n", x[0][i], x[1][i], x[2][i], y[i]);

//            }

//            auto_ls_fit(x, clry, nums, start, band_index, coefs);
////            status = c_glmnet(nums, df-1, &x[0][0], y, nlam, ulam, alpha, &lmu, cfs);
////            if (status != SUCCESS)
////            {
////                sprintf(errmsg, "Calling c_glmnet when df = %d", df);
////                RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
////            }

//            break;
//    }

//    /* predict least-square model results */
//    if (df == 5)
//    {
//        auto_ts_predict(clrx, coefs, df, band_index, start, end, yhat);
//        for (i = 0; i < nums; i++)
//        {
//            v_dif[band_index][i] = clry[band_index][i+start] - yhat[i];
//        }
//        matlab_2d_array_norm(v_dif, band_index, nums, &v_dif_norm);
//        *rmse = v_dif_norm / sqrt((float)(nums - df));
//    }

//    /* Free allocated memory */
//    free(yhat);
//    if (df == 5)
//    {
//        if (free_2d_array ((void **) x) != SUCCESS)
//        {
//            RETURN_ERROR ("Freeing memory: x\n", FUNC_NAME, ERROR);
//        }
//    }

//    return (SUCCESS);

//}

/******************************************************************************
MODULE:  c_glmnet

PURPOSE:  The R equivalant function in C for Lasso regression fitting

RETURN VALUE:
Type = int
ERROR error in allocating memories
SUCCESS no error encounted

HISTORY:
Date        Programmer       Original development
--------    ---------------  -------------------------------------
11/23/2015   Song Guo         Geoscience Australia
12/16/2015   Song Guo         Fixed bug to return the
                              correct order of coefficients

NOTES:
******************************************************************************/

int c_glmnet(
    int no,      // number of observations (no)
    int ni,      // number of predictor variables (ni)
    float *x,    // input matrix, x[ni][no]
    float *y,    // response vaiable, of dimentions (no)
    int nlam,    // number of lambda values
    float *ulam, // value of lambda values, of dimentions (nlam)
    float parm,  // the alpha variable

    int *lmu,                // lmu = actual number of lamda values (solutions)
    double cfs[nlam][ni + 1] // results = cfs[lmu][ni + 1]
)
{
    float w[no];     // weight(no), default to a sequence of 1's
    float vp[ni];    // penalty factor (ni), default to a sequence of 1's
    float cl[ni][2]; // lower and upper limits (ni, 2), default (-inf, +inf)

    int ka = 1;      // (ni < 600)? 1:2 20190912 SY
    int jd[1] = {0}; // 20190912 SY
    // int jd[2] = {1, 0};  //20190912 SY
    int ne = ni + 1;            // dfmax = ni + 1
    int nx = min(ne * 1.2, ni); // nx = min(dfmax * 2 + 20, ni) 20190912 SY

    float flmin = 1.0;   // if lambda is NULL, then flmin is derived from lambda.min.ration:
                         // (no < ni)? 0.01:0.0001; otherwise flmin = 1.0
    float thr = 1.0e-04; // thresh in R, default 1.0e-07, change it to 10e-4 to keep consistent with matlab version SY
    int isd = 1;         // derivide from standardize in R, default is True = 1
    int maxit = 10000;   // default is 10000     SY

    float a0[nlam];     //   a0(lmu) = intercept values for each solution
    float ca[nlam][nx]; // ca(nx,lmu) = compressed coefficient values for each solution
    int ia[nx];         //   ia(nx) = pointers to compressed coefficients
    int nin[nlam];      //   nin(lmu) = number of compressed coefficients for each solution
    float rsq[nlam];    //   rsq(lmu) = R**2 values for each solution
    float alm[nlam];    //   alm(lmu) = lamda values corresponding to each solution
    int nlp;            // actual number of passes over the data for all lamda values
    int jerr;           // error flag

    int i, j;

    for (i = 0; i < no; i++)
    {
        w[i] = 1;
    }

    for (i = 0; i < ni; i++)
    {
        vp[i] = 1;
        cl[i][0] = -INFINITY;
        cl[i][1] = INFINITY;
    }

    // printf("step1 \n");
    elnet_(&ka, &parm, &no, &ni, x, y, w, jd, vp, &ne, &nx,
           &nlam, &flmin, ulam, &thr, &isd,
           lmu, a0, &ca[0][0], ia, nin, rsq, alm, &nlp, &jerr);

    // printf("step2 \n");
    //  solns_(&ni, &nx, lmu, &ca[0][0], ia, nin, &b[0][0]);

    for (i = 0; i < *lmu; i++)
    {
        cfs[i][0] = (double)a0[i];
        if (nin[0] == 0)
        {
            for (j = 0; j < ni; j++)
            {
                cfs[i][j + 1] = 0;
            }
        }
        else
        {
            int *oja;
            oja = malloc(nin[0] * sizeof(int));
            for (int k = 0; k < nin[0]; k++)
                oja[k] = k;

            quick_sort_index(ia, oja, 0, nin[0] - 1);
            // printf("step3 \n");

            for (j = 0; j < ni; j++)
            {
                cfs[i][j + 1] = 0;
            }

            for (j = 0; j < nin[0]; j++)
            {
                cfs[i][ia[j]] = (double)ca[i][oja[j]];
            }

            free(oja);
        }
    }

    return jerr;
}

/******************************************************************************
MODULE:  max_array_int

PURPOSE:  calculate max of array


HISTORY:
Date        Programmer       Original development
--------    ---------------  -------------------------------------
11/15/2018   Su Ye           Reason


NOTES:
******************************************************************************/

void max_array_int(int a[], int num_elements, int *max_element, int *max_loc)
{
    int i;
    int max = -999999;
    for (i = 0; i < num_elements; i++)
    {
        if (a[i] > max)
        {
            *max_element = a[i];
            *max_loc = i;
            max = a[i];
        }
    }
}

/******************************************************************************
MODULE:  max_array_int


HISTORY:
Date        Programmer       Original development
--------    ---------------  -------------------------------------
11/15/2018   Su Ye           Reason


NOTES:
******************************************************************************/

double normalCDF(double u)
{
    static const double a[5] =
        {
            1.161110663653770e-002, 3.951404679838207e-001, 2.846603853776254e+001,
            1.887426188426510e+002, 3.209377589138469e+003};
    static const double b[5] =
        {
            1.767766952966369e-001, 8.344316438579620e+000, 1.725514762600375e+002,
            1.813893686502485e+003, 8.044716608901563e+003};
    static const double c[9] =
        {
            2.15311535474403846e-8, 5.64188496988670089e-1, 8.88314979438837594e00,
            6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02,
            1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725E03};
    static const double d[9] =
        {
            1.00000000000000000e00, 1.57449261107098347e01, 1.17693950891312499e02,
            5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03,
            4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03};
    static const double p[6] =
        {
            1.63153871373020978e-2, 3.05326634961232344e-1, 3.60344899949804439e-1,
            1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4};
    static const double q[6] =
        {
            1.00000000000000000e00, 2.56852019228982242e00, 1.87295284992346047e00,
            5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3};
    double y, z;

    y = fabs(u);
    if (y <= 0.46875 * 1.4142135623730950488016887242097)
    {
        /* evaluate erf() for |u| <= sqrt(2)*0.46875 */
        z = y * y;
        y = u * ((((a[0] * z + a[1]) * z + a[2]) * z + a[3]) * z + a[4]) / ((((b[0] * z + b[1]) * z + b[2]) * z + b[3]) * z + b[4]);
        return 0.5 + y;
    }

    z = exp(-y * y / 2) / 2;
    if (y <= 4.0)
    {
        /* evaluate erfc() for sqrt(2)*0.46875 <= |u| <= sqrt(2)*4.0 */
        y = y / 1.4142135623730950488016887242097;
        y =
            ((((((((c[0] * y + c[1]) * y + c[2]) * y + c[3]) * y + c[4]) * y + c[5]) * y + c[6]) * y + c[7]) * y + c[8]) / ((((((((d[0] * y + d[1]) * y + d[2]) * y + d[3]) *
                                                                                                                                    y +
                                                                                                                                d[4]) *
                                                                                                                                   y +
                                                                                                                               d[5]) *
                                                                                                                                  y +
                                                                                                                              d[6]) *
                                                                                                                                 y +
                                                                                                                             d[7]) *
                                                                                                                                y +
                                                                                                                            d[8]);

        y = z * y;
    }
    else
    {
        /* evaluate erfc() for |u| > sqrt(2)*4.0 */
        z = z * 1.4142135623730950488016887242097 / y;
        y = 2 / (y * y);
        y = y * (((((p[0] * y + p[1]) * y + p[2]) * y + p[3]) * y + p[4]) * y + p[5]) /
            (((((q[0] * y + q[1]) * y + q[2]) * y + q[3]) * y + q[4]) * y + q[5]);
        y = z * (0.564189583547756286948 - y);
    }
    return (u < 0.0 ? y : 1 - y);
}

/******************************************************************************
MODULE:  normalQuantile


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
11/15/2018   Su Ye           Original develpment


NOTES:
******************************************************************************/

double normalQuantile(double p)
{
    double q, t, u;

    static const double a[6] =
        {
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] =
        {
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01};
    static const double c[6] =
        {
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] =
        {
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00};

    q = min(p, 1 - p);

    if (q > 0.02425)
    {
        /* Rational approximation for central region. */
        u = q - 0.5;
        t = u * u;
        u = u * (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) /
            (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1);
    }
    else
    {
        /* Rational approximation for tail region. */
        t = sqrt(-2 * log(q));
        u = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1);
    }

    /* The relative error of the approximation has absolute value less
    than 1.15e-9.  One iteration of Halley's rational method (third
    order) gives full machine precision... */
    t = normalCDF(u) - q;                                      /* error */
    t = t * 2.506628274631000502415765284811 * exp(u * u / 2); /* f(u)/df(u) */
    u = u - t / (1 + u * t / 2);                               /* Halley's method */

    return (p > 0.5 ? -u : u);
}

/******************************************************************************
MODULE:  chi2inv


HISTORY:
Date        Programmer       Original development
--------    ---------------  -------------------------------------
11/15/2018   Su Ye           Reason


NOTES:
******************************************************************************/

double chi2inv(double P, unsigned int dim)
{
    if (P == 0)
    {
        return 0;
    }
    else
        return dim * pow(1.0 - 2.0 / (9 * dim) +
                             sqrt(2.0 / (9 * dim)) * normalQuantile(P),
                         3);
}

/******************************************************************************
MODULE:  MeanAngl


HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
11/15/2018   Su Ye           Original development


NOTES:
******************************************************************************/

double MeanAngl(
    double **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,   // input: the number of lasso band
    int i_count      // input: the number of consecutive observations
)
{
    double y;
    int i, j;
    double product;
    double norm1;
    double norm2;
    double angl_sum = 0;
    double tmp;
    if (i_count > 1)
    {
        for (i = 0; i < i_count - 1; i++)
        {
            product = 0;
            norm1 = 0;
            norm2 = 0;
            for (j = 0; j < lasso_num; j++)
            {
                product += v_diff[j][i] * v_diff[j][i + 1];
                norm1 += v_diff[j][i] * v_diff[j][i];
                norm2 += v_diff[j][i + 1] * v_diff[j][i + 1];
            }
            tmp = acos(product / (sqrt(norm1) * sqrt(norm2)));
            angl_sum += (tmp * 180.0) / PI;
        }
        y = angl_sum / (i_count - 1);
    }
    else
    {
        y = 0;
    }
    return y;
}

float MeanAngl_float(
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count     // input: the number of consecutive observations
)
{
    float y;
    int i, j;
    float product;
    float norm1;
    float norm2;
    float angl_sum = 0;
    float tmp;
    if (i_count > 1)
    {
        for (i = 0; i < i_count - 1; i++)
        {
            product = 0;
            norm1 = 0;
            norm2 = 0;
            for (j = 0; j < lasso_num; j++)
            {
                product += v_diff[j][i] * v_diff[j][i + 1];
                norm1 += v_diff[j][i] * v_diff[j][i];
                norm2 += v_diff[j][i + 1] * v_diff[j][i + 1];
            }
            tmp = acosf(product / (sqrtf(norm1) * sqrtf(norm2)));
            angl_sum += (tmp * 180.0) / PI;
        }
        y = angl_sum / (i_count - 1);
    }
    else
    {
        y = 0;
    }
    return y;
}

float MeanAngl_float_selective(
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count,    // input: the number of consecutive observations
    int *lasso_bands)
{
    float y;
    int i, j;
    float product;
    float norm1;
    float norm2;
    float angl_sum = 0;
    float tmp;
    if (i_count > 1)
    {
        for (i = 0; i < i_count - 1; i++)
        {
            product = 0;
            norm1 = 0;
            norm2 = 0;
            for (j = 0; j < lasso_num; j++)
            {
                product += v_diff[lasso_bands[j]][i] * v_diff[lasso_bands[j]][i + 1];
                norm1 += v_diff[lasso_bands[j]][i] * v_diff[lasso_bands[j]][i];
                norm2 += v_diff[lasso_bands[j]][i + 1] * v_diff[lasso_bands[j]][i + 1];
            }
            tmp = acosf(product / (sqrtf(norm1) * sqrtf(norm2)));
            angl_sum += (tmp * 180.0) / PI;
        }
        y = angl_sum / (i_count - 1);
    }
    else
    {
        y = 0;
    }
    return y;
}

float MediumAngl(
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count     // input: the number of consecutive observations
)
{
    float y;
    int i, j;
    float product;
    float norm1;
    float norm2;
    float *angle;
    if (i_count > 1)
    {
        angle = (float *)malloc((i_count - 1) * sizeof(float));

        for (i = 0; i < i_count - 1; i++)
        {
            product = 0;
            norm1 = 0;
            norm2 = 0;
            for (j = 0; j < lasso_num; j++)
            {
                product += v_diff[j][i] * v_diff[j][i + 1];
                norm1 += v_diff[j][i] * v_diff[j][i];
                norm2 += v_diff[j][i + 1] * v_diff[j][i + 1];
            }
            angle[i] = (acos(product / (sqrt(norm1) * sqrt(norm2))) * 180.0) / PI;
            // printf("%f\n", angle[i]);
        }

        quick_sort_float(angle, 0, i_count - 2);

        int m = (i_count - 1) / 2;

        if ((i_count - 1) % 2 == 0)
        {
            y = (angle[m - 1] + angle[m]) / 2.0;
        }
        else
        {
            y = angle[m];
        }

        free(angle);
    }
    else
    {
        y = 0;
    }
    return y;
}

float angl_scatter_measure(
    float *med_diff,
    float **v_diff, // input: a two-dimensional vector of different (i_count * lasso_num)
    int lasso_num,  // input: the number of lasso band
    int i_count,    // input: the number of consecutive observations
    int *lasso_bands)
{
    float y;
    int i, j;
    float product;
    float norm1;
    float norm2;
    float *angle;
    float angle_sum = 0;
    float interm;
    if (i_count > 2)
    {
        angle = (float *)malloc((i_count) * sizeof(float));

        for (i = 0; i < i_count; i++)
        {
            product = 0;
            norm1 = 0;
            norm2 = 0;
            for (j = 0; j < lasso_num; j++)
            {
                product += v_diff[lasso_bands[j]][i] * med_diff[lasso_bands[j]];
                norm1 += v_diff[lasso_bands[j]][i] * v_diff[lasso_bands[j]][i];
                norm2 += med_diff[lasso_bands[j]] * med_diff[lasso_bands[j]];
                // printf("%f\n", med_diff[j]);
            }
            interm = (float)(product / (sqrtf(norm1) * sqrtf(norm2)));
            if (interm > 1)
                interm = 1;
            angle[i] = (acos(interm) * 180.0) / PI;
            angle_sum = angle_sum + angle[i];
            // printf("%f\n", angle[i]);
        }

        //        quick_sort_double(angle, 0, i_count-1);

        //        int m = (i_count - 1) / 2;

        //        if (i_count % 2 == 0)
        //        {
        //            y = (angle[m - 1] + angle[m]) / 2.0;
        //        }
        //        else
        //        {
        //            y = angle[m] ;
        //        }
        y = angle_sum / (i_count);
        free(angle);
    }
    else
    {
        y = 0;
    }
    return y;
}

int singleband_variogram(
    float *array,    /* I: input array                                    */
    int i_start,     /* I: dimension 2 start index                        */
    int i_end,       /* I: dimension 2 end index                          */
    float *variogram /* O: outputted median variogran for dates           */
)
{
    char FUNC_NAME[] = "singleband_variogram";
    int len = i_end - i_start;
    int m, j;
    float *var;
    var = (float *)malloc(len * sizeof(float));
    if (var == NULL)
    {
        RETURN_ERROR("Allocating var memory", FUNC_NAME, ERROR);
    }

    for (j = i_start; j < i_end; j++)
    {
        var[j - i_start] = fabs(array[j + 1] - array[j]);
        // printf("%d var for band %d: %f\n", j, i+1, (float)var[j]);
    }

    quick_sort_float(var, 0, i_end - i_start - 1);
    //            for (j = 0; j < dim2_end; j++)
    //            {
    //               printf("%f\n", var[j]);
    //            }
    m = len / 2;
    if (len % 2 == 0)
        *variogram = (var[m - 1] + var[m]) / 2.0;

    else
        *variogram = var[m];

    free(var);
    return SUCCESS;
}

int singleband_minvariogram(
    int *clrx,
    float *array,    /* I: input array                                    */
    int i_start,     /* I: dimension 2 start index                        */
    int i_end,       /* I: dimension 2 end index                          */
    float *variogram /* O: outputted median variogran for dates           */
)
{
    char FUNC_NAME[] = "singleband_variogram";
    int len = i_end - i_start;
    int m, j;
    float *var;
    var = (float *)malloc(len * sizeof(float));
    if (var == NULL)
    {
        RETURN_ERROR("Allocating var memory", FUNC_NAME, ERROR);
    }

    for (j = i_start; j < i_end; j++)
    {
        var[j - i_start] = ((float)fabs(array[j + 1] - array[j])) / ((float)(clrx[j + 1] - clrx[j]));
        // printf("%d var for band %d: %f\n", j, i+1, (float)var[j]);
    }

    quick_sort_float(var, 0, i_end - i_start - 1);
    //            for (j = 0; j < dim2_end; j++)
    //            {
    //               printf("%f\n", var[j]);
    //            }
    m = len / 2;
    if (len % 2 == 0)
        *variogram = (var[m - 1] + var[m]) / 2.0;

    else
        *variogram = var[m];

    free(var);
    return SUCCESS;
}

int yearmonth2doy(
    int year,
    int month,
    int day)
{
    if (TRUE == is_leap_year(year))
    {
        if (month == 1)
            return day;
        else if (month == 2)
            return 31 + day;
        else if (month == 3)
            return 60 + day;
        else if (month == 4)
            return 91 + day;
        else if (month == 5)
            return 121 + day;
        else if (month == 6)
            return 152 + day;
        else if (month == 7)
            return 182 + day;
        else if (month == 8)
            return 213 + day;
        else if (month == 9)
            return 244 + day;
        else if (month == 10)
            return 274 + day;
        else if (month == 11)
            return 305 + day;
        else if (month == 12)
            return 335 + day;
    }
    else
    {
        if (month == 1)
            return day;
        else if (month == 2)
            return 31 + day;
        else if (month == 3)
            return 59 + day;
        else if (month == 4)
            return 90 + day;
        else if (month == 5)
            return 120 + day;
        else if (month == 6)
            return 151 + day;
        else if (month == 7)
            return 181 + day;
        else if (month == 8)
            return 212 + day;
        else if (month == 9)
            return 243 + day;
        else if (month == 10)
            return 273 + day;
        else if (month == 11)
            return 304 + day;
        else if (month == 12)
            return 334 + day;
    }
    return SUCCESS;
}

/* translated from https://github.com/USGS-EROS/lcmap-pyccd/blob/develop/ccd/qa.py */
bool checkbit(
    int packedint,
    int offset)
{
    int bit;
    bit = 1 << offset;
    if ((packedint & bit) > 0)
        return TRUE;
    else
        return FALSE;
}

// int qabitval
//(
//     int packedint
//)
//{
//     if (checkbit(packedint, QA_FILL))
//         return CFMASK_FILL;
//     else if (checkbit(packedint, QA_CLOUD))
//         return CFMASK_CLOUD;
//     else if (checkbit(packedint, QA_SHADOW))
//         return CFMASK_SHADOW;
//     else if (checkbit(packedint, QA_SNOW))
//         return CFMASK_SNOW;
//     else if (checkbit(packedint, QA_WATER))
//         return CFMASK_WATER;
//     else if (checkbit(packedint, QA_CLEAR))
//         return CFMASK_CLEAR;
//     // L8 Cirrus and Terrain Occlusion
//     else if (checkbit(packedint, QA_CIRRUS1) &
//           checkbit(packedint, QA_CIRRUS2))
//         return CFMASK_CLEAR;
//     else if (checkbit(packedint, QA_OCCLUSION))
//         return CFMASK_CLEAR;
// }

double X2(int nu, double pr)
{
    double t0 = 0.0;
    double t = 2.0;
    double diff;
    double delta = 0.1;
    double p;
    int n = 0;

    t = 0.0;
    while ((p = Chi_Square_Distribution(t, nu)) < pr)
    {
        t += delta;
        if (p > 0.999999)
        {
            delta /= 10.0;
            t = (t - delta) / 10;
        }
    }
    t0 = t - delta;
    if (t0 < 0.0)
        t0 = 0.0;
    while ((p = Chi_Square_Distribution(t0, nu)) > pr)
    {
        t0 -= delta;
        if (p < 0.001)
        {
            delta /= 10.0;
            t0 = (t0 + delta) / 10;
        }
    }
    while (fabs(t - t0) > 1.e-6)
    {
        if (t < 0.0)
            t = 0.0;
        diff = Chi_Square_Distribution(t, nu) - pr;
        diff /= Chi_Square_Density(t, nu);
        diff /= 2.0;
        t0 = t;
        t -= diff;
        n++;
        if (n > 40)
            exit(0);
    }
    return t;
}
