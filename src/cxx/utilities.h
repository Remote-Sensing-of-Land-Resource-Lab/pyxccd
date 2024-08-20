#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#define LOG_MESSAGE(message, module)           \
    write_message((message), (module), "INFO", \
                  __FILE__, __LINE__, stdout);

#define WARNING_MESSAGE(message, module)          \
    write_message((message), (module), "WARNING", \
                  __FILE__, __LINE__, stdout);

#define ERROR_MESSAGE(message, module)          \
    write_message((message), (module), "ERROR", \
                  __FILE__, __LINE__, stdout);

#define RETURN_ERROR(message, module, status)       \
    {                                               \
        write_message((message), (module), "ERROR", \
                      __FILE__, __LINE__, stdout);  \
        return (status);                            \
    }

#define RETURN_WARNING(message, module, status)       \
    {                                                 \
        write_message((message), (module), "WARNING", \
                      __FILE__, __LINE__, stdout);    \
        return (status);                              \
    }

void write_message(
    const char *message, /* I: message to write to the log */
    const char *module,  /* I: module the message is from */
    const char *type,    /* I: type of the error */
    char *file,          /* I: file the message was generated in */
    int line,            /* I: line number in the file where the message was
                               generated */
    FILE *fd             /* I: where to write the log message */
);

char *sub_string /* explicit control of a substring function  */
    (
        const char *source, /* I: input string                           */
        size_t start,       /* I: index for start of sub string          */
        size_t length       /* I: number of characters to grab           */
    );

void quick_sort(int arr[], char *brr[], int crr[], int left, int right);
int partition(int arr[], char *brr[], int crr[], int left, int right);
void quick_sort_float(float arr[], int left, int right);
void quick_sort_double(double arr[], int left, int right);
void quick_sort_long(int64_t arr[], int left, int right);
int partition_float(float arr[], int left, int right);
int partition_double(double arr[], int left, int right);
int partition_long(int64_t arr[], int left, int right);
int partition_index(int arr[], int *index, int left, int right);
void quick_sort_buf(int arr[], short int **brr, short int *fmask_buf, int left, int right);
void quick_sort_index(int arr[], int *index, int left, int right);
int partition_buf(int arr[], short int **buf, short int *fmask_buf, int left, int right);
int partition_buf_sensor(int arr[], short int **buf, short int *fmask_buf, short int *sensor, int left, int right);

// int get_variables
//(
//     int argc,              /* I: number of cmd-line args                    */
//     char *argv[],          /* I: string of cmd-line args                    */
//     int *mode,              /* O: 1 - pixel-based; 2 - scanline-based;       */
//                            /*    3 - wholescene                             */
//     char *in_path,         /* O: directory locaiton for input data          */
//     char *out_path,        /* O: directory location for output files        */
//     int *n_cores,            /* O: number of cores                            */
//     int *row,              /* O: (required for mode 1 and mode 2)           */
//                            /*    row number for the pixel                   */
//     int *col,              /* O: (required for mode 1)                      */
//                            /* O: col number for the pixel                   */
//     int *task,
//     char *user_mask_path,   /* O: directory location for user_mask           */
//     bool *b_fastmode,
//     bool *b_outputCSV
//);

// int get_args
//(
//     int argc,              /* I: number of cmd-line args                    */
//     char *argv[],          /* I: string of cmd-line args                    */
//     int *mode,             /* O: 1 - pixel-based; 2 - scanline-based; 3 - wholescene; 4-validation */
//     char *in_path,         /* O: directory locaiton for input data          */
//     char *out_path,        /* O: directory location for output files        */
//     int *n_cores,          /* O: the number of cores                        */
//     int *row,              /* O: row number for the pixel                   */
//     int *col,              /* O: col number for the pixel                   */
//     int *method,
//     char *user_mask_path,
//     double *probability_threshold,
//     int *min_days_conse,
//     bool *b_fastmode,
//     bool *b_outputCSV
//);

int find_index_clrx(
    int *clrx,
    int end,
    int input_ordinal_date);

int usage_message();

int preprocessing(
    int64_t *buf_b,        /* I:  Landsat blue spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_g,        /* I:  Landsat green spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_r,        /* I:  Landsat red spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_n,        /* I:  Landsat NIR spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s1,       /* I:  Landsat swir1 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_s2,       /* I:  Landsat swir2 spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *buf_t,        /* I:  Landsat thermal spectral time series.The dimension is (n_obs, 7). Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,    /* I:   mask time series  */
    int *valid_num_scenes, /* I/O: * number of scenes after cfmask counts and  */
    int *id_range,
    int *clear_sum,  /* I/O: Total number of clear cfmask pixels          */
    int *water_sum,  /* I/O: counter for cfmask water pixels.             */
    int *shadow_sum, /* I/O: counter for cfmask shadow pixels.            */
    int *sn_sum,     /* I/O: Total number of snow cfmask pixels           */
    int *cloud_sum,  /* I/O: counter for cfmask cloud pixels.             */
    bool b_c2        /* I: indicate if it is collection 2   */
);

int preprocessing_flex(
    int64_t *ts_data,      /* I:  band 7 time series. Invalid (qa is filled value (255)) must be removed */
    int64_t *fmask_buf,    /* I:   mask time series  */
    int *valid_num_scenes, /* I/O: * number of scenes after cfmask counts and  */
    int *id_range,
    int *clear_sum,  /* I/O: Total number of clear cfmask pixels          */
    int *water_sum,  /* I/O: counter for cfmask water pixels.             */
    int *shadow_sum, /* I/O: counter for cfmask shadow pixels.            */
    int *sn_sum,     /* I/O: Total number of snow cfmask pixels           */
    int *cloud_sum,  /* I/O: counter for cfmask cloud pixels.             */
    int nbands       /* the number of input bands*/
);

double angle_decaying(
    double input,
    double lowbound,
    double highbound);

#endif /* UTILITIES_H */
