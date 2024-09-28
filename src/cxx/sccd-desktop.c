#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <stdio.h>
#include <sys/timeb.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include "defines.h"
#include "cold.h"
#include "const.h"
#include "utilities.h"
#include "2d_array.h"
#include "input.h"
#include "output.h"
#include "misc.h"
#include "s_ccd.h"
#include "cold_flex.h"
#include "s_ccd_flex.h"

#define MAX_N_STATES 10000

/******************************************************************************
MODULE:  get_coldparameters
PURPOSE:  Gets the parameters from yaml
RETURN VALUE:
Type = int
Value           Description
-----           -----------
FAILURE         Error Can't locate yaml file
SUCCESS         No errors encountered
SOURCE: https://stackoverflow.com/questions/49785153/c-reading-from-txt-file-into-struct
HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
09/01/2021  Su Ye            orginal develop
******************************************************************************/
int get_coldparameters(
    int *n_rows,
    int *n_cols,
    int *n_block_x,
    int *n_block_y,
    int *CM_OUTPUT_INTERVAL,
    float *probability_threshold,
    int *conse)
{
    char cwd[MAX_STR_LEN]; // current directory path
    // char var_path[MAX_STR_LEN];
    FILE *var_fp;
    char line[MAX_STR_LEN];
    char var_path[MAX_STR_LEN];
    char errmsg[MAX_STR_LEN]; /* error message   */
    char *token;
    const char deli[] = ":";
    char FUNC_NAME[] = "get_coldparameters";
    char s1[] = "n_rows";
    char s2[] = "n_cols";
    char s3[] = "n_block_x";
    char s4[] = "n_block_y";
    char s5[] = "CM_OUTPUT_INTERVAL";
    char s6[] = "probability_threshold";
    char s7[] = "conse";

    getcwd(cwd, sizeof(cwd));
    // printf("getvariable");
    sprintf(var_path, "%s/%s", cwd, "config.yaml");

    var_fp = fopen(var_path, "r");

    if (var_fp == NULL)
    {
        sprintf(errmsg, "no config.yaml was found in %s \n", cwd);
        RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
    }

    while (true)
    {
        if (fscanf(var_fp, " %[^\n]", line) != EOF)
        {
            token = strtok(line, deli);
            if (strcmp(token, s1) == 0)
            {
                token = strtok(NULL, deli);
                *n_rows = atoi(token);
            }
            else if (strcmp(token, s2) == 0)
            {
                token = strtok(NULL, deli);
                *n_cols = atoi(token);
            }
            else if (strcmp(token, s3) == 0)
            {
                token = strtok(NULL, deli);
                *n_block_x = atoi(token);
            }
            else if (strcmp(token, s4) == 0)
            {
                token = strtok(NULL, deli);
                *n_block_y = atoi(token);
            }
            else if (strcmp(token, s5) == 0)
            {
                token = strtok(NULL, deli);
                *CM_OUTPUT_INTERVAL = atoi(token);
            }
            else if (strcmp(token, s6) == 0)
            {
                token = strtok(NULL, deli);
                *probability_threshold = atof(token);
            }
            else if (strcmp(token, s7) == 0)
            {
                token = strtok(NULL, deli);
                *conse = atoi(token);
            }
        }
        else
        {
            break; // end of file
        }
    }

    fclose(var_fp);

    if (*n_rows == 0)
        RETURN_ERROR("n_rows is missing in the parameter.yaml", FUNC_NAME, ERROR);
    if (*n_cols == 0)
        RETURN_ERROR("n_cols is missing in the parameter.yaml", FUNC_NAME, ERROR);
    if (*n_block_x == 0)
        RETURN_ERROR("n_block_x is missing in the parameter.yaml", FUNC_NAME, ERROR);
    if (*n_block_y == 0)
        RETURN_ERROR("n_block_y is missing in the parameter.yaml", FUNC_NAME, ERROR);
    if (*CM_OUTPUT_INTERVAL == 0)
        RETURN_ERROR("CM_OUTPUT_INTERVAL is missing in the parameter.yaml", FUNC_NAME, ERROR);
    if (*probability_threshold == 0)
        RETURN_ERROR("probability_threshold is missing in the parameter.yaml", FUNC_NAME, ERROR);
    if (*conse == 0)
        RETURN_ERROR("conse is missing in the parameter.yaml", FUNC_NAME, ERROR);
    return SUCCESS;
}

int getnrtstructurefromtxt(char *nrtoutput_affix, Output_sccd *s_rec_cg, output_nrtmodel *nrt_model,
                           output_nrtqueue *obs_queue, int *nrt_mode, int *num_fc,
                           int *num_obs_queue, int *pos, short *min_rmse)
{
    char line[100000];
    // read nrt mode
    FILE *ptr;
    int nline = 1;
    const char *mode_ext = "_nrt_mode";
    const char *reccg_ext = "_rec_cg";
    const char *queue_ext = "_nrt_queue";
    const char *nrtmodel_ext = "_nrt_model";
    const char *rmse_ext = "_min_rmse";
    char mode_with_extension[MAX_STR_LEN];
    char reccg_with_extension[MAX_STR_LEN];
    char nrt_queue_with_extension[MAX_STR_LEN];
    char model_with_extension[MAX_STR_LEN];
    char rmse_with_extension[MAX_STR_LEN];

    // read mode
    strcpy(mode_with_extension, nrtoutput_affix); /* copy name into the new var */
    strcat(mode_with_extension, mode_ext);
    /* add the extension */
    ptr = fopen(mode_with_extension, "rb"); // r for read, b for binary
    fread(nrt_mode, sizeof(int), 1, ptr);
    fclose(ptr);

    // read rec_cg
    strcpy(reccg_with_extension, nrtoutput_affix); /* copy name into the new var */
    strcat(reccg_with_extension, reccg_ext);
    /* add the extension */
    ptr = fopen(reccg_with_extension, "rb"); // r for read, b for binary
    while (fread(&s_rec_cg[*num_fc], sizeof(Output_sccd), 1, ptr) == 1)
    {
        *num_fc = *num_fc + 1;
    }
    fclose(ptr);

    // read min rmse
    strcpy(rmse_with_extension, nrtoutput_affix); /* copy name into the new var */
    strcat(rmse_with_extension, rmse_ext);
    /* add the extension */
    ptr = fopen(rmse_with_extension, "rb"); // r for read, b for binary
    if (fread(min_rmse, sizeof(short), TOTAL_IMAGE_BANDS_SCCD, ptr) != TOTAL_IMAGE_BANDS_SCCD)
    {
        printf("reading minimum rmse fails \n");
    }

    fclose(ptr);
    if ((*nrt_mode == NRT_MONITOR_SNOW) | (*nrt_mode == NRT_MONITOR_STANDARD))
    {
        strcpy(model_with_extension, nrtoutput_affix); /* copy name into the new var */
        strcat(model_with_extension, nrtmodel_ext);
        /* add the extension */
        ptr = fopen(model_with_extension, "rb"); // r for read, b for binary
        fread(nrt_model, sizeof(output_nrtmodel), 1, ptr);
        fclose(ptr);
    }
    else if ((*nrt_mode == NRT_QUEUE_SNOW) | (*nrt_mode == NRT_QUEUE_STANDARD))
    {
        strcpy(nrt_queue_with_extension, nrtoutput_affix); /* copy name into the new var */
        strcat(nrt_queue_with_extension, queue_ext);
        /* add the extension */
        ptr = fopen(nrt_queue_with_extension, "rb"); // r for read, b for binary
        while (fread(&obs_queue[*num_obs_queue], sizeof(output_nrtqueue), 1, ptr) == 1)
        {
            *num_obs_queue = *num_obs_queue + 1;
        }
        fclose(ptr);
    }

    return SUCCESS;
}

int main(int argc, char *argv[])
{
    int result;
    /* inputted argument, exe mode */
    char in_path[MAX_STR_LEN];
    float probability_threshold;
    int conse;

    // printf("nonono");

    /**************************************************************/
    /*                                                            */
    /*   read CCD variable                                        */
    /*                                                            */
    /**************************************************************/
    /* need to recover for exe */
    //    result = get_variables(argc, argv, &mode, in_path, out_path, &n_cores,
    //                           &row, &col, &task, mask_path, &b_fastmode, &b_outputCSV);

    //    char in_path[] = "/media/su/DataFactory/Dissertation_Meta/Landsat/ENVI_LandsatARD";
    //    char mask_path[] = "/home/su/Documents/RProjects/Dissertation/Results/hostspecies_mask.envi";
    //    char out_path[] = "/media/su/DataFactory/Dissertation_Meta/Landsat/ccdc_result/RUN0";

    //    int n_cores = 8;
    //    int row = 310;
    //    int col = 10;
    //    //int row = 1;
    //    //int col =
    //    int mode = 2;                    /* CCD detection mode
    //                                        3: whole images; 1: pixel-based;
    //                                        2: scanline-based*/
    /* inputted argument */ /* CCD detection mode
                         3: whole images; 1: pixel-based;
                         2: scanline-based*/
    int method;
    // bool verbose = TRUE;

    char msg_str[MAX_STR_LEN]; /* Input data scene name                 */
    int i, j;                  /* Loop counters                         */
    int starting_date;
    int status;                /* Return value from function call       */
    char FUNC_NAME[] = "main"; /* For printing error messages           */

    int64_t *sdate;    /* Pointer to list of acquisition dates  */
    char **scene_list; /* 2-D array for list of scene IDs       */

    int num_scenes; /* Number of input scenes defined        */

    time_t now; /* For logging the start, stop, and some     */

    int num_fc = 0; /* the number of functional curve        */
    char out_filename[MAX_STR_LEN];
    // block_num = (int)meta->lines / threads;

    /**************************************************************************/
    /*                   Parallel scanline processing                         */
    /**************************************************************************/

    int64_t *fmask_buf; /* fmask buf, valid pixels only*/
    int64_t **buf;      /* This is the image bands buffer, valid pixel only*/
    // int n_working_process; /* total processes - 1 (master process)*/
    int i_col;
    FILE *fhoutput;
    FILE *fhoutput_cm;
    FILE *fhoutput_cm_date;
    FILE *fhoutput_cm_direction; /* This is the image bands buffer, valid pixel only*/
    // short int *tmp_fmask_buf;              /* fmask buf, valid pixels only          */
    // int *tmp_valid_date_array;             /* Sdate array after cfmask filtering    */
    Output_t *rec_cg;
    Output_sccd *s_rec_cg;
    Output_t_flex *rec_cg_flex;
    Output_sccd_flex *s_rec_cg_flex;
    int64_t *sensor_buf;
    double tcg;
    int n_cm_maps = 0;
    short int *CM_outputs;
    short int *CM_outputs_date;
    unsigned char *CMdirection_outputs;
    // bool b_singleline = FALSE;
    int num_breakdatemaps;
    int64_t *breakdates_block;
    // int sample_row = 0;
    // int sample_col = 0;

    char CM_filename[MAX_STR_LEN];
    char CM_direction_filename[MAX_STR_LEN];
    char CM_date_filename[MAX_STR_LEN];
    char CM_fullpath[MAX_STR_LEN];
    char CM_direction_fullpath[MAX_STR_LEN];
    char CM_date_fullpath[MAX_STR_LEN];
    char out_fullpath[MAX_STR_LEN];
    char out_fullpath_tmp[MAX_STR_LEN];
    bool b_obcold_reconstruction = FALSE; // indicate if it is used to reconstruct rec_cg in obcold
    bool b_outputCM = FALSE;
    bool b_partition = TRUE; //
    int n_rows = 0;
    int n_cols = 0;
    // int ROW_STEP = 0;
    // int PARTITION = 0;
    int CM_OUTPUT_INTERVAL;
    int nvals;
    Input_meta_t *meta;
    int n_block_v;
    int n_block_h;
    int current_block_x = 0;
    int current_block_y = 0;

    int block_width = 1;
    int block_height = 1;
    FILE *sampleFile;
    char *csv_row;
    int valid_scene_count = 0;
    int pixel_qa;
    num_scenes = MAX_SCENE_LIST;
    output_nrtmodel *nrt_model;
    output_nrtmodel_flex *nrt_model_flex;
    int num_obs_queue = 0;
    output_nrtqueue_flex *obs_queue_flex;
    output_nrtqueue *obs_queue;
    int nrt_mode = 0;
    int pos;
    short min_rmse[TOTAL_IMAGE_BANDS] = {0, 0, 0, 0, 0, 0, 0};
    int cm_output_interval;
    bool b_c2 = FALSE;
    bool b_header_csv = FALSE;
    int headline = -1;
    bool b_pinpoint = TRUE;
    Output_sccd_pinpoint *rec_cg_pinpoint;
    Output_sccd_pinpoint_flex *rec_cg_pinpoint_flex;
    int num_fc_pinpoint = 0;
    if (b_header_csv == TRUE)
        headline = 0; // skip the head line of csv
    double gate_tcg = 9.236;
    double gap_days = NUM_YEARS;
    bool b_coefs_records = TRUE;
    int n_coefs_records = 0;
    nrt_coefs_records *coefs_records;
    nrt_coefs_records_flex *coefs_records_flex;
    tcg = 15.086;
    double pcg = 0.99;
    int state_intervaldays = 30.4;
    int n_state = 0;
    int64_t *state_days;
    int64_t *buf_stack;
    double *state_ensemble;
    double t_cg;
    double max_t_cg;

    /**************************************************************/
    /*                                                            */
    /*   preprocessing inputted parameters                        */
    /*                                                            */
    /**************************************************************/
    if (argv[1][0] == 'r')
    {
        b_obcold_reconstruction = TRUE;
        method = COLD;
    }
    else if (argv[1][0] == 's')
    {
        method = SCCD;
        cm_output_interval = 60;
    }
    else if (argv[1][0] == 'c')
    {
        method = COLD;
        cm_output_interval = 60;
    }
    else if (argv[1][0] == 'o')
    {
        b_outputCM = TRUE;
        method = COLD;
        cm_output_interval = 60;
    }
    else if (argv[1][0] == 't')
    {
        method = SCCDONLINE;
        cm_output_interval = 999999; // assigned an extreme to the interval as only one cm is saved
    }
    else if (argv[1][0] == 'f')
    {
        method = COLD_FLEX;
        cm_output_interval = 60;
    }
    else if (argv[1][0] == 'g')
    {
        method = SCCD_FLEX;
        cm_output_interval = 60;
    }
    else
        RETURN_ERROR("The second input parameter has to be r, s, o, f, g, c", FUNC_NAME, FAILURE);

    /* stack is complete image, not partitions */
    if (argv[1][1] == 'c')
    {
        b_partition = FALSE;
    }

    // argv[2] and argv[3] are in_path and out_path
    //  optional parameters

    get_coldparameters(&n_rows, &n_cols, &n_block_h, &n_block_v, &CM_OUTPUT_INTERVAL, &probability_threshold, &conse);

    // probability_threshold = (double)strtol(argv[6], NULL, 10);
    //    probability_threshold = DEFAULT_PROBABILITY;
    //    conse = DEFAULT_CONSE;

    meta = (Input_meta_t *)malloc(sizeof(Input_meta_t));
    if (meta == NULL)
    {
        RETURN_ERROR("ERROR allocating meta",
                     FUNC_NAME, FAILURE);
    }

    /**************************************************************/
    /*                                                            */
    /*   record the start time of just the COLD                   */
    /*     algorithm.  Up until here, it has all just been        */
    /*     setting it up......                                    */
    /*                                                            */
    /**************************************************************/
    time(&now); /*     intermediate times.                   */
    snprintf(msg_str, sizeof(msg_str), "COLD starts \n", ctime(&now));
    LOG_MESSAGE(msg_str, FUNC_NAME);

    /* if out_path not exist, create it*/
    //    struct stat st_tmp = {0};
    //    if (stat(argv[3], &st_tmp) == -1) {
    //        mkdir(argv[3], 0700);
    //    }

    // chi-square probability
    // tcg = X2(NUM_LASSO_BANDS, probability_threshold);

    /**************************************************************/
    /*                                                            */
    /* Now that we know the actual number of scenes, allocate     */
    /* memory for date array.                                     */
    /*                                                            */
    /**************************************************************/
    sdate = (int64_t *)malloc(num_scenes * sizeof(int64_t));

    if (sdate == NULL)
    {
        RETURN_ERROR("ERROR allocating sdate memory", FUNC_NAME, FAILURE);
    }

    buf = (int64_t **)allocate_2d_array(TOTAL_IMAGE_BANDS, MAX_SCENE_LIST, sizeof(int64_t));
    if (buf == NULL)
    {
        RETURN_ERROR("Allocating buf", FUNC_NAME, FAILURE);
    }

    buf_stack = (int64_t *)malloc(MAX_SCENE_LIST * sizeof(int64_t) * 5);
    if (buf_stack == NULL)
    {
        RETURN_ERROR("Allocating buf", FUNC_NAME, FAILURE);
    }

    fmask_buf = (int64_t *)malloc(num_scenes * sizeof(int64_t));
    if (fmask_buf == NULL)
    {
        RETURN_ERROR("Allocating fmask_buf", FUNC_NAME, FAILURE);
    }

    sensor_buf = (int64_t *)malloc(MAX_SCENE_LIST * sizeof(int64_t));
    if (sensor_buf == NULL)
    {
        RETURN_ERROR("Allocating sensor_buf ", FUNC_NAME, FAILURE);
    }

    csv_row = malloc(MAX_STR_LEN * sizeof(char));
    if (csv_row == NULL)
    {
        RETURN_ERROR("Allocating  csv_row", FUNC_NAME, FAILURE);
    }

    // printf("%d \n", num_scenes);
    sampleFile = fopen(argv[2], "r");
    int row_count = 0;
    while (fgets(csv_row, 255, sampleFile) != NULL)
    {
        if (row_count != headline) // we skip first line because it is a header
        {
            sdate[valid_scene_count] = atoi(strtok(csv_row, ","));
            buf[0][valid_scene_count] = (int64_t)atoi(strtok(NULL, ","));
            buf[1][valid_scene_count] = (int64_t)atoi(strtok(NULL, ","));
            buf[2][valid_scene_count] = (int64_t)atoi(strtok(NULL, ","));
            buf[3][valid_scene_count] = (int64_t)atoi(strtok(NULL, ","));
            buf[4][valid_scene_count] = (int64_t)atoi(strtok(NULL, ","));
            buf[5][valid_scene_count] = (int64_t)atoi(strtok(NULL, ","));
            buf[6][valid_scene_count] = (int64_t)atoi(strtok(NULL, ","));
            pixel_qa = atoi(strtok(NULL, ","));
            //                if (training_type == 1)
            //                {
            //                   fmask_buf[valid_scene_count] = (short)pixel_qa;
            //                   sensor_buf[valid_scene_count] = (short)atoi(strtok(NULL, ","));
            //                }
            //                else
            //                   fmask_buf[valid_scene_count] = (short)qabitval(pixel_qa);
            // fmask_buf[valid_scene_count] = (short)qabitval(pixel_qa);
            fmask_buf[valid_scene_count] = (int64_t)pixel_qa;
            // sensor_buf[valid_scene_count] = (short)atoi(strtok(NULL, ","));

            valid_scene_count++;
        }
        row_count++;
    }
    fclose(sampleFile);
    if (sdate[valid_scene_count - 1] - sdate[0] == 0)
    {
        n_cm_maps = 1;
    }
    else
    {
        n_cm_maps = (sdate[valid_scene_count - 1] - sdate[0]) / cm_output_interval + 1;
    }

    if (method == SCCDONLINE)
        starting_date = 0;
    else
        starting_date = sdate[0];

    state_days = (int64_t *)malloc((sdate[valid_scene_count - 1] - sdate[0] + 1) * sizeof(int64_t));

    state_ensemble = (double *)malloc((sdate[valid_scene_count - 1] - sdate[0] + 1) * 3 * sizeof(double));
    //    sensor_buf = (short int *) malloc(num_scenes * sizeof(short int));
    //    if(sensor_buf == NULL)
    //    {
    //        RETURN_ERROR ("Allocating sensor_buf fails", FUNC_NAME, FAILURE);
    //    }

    //    scene_list = (char **) allocate_2d_array (MAX_SCENE_LIST, ARD_STR_LEN,
    //                                                       sizeof (char));
    //    if(scene_list == NULL)
    //    {
    //        RETURN_ERROR ("Allocating scene_list", FUNC_NAME, FAILURE);
    //    }

    CM_outputs = malloc(sizeof(short int) * n_cm_maps);
    if (CM_outputs == NULL)
    {
        printf("%d h%d v%d", n_cm_maps, current_block_x, current_block_y);
        RETURN_ERROR("ERROR allocating CM_outputs", FUNC_NAME, FAILURE);
    }

    CMdirection_outputs = malloc(sizeof(unsigned char) * n_cm_maps);
    if (CMdirection_outputs == NULL)
    {
        RETURN_ERROR("ERROR allocating CM_outputs_date", FUNC_NAME, FAILURE);
    }

    CM_outputs_date = malloc(sizeof(short int) * n_cm_maps);
    if (CM_outputs_date == NULL)
    {
        RETURN_ERROR("ERROR allocating CM_outputs_date", FUNC_NAME, FAILURE);
    }

    s_rec_cg = malloc(NUM_FC * sizeof(Output_sccd));
    if (s_rec_cg == NULL)
    {
        RETURN_ERROR("ERROR allocating s_rec_cg",
                     FUNC_NAME, FAILURE);
    }

    rec_cg_pinpoint = malloc(NUM_FC * sizeof(Output_sccd_pinpoint));
    if (rec_cg_pinpoint == NULL)
    {
        RETURN_ERROR("ERROR allocating rec_cg_pinpoint",
                     FUNC_NAME, FAILURE);
    }

    rec_cg_pinpoint_flex = malloc(NUM_FC * sizeof(Output_sccd_pinpoint_flex));
    if (rec_cg_pinpoint_flex == NULL)
    {
        RETURN_ERROR("ERROR allocating rec_cg_pinpoint_flex",
                     FUNC_NAME, FAILURE);
    }

    nrt_model = malloc(sizeof(output_nrtmodel));
    if (nrt_model == NULL)
    {
        RETURN_ERROR("ERROR allocating nrt_model",
                     FUNC_NAME, FAILURE);
    }

    nrt_model_flex = malloc(sizeof(output_nrtmodel_flex));
    if (nrt_model_flex == NULL)
    {
        RETURN_ERROR("ERROR allocating nrt_model_flex",
                     FUNC_NAME, FAILURE);
    }

    obs_queue = malloc(MAX_OBS_QUEUE * sizeof(output_nrtmodel));
    if (obs_queue == NULL)
    {
        RETURN_ERROR("ERROR allocating obs_queue",
                     FUNC_NAME, FAILURE);
    }

    coefs_records = malloc(MAX_OBS * sizeof(nrt_coefs_records));
    if (coefs_records == NULL)
    {
        RETURN_ERROR("Allocating  coefs_records", FUNC_NAME, FAILURE);
    }

    /******************************************************************/
    /*                                                                */
    /* Start reading break info into the memory                          */
    /*                                                                */
    /******************************************************************/
    if (b_obcold_reconstruction == TRUE)
    {

        breakdates_block = malloc(MAX_YEAR_RANGE * sizeof(int64_t));
        if (breakdates_block == NULL)
        {
            RETURN_ERROR("Allocating breakdates_block", FUNC_NAME, FAILURE);
        }
        snprintf(msg_str, sizeof(msg_str), "breakdatemap reading finished\n");
        LOG_MESSAGE(msg_str, FUNC_NAME)

        sampleFile = fopen(argv[4], "r");
        num_breakdatemaps = 0;
        row_count = 0;
        while (fgets(csv_row, 255, sampleFile) != NULL)
        {
            if (row_count != 0) // we skip first line because it is a header
            {
                breakdates_block[num_breakdatemaps] = atoi(strtok(csv_row, ","));
                num_breakdatemaps++;
            }
            row_count++;
        }
        fclose(sampleFile);
    }

    /******************************************************************/
    /*                                                                */
    /* reading existing SCCD data structure  for only sccd                         */
    /*                                                                */
    /******************************************************************/
    if (SCCDONLINE == method)
    {
        result = getnrtstructurefromtxt(argv[4], s_rec_cg, nrt_model,
                                        obs_queue, &nrt_mode, &num_fc,
                                        &num_obs_queue, &pos, min_rmse);
    }

    /******************************************************************/
    /*                                                                */
    /*       Reading dataset finished                                 */
    /*                                                                */
    /******************************************************************/

    snprintf(msg_str, sizeof(msg_str), "fopen and fread finished \n");
    LOG_MESSAGE(msg_str, FUNC_NAME)

    // for(original_row = starting_row; original_row < starting_row + n_block; original_row++)

    /******************************************************************/
    /*                                                                */
    /*        Set up outputting dataset                                */
    /*                                                                */
    /******************************************************************/
    if (b_obcold_reconstruction == TRUE)
    {
        sprintf(out_filename, "record_change_x%d_y%d_obcold.dat", current_block_x, current_block_y);
        sprintf(out_fullpath, "%s/obcold/%s", argv[4], out_filename);
        sprintf(out_fullpath_tmp, "%s/obcold/%s.part", argv[4], out_filename);
    }
    else
    {
        if (method == COLD)
        {
            sprintf(out_filename, "record_change_x%d_y%d_cold.dat", current_block_x, current_block_y);
            sprintf(CM_filename, "CM_h%d_v%d.dat", current_block_x, current_block_y);
            sprintf(CM_date_filename, "CM_date_x%d_y%d.dat", current_block_x, current_block_y);
            sprintf(CM_direction_filename, "CM_direction_x%d_y%d.dat", current_block_x, current_block_y);
            sprintf(out_fullpath, "%s/%s", argv[3], out_filename);
            sprintf(out_fullpath_tmp, "%s/%s.part", argv[3], out_filename);
            sprintf(CM_fullpath, "%s/%s", argv[3], CM_filename);
            sprintf(CM_direction_fullpath, "%s/%s", argv[3], CM_direction_filename);
            sprintf(CM_date_fullpath, "%s/%s", argv[3], CM_date_filename);
        }
        else if ((method == SCCD) | (method == SCCDONLINE))
        {
            sprintf(out_filename, "record_change_x%d_y%d_sccd.dat", current_block_x, current_block_y);
            sprintf(out_fullpath, "%s/%s", argv[3], out_filename);
            sprintf(out_fullpath_tmp, "%s/%s.part", argv[3], out_filename);
        }
    }

    fhoutput = fopen(out_fullpath_tmp, "w");

    if (b_outputCM == TRUE)
    {
        fhoutput_cm = fopen(CM_fullpath, "w");
        fhoutput_cm_direction = fopen(CM_direction_fullpath, "w");
        fhoutput_cm_date = fopen(CM_date_fullpath, "w");
    }

    snprintf(msg_str, sizeof(msg_str), "COLD processing for Block_x%d_y%d started\n",
             current_block_x, current_block_y);
    LOG_MESSAGE(msg_str, FUNC_NAME)

    /******************************************************************/
    /*                                                                */
    /*           Begin CCDC scanline processing                       */
    /*                                                                */
    /******************************************************************/
    for (i_col = 0; i_col < block_width * block_height; i_col++)
    {
        if (b_obcold_reconstruction == TRUE)
        {
            rec_cg = malloc(NUM_FC * sizeof(Output_t));
            if (rec_cg == NULL)
            {
                RETURN_ERROR("ERROR allocating rec_cg",
                             FUNC_NAME, FAILURE);
            }
            result = obcold_reconstruction_procedure(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], fmask_buf, sdate,
                                                     valid_scene_count, breakdates_block, num_breakdatemaps,
                                                     i_col + 1, conse, conse, rec_cg, &num_fc);
            // snprintf (msg_str, sizeof(msg_str), "col %d reconstruction calculation finished\n", i_col+1);
            // LOG_MESSAGE (msg_str, FUNC_NAME)
            for (i = 0; i < num_fc; i++)
            {
                result = write_output_binary(fhoutput, rec_cg[i]);
            }
            free(rec_cg);
        }
        else
        {
            if (method == COLD)
            {

                rec_cg = malloc(NUM_FC * sizeof(Output_t));
                if (rec_cg == NULL)
                {
                    RETURN_ERROR("ERROR allocating rec_cg",
                                 FUNC_NAME, FAILURE);
                }
                for (i = 0; i < n_cm_maps; i++)
                {
                    CM_outputs[i] = NA_VALUE;
                    CM_outputs_date[i] = NA_VALUE;
                }

                //                printf("temporal success2 with processid %d\n", process_id);
                //                printf("valid_scene_count_scanline[i_col] is %d;i_col is %d; original_row is %d; probability threshold is %f\n",
                //                       valid_scene_count_scanline[i_col], i_col, original_row, probability_threshold);
                result = cold(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], fmask_buf, sdate, valid_scene_count,
                              i_col + 1, tcg, conse, b_outputCM, starting_date, b_c2, rec_cg, &num_fc, CM_OUTPUT_INTERVAL, CM_outputs,
                              CM_outputs_date, gap_days);

                // snprintf (msg_str, sizeof(msg_str), "pixel %d COLD calculation finished\n", i_col+1);
                // LOG_MESSAGE (msg_str, FUNC_NAME)

                for (i = 0; i < num_fc; i++)
                {
                    result = write_output_binary(fhoutput, rec_cg[i]);
                }

                if (b_outputCM == TRUE)
                {
                    nvals = fwrite(CM_outputs, sizeof(int16), n_cm_maps, fhoutput_cm);
                    if (nvals != n_cm_maps)
                    {
                        RETURN_ERROR("Incorrect amount of data written", FUNC_NAME, ERROR);
                    }
                    nvals = fwrite(CM_outputs_date, sizeof(int8), n_cm_maps, fhoutput_cm_date);
                    if (nvals != n_cm_maps)
                    {
                        RETURN_ERROR("Incorrect amount of data written", FUNC_NAME, ERROR);
                    }
                }

                free(rec_cg);
                // for debug
                // printf("free stage 7 \n");
            }
            else if (method == COLD_FLEX)
            {
                int nbands = 4;
                rec_cg_flex = malloc(NUM_FC * sizeof(Output_t_flex));
                t_cg = X2(nbands, pcg);
                max_t_cg = X2(nbands, 0.99999);
                if (rec_cg_flex == NULL)
                {
                    RETURN_ERROR("ERROR allocating rec_cg_flex",
                                 FUNC_NAME, FAILURE);
                }

                //                printf("temporal success2 with processid %d\n", process_id);
                //                printf("valid_scene_count_scanline[i_col] is %d;i_col is %d; original_row is %d; probability threshold is %f\n",
                //                       valid_scene_count_scanline[i_col], i_col, original_row, probability_threshold);
                for (i = 0; i < valid_scene_count; i++)
                {
                    for (j = 0; j < nbands; j++)
                    {
                        buf_stack[i * nbands + j] = buf[j + 1][i];
                    }
                }

                result = cold_flex(buf_stack, fmask_buf, sdate, nbands, 1, 4, valid_scene_count, i_col + 1, t_cg, max_t_cg, conse, b_outputCM, starting_date, rec_cg_flex, &num_fc, CM_OUTPUT_INTERVAL, CM_outputs,
                                   CM_outputs_date, gap_days);

                // snprintf (msg_str, sizeof(msg_str), "pixel %d COLD calculation finished\n", i_col+1);
                // LOG_MESSAGE (msg_str, FUNC_NAME)
                free(rec_cg_flex);
            }
            else if (method == SCCD_FLEX)
            {
                int nbands = 4;
                t_cg = X2(nbands, pcg);
                max_t_cg = X2(nbands, 0.9999);
                s_rec_cg_flex = malloc(NUM_FC * sizeof(Output_sccd_flex));
                if (s_rec_cg_flex == NULL)
                {
                    RETURN_ERROR("ERROR allocating s_rec_cg_flex",
                                 FUNC_NAME, FAILURE);
                }

                //                printf("temporal success2 with processid %d\n", process_id);
                //                printf("valid_scene_count_scanline[i_col] is %d;i_col is %d; original_row is %d; probability threshold is %f\n",
                //                       valid_scene_count_scanline[i_col], i_col, original_row, probability_threshold);
                for (i = 0; i < valid_scene_count; i++)
                {
                    for (j = 0; j < nbands; j++)
                    {
                        buf_stack[i * nbands + j] = buf[j + 1][i];
                    }
                }

                result = sccd_flex(buf_stack, fmask_buf, sdate, nbands, 1, 4, valid_scene_count, tcg,
                                   max_t_cg, &num_fc, &nrt_mode, s_rec_cg_flex, nrt_model_flex, &num_obs_queue,
                                   obs_queue_flex, min_rmse, conse, b_c2, b_pinpoint, rec_cg_pinpoint_flex,
                                   &num_fc_pinpoint, gate_tcg, t_cg, b_coefs_records,
                                   state_intervaldays, &n_state, state_days, state_ensemble);

                // snprintf (msg_str, sizeof(msg_str), "pixel %d COLD calculation finished\n", i_col+1);
                // LOG_MESSAGE (msg_str, FUNC_NAME)
                free(s_rec_cg_flex);
            }
            else
            {
                for (i = 0; i < n_cm_maps; i++)
                {
                    CM_outputs[i] = NA_VALUE;
                    CM_outputs_date[i] = NA_VALUE;
                }
                result = sccd(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], fmask_buf,
                              sdate, valid_scene_count, tcg, &num_fc, &nrt_mode, s_rec_cg, nrt_model,
                              &num_obs_queue, obs_queue, min_rmse, conse, b_c2, b_pinpoint,
                              rec_cg_pinpoint, &num_fc_pinpoint, gate_tcg, 15.086, b_coefs_records,
                              state_intervaldays, &n_state, state_days, state_ensemble);

                // printf("free stage 9 \n");
                for (i = 0; i < num_fc; i++)
                {
                    result = write_output_binary_sccd(fhoutput, s_rec_cg[i]);
                }

                // printf("free stage 10 \n");
            }
        }
        if (result != SUCCESS)
        {
            fclose(fhoutput);
            RETURN_ERROR("ccd procedure fails \n", FUNC_NAME, FAILURE);
        }
        //            int count;
    } // for(i_col = 0; i_col < block_width * block_height; i_col++)

    fclose(fhoutput);

    if (b_outputCM == TRUE)
    {
        fclose(fhoutput_cm);
        fclose(fhoutput_cm_direction);
        fclose(fhoutput_cm_date);
    }

    if (rename(out_fullpath_tmp, out_fullpath) == 0)
    {
        snprintf(msg_str, sizeof(msg_str), "Block_h%d_v%d finished\n", current_block_x, current_block_y);
        LOG_MESSAGE(msg_str, FUNC_NAME);
    }
    else
    {
        snprintf(msg_str, sizeof(msg_str), "Block_h%d_v%d renaming failed\n", current_block_x, current_block_y);
        LOG_MESSAGE(msg_str, FUNC_NAME);
    }

    /********************************************************/
    /*                   free memory                        */
    /********************************************************/

    free(meta);
    status = free_2d_array((void **)buf);
    if (status != SUCCESS)
    {
        RETURN_ERROR("Freeing memory: buf\n", FUNC_NAME, FAILURE);
    }
    free(buf_stack);

    free(fmask_buf);

    free(sensor_buf);

    free(sdate);
    free(CM_outputs);
    free(CM_outputs_date);
    free(CMdirection_outputs);
    free(csv_row);

    if (b_obcold_reconstruction == TRUE)
    {
        free(breakdates_block);
    }

    free(s_rec_cg);
    free(rec_cg_pinpoint);
    free(rec_cg_pinpoint_flex);
    free(obs_queue);
    free(nrt_model);
    free(nrt_model_flex);
    free(coefs_records);
    free(state_days);
    free(state_ensemble);
    // free(fh);
    return SUCCESS;
}
