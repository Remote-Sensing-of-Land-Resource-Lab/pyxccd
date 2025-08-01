#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
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

/******************************************************************************
MODULE:  get_args
PURPOSE:  Gets the command-line arguments and validates that the required
arguments were specified.
RETURN VALUE:
Type = int
Value           Description
-----           -----------
FAILURE         Error getting the command-line arguments or a command-line
                argument and associated value were not specified
SUCCESS         No errors encountered
HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
11/15/2018  Su Ye            orginal develop
******************************************************************************/
int get_variables_hpc(
    int argc,                      /* I: number of cmd-line args                    */
    char *argv[],                  /* I: string of cmd-line args                    */
    char *in_path,                 /* O: directory locaiton for input data          */
    char *out_path,                /* O: directory location for output files        */
    int *method,                   /* O: 1 - COLD, 2- S-CCD          */
    char *user_mask_path,          /* O: directory location for user_mask           */
    double *probability_threshold, /* O: probability threshold between 0 and 1  */
    int *min_days_conse            /* O: minimum peek days, S-CCD only  */
)
{
    char cwd[MAX_STR_LEN]; // current directory path
    // char cwd[] = "/Documents/QTProjects/build-CCDC_C-Desktop-Debug";
    char var_path[MAX_STR_LEN];
    FILE *var_fp;
    char line1[MAX_STR_LEN], line2[MAX_STR_LEN], line3[MAX_STR_LEN],
        line4[MAX_STR_LEN], line5[MAX_STR_LEN], line6[MAX_STR_LEN];
    char FUNC_NAME[] = "get_variables";

    // when there is no variable command-line argument,
    // use the default variable text path

    if (argc < 2)
    {
        getcwd(cwd, sizeof(cwd));
        // printf("getvariable");
        sprintf(var_path, "%s/%s", cwd, "variables");
    }
    else if (argc == 2)
    {
        strcpy(var_path, argv[1]);
    }
    else
    {
        RETURN_ERROR("only the argument for variable file "
                     "path can be added in the command line",
                     FUNC_NAME, ERROR);
    }

    var_fp = fopen(var_path, "r");

    if (var_fp == NULL)
    {
        RETURN_ERROR("There is no variable file in the bin folder", FUNC_NAME, ERROR);
    }

    fscanf(var_fp, "%s\n", line1);
    // in_path = strchr(line2, '=') + 1;getvariable
    strcpy(in_path, strchr(line1, '=') + 1);

    fscanf(var_fp, "%s\n", line2);
    strcpy(out_path, strchr(line2, '=') + 1);

    //    fscanf(var_fp, "%s\n", line4);
    //    *n_cores = atoi(strchr(line4, '=') + 1);

    //    fscanf(var_fp, "%s\n", line5);
    //    *n_nodes = atoi(strchr(line5, '=') + 1);
    fscanf(var_fp, "%s\n", line3);
    *method = atoi(strchr(line3, '=') + 1);

    fscanf(var_fp, "%s\n", line4);
    strcpy(user_mask_path, strchr(line4, '=') + 1);

    //    fscanf(var_fp, "%s\n", line7);
    //    *row = atoi(strchr(line7, '=') + 1);

    //    fscanf(var_fp, "%s\n", line8);
    //    *col = atoi(strchr(line8, '=') + 1);

    fscanf(var_fp, "%s\n", line5);
    *probability_threshold = atof(strchr(line5, '=') + 1);

    fscanf(var_fp, "%s\n", line6);
    *min_days_conse = atoi(strchr(line6, '=') + 1);
    // printf("debug: step4\n");

    // check if variable inputs are complete
    if (*method > 2 || *method < 0)
    {
        RETURN_ERROR("Variables error: method variable must be either 1 or 2", FUNC_NAME, ERROR);
    }
    if (*in_path == '\0')
    {
        RETURN_ERROR("Variables error: 'in_path' "
                     "cannot be empty in 'Variables' file!",
                     FUNC_NAME, ERROR);
    }
    if (*out_path == '\0')
    {
        RETURN_ERROR("Variables error: 'out_path' "
                     "cannot be empty in 'Variables' file!",
                     FUNC_NAME, ERROR);
    }

    //    fscanf(var_fp, "%s\n", line);
    //    fscanf(var_fp, "%s\n", line);
    //    fscanf(var_fp, "%s\n", line);
    //    fscanf(var_fp, "in_path=%s\n", in_path);
    //    fscanf(var_fp, "out_path=%s\n", out_path);
    //    fscanf(var_fp, "n_cores=%d\n", n_cores);
    //    fscanf(var_fp, "user_mask_path=%s\n", user_mask_path);
    //    fscanf(var_fp, "row=%d\n", row);
    //    fscanf(var_fp, "col=%d\n", col);

    fclose(var_fp);
    return SUCCESS;
}

int sccd_hpc(
    short int **buf,       /* I/O:  pixel-based time series           */
    short int *fmask_buf,  /* I:  mask-based time series              */
    int *valid_date_array, /* I: valid date time series               */
    int valid_num_scenes,  /* I: number of valid scenes under cfmask fill counts  */
    Output_t_sccd *rec_cg, /* O: outputted structure for CCDC results    */
    int *num_fc,           /* O: number of fitting curves                       */
    int num_samples,       /* I: column number per scanline                    */
    int col_pos,           /* I: column position of current processing pixel   */
    int row_pos,           /* I: raw position of current processing pixel  */
    double probability_threshold,
    int min_days_conse,
    short int *sensor_buf)
{
    /* hpc version temporally set these parameter as fixed */
    bool b_fastmode = TRUE;
    char *states_output_dir;
    int training_type = 0;      /* '0' is normal */
    int monitorwindow_lowerlin; /* for training process*/
    int monitorwindow_upperlim; /* for training process*/
    int n_focus_variable = DEFAULT_N_FOCUS_VARIABLE;
    int n_total_variable = TOTAL_IMAGE_BANDS;
    int focus_blist[TOTAL_IMAGE_BANDS] = {1, 2, 3, 4, 5};
    bool NDVI_INCLUDED = FALSE;
    bool NBR_INCLUDED = FALSE;
    bool RGI_INCLUDED = FALSE;
    bool TCTWETNESS_INCLUDED = FALSE;
    bool TCTGREENNESS_INCLUDED = FALSE;
    bool EVI_INCLUDED = FALSE;
    bool DI_INCLUDED = FALSE;
    bool NDMI_INCLUDED = FALSE;
    bool b_landspecific = FALSE;
    short int auxval = 0;
    /* hpc version temporally set these parameter as fixed */

    int clear_sum = 0;  /* Total number of clear cfmask pixels          */
    int water_sum = 0;  /* counter for cfmask water pixels.             */
    int shadow_sum = 0; /* counter for cfmask shadow pixels.            */
    int sn_sum = 0;     /* Total number of snow cfmask pixels           */
    int cloud_sum = 0;  /* counter for cfmask cloud pixels.             */
    double sn_pct;      /* Percent of snow pixels.                      */
    int status;
    int *id_range;
    int i, j, k;
    char FUNC_NAME[] = "sccd";
    int result;

    id_range = (int *)malloc(valid_num_scenes * sizeof(int));
    // printf("valid_num_scenes is %d\n", valid_num_scenes);

    status = preprocessing(buf, fmask_buf, &valid_num_scenes, id_range, &clear_sum,
                           &water_sum, &shadow_sum, &sn_sum, &cloud_sum);

    if (status != SUCCESS)
    {
        RETURN_ERROR("Error for preprocessing.", FUNC_NAME, ERROR);
    }

    // clear_pct is not used anymore in V13.01
    // clr_pct = (double) clear_sum / (double) (valid_num_scenes);

    sn_pct = (double)sn_sum / (double)(sn_sum + clear_sum + 0.01);

    for (i = 0; i < NUM_FC; i++)
    {
        rec_cg[i].pos = num_samples * (row_pos - 1) + col_pos;

        // the below is to initialize
        rec_cg[i].category = -9999;
        // rec_cg[i].land_type = 0;

        for (j = 0; j < TOTAL_IMAGE_BANDS + TOTAL_INDICES; j++)
        {
            rec_cg[i].obs_disturb[j] = NA_VALUE;
            rec_cg[i].rmse[j] = NA_VALUE;
            rec_cg[i].magnitude[j] = NA_VALUE;
            for (k = 0; k < FLEX_SCCD_NUM_C - 1; k++)
            {
                rec_cg[i].state_disturb[j][k] = NA_VALUE;
                rec_cg[i].coefs[j][k] = NA_VALUE;
            }
            rec_cg[i].coefs[j][FLEX_SCCD_NUM_C - 1] = NA_VALUE;
        }
    }

    if (clear_sum < N_TIMES * MAX_NUM_C)
    {
        result = sccd_inefficientobs_procedure(valid_num_scenes, valid_date_array, buf,
                                               fmask_buf, id_range, sn_pct, rec_cg, num_fc);
    }
    else
    {

        /**************************************************************/
        /*                                                            */
        /* standard_procedure for CCD                                 */
        /*                                                            */
        /**************************************************************/

        result = sccd_stand_procedure(valid_num_scenes, valid_date_array, buf,
                                      fmask_buf, id_range, rec_cg, num_fc,
                                      states_output_dir, b_fastmode, probability_threshold,
                                      min_days_conse, training_type, monitorwindow_lowerlin,
                                      monitorwindow_upperlim, sensor_buf, n_focus_variable,
                                      n_total_variable, focus_blist, NDVI_INCLUDED,
                                      NBR_INCLUDED, RGI_INCLUDED, TCTWETNESS_INCLUDED, TCTGREENNESS_INCLUDED,
                                      EVI_INCLUDED, DI_INCLUDED, NDMI_INCLUDED, b_landspecific, auxval);
    }

    // printf("free stage 7 \n");
    // printf("num_fc is %d\n", *num_fc);
    free(id_range);
    id_range = NULL;
    // printf("free stage 8 \n");
    if (result == SUCCESS)
    {
        return (SUCCESS);
    }
    else
    {
        return (FAILURE);
    }
}

int main(int argc, char *argv[])
{

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
    /* inputted argument */
    int mode; /* CCD detection mode
              3: whole images; 1: pixel-based;
              2: scanline-based*/
    char in_path[MAX_STR_LEN];
    char out_path[MAX_STR_LEN];
    int n_cores;
    int row;
    int col;
    int METHOD;
    char mask_path[MAX_STR_LEN];
    int min_days_conse;
    double probability_threshold;
    int output_mode;
    // bool verbose = TRUE;
    bool b_fastmode;
    bool b_outputCSV; /* output point-based time series csv */
    int monitorwindow_lowerlin;
    int monitorwindow_upperlim;
    int training_type; /* for training process*/
    int bandselection_bit;
    char classification_config[MAX_STR_LEN];

    char scene_list_filename[] = "scene_list.txt"; /* file name containing list of input sceneIDs */
    char msg_str[MAX_STR_LEN];                     /* Input data scene name                 */
    int i, j;                                      /* Loop counters                         */
    char scene_list_directory[MAX_STR_LEN];        /* full directory of scene list*/
    int status;                                    /* Return value from function call       */
    char FUNC_NAME[] = "main";                     /* For printing error messages           */

    int *sdate;          /* Pointer to list of acquisition dates  */
    char **scene_list;   /* 2-D array for list of scene IDs       */
    char *scene_list_1d; /* 2-D array for list of scene IDs       */

    int num_scenes;           /* Number of input scenes defined        */
    Input_meta_t *meta;       /* Structure for ENVI metadata hdr info  */
    char tmpstr[MAX_STR_LEN]; /* char string for text manipulation      */

    time_t now; /* For logging the start, stop, and some     */
    int result;

    int num_fc; /* the number of functional curve        */
    char out_filename[MAX_STR_LEN];
    char errmsg[MAX_STR_LEN]; /* for printing error text to the log.  */

    // block_num = (int)meta->lines / threads;

    /**************************************************************************/
    /*                   Parallel scanline processing                         */
    /**************************************************************************/

    int n_process;
    int n_block;
    int process_id;
    int n_remain_line;

    int **valid_date_array_scanline;
    short int **fmask_buf_scanline; /* fmask buf, valid pixels only*/
    int *valid_scene_count_scanline;
    short int **buf;      /* This is the image bands buffer, valid pixel only*/
    short int *tmp_buf;   /* This is the image bands buffer, valid pixel only*/
    bool isUserMaskExist; /*  if user mask exists or not        */
    char *user_mask_scanline;
    bool user_mask_hasvalidcount = FALSE;
    char out_fullpath[MAX_STR_LEN] = "";
    // int n_working_process; /* total processes - 1 (master process)*/
    int i_col;
    int *num_fc_scanline;
    MPI_File fh[MAX_SCENE_LIST];
    FILE *f_bip[MAX_SCENE_LIST];
    MPI_File fusermask_bip;
    MPI_File fhoutput;
    FILE *fdoutput;
    FILE *fd;
    char img_filename[MAX_STR_LEN]; /* file name constructed from sceneID   */
    int k, b;
    MPI_Status mpi_status;
    short int **tmp_buf_2d;    /* This is the image bands buffer, valid pixel only*/
    short int *tmp_fmask_buf;  /* fmask buf, valid pixels only          */
    int *tmp_valid_date_array; /* Sdate array after cfmask filtering    */
    int n_rows;
    int n_cols;
    Output_t *rec_cg;
    Output_t_sccd *s_rec_cg;
    int n_first_block; // the row number for the first block, we assigned less rows as the primary node took charge of communication
    int new_n_block;   // the row number for each block
    int starting_row;  // the starting row for each block
    short int tmp_sensor;
    int interval;
    short int *sensor_buf;

    /* create custom type for CCDC output reccg_type*/
    Output_t rec_cg_scanline_Sample[1]; /* for calculating disp */
    MPI_Datatype mpi_reccg_type;
    MPI_Datatype mpi_elem_type[11] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                      MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    int blocklen[11] = {1, 1, 1, 1, 1, 1, 1, 1,
                        TOTAL_IMAGE_BANDS * NUM_COEFFS, TOTAL_IMAGE_BANDS, TOTAL_IMAGE_BANDS};
    MPI_Aint disp[11];

    /* s-ccd */
    Output_t_sccd rec_cg_scanline_Sample_sccd[1]; /* for calculating disp */
    MPI_Datatype mpi_reccg_type_sccd;
    MPI_Datatype mpi_elem_type_sccd[14] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                           MPI_SHORT, MPI_SHORT, MPI_INT, MPI_INT, MPI_DOUBLE,
                                           MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    int blocklen_sccd[14] = {1, 1, 1, 1, 1, 1, 1, 1, 1, TOTAL_IMAGE_BANDS * FLEX_SCCD_NUM_C,
                             TOTAL_IMAGE_BANDS, TOTAL_IMAGE_BANDS * (FLEX_SCCD_NUM_C - 1),
                             TOTAL_IMAGE_BANDS, TOTAL_IMAGE_BANDS};
    MPI_Aint disp_sccd[14];

    //    tmp_buf = (short int **) allocate_2d_array (TOTAL_IMAGE_BANDS, num_scenes, sizeof (short int));
    //    if(tmp_buf == NULL)
    //    {
    //        RETURN_ERROR ("Allocating tmp_buf", FUNC_NAME, FAILURE);
    //    }

    // MPI_Init(&argc, &argv);
    MPI_Init(NULL, NULL);

    disp[0] = 0;
    disp[1] = sizeof(rec_cg_scanline_Sample[0].t_start);
    disp[2] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start);
    disp[3] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
              sizeof(rec_cg_scanline_Sample[0].t_break);
    disp[4] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
              sizeof(rec_cg_scanline_Sample[0].t_break) + sizeof(rec_cg_scanline_Sample[0].pos);
    disp[5] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
              sizeof(rec_cg_scanline_Sample[0].t_break) + sizeof(rec_cg_scanline_Sample[0].pos) +
              sizeof(rec_cg_scanline_Sample[0].num_obs);
    disp[6] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
              sizeof(rec_cg_scanline_Sample[0].t_break) + sizeof(rec_cg_scanline_Sample[0].pos) +
              sizeof(rec_cg_scanline_Sample[0].num_obs) + sizeof(rec_cg_scanline_Sample[0].category);
    disp[7] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
              sizeof(rec_cg_scanline_Sample[0].t_break) + sizeof(rec_cg_scanline_Sample[0].pos) +
              sizeof(rec_cg_scanline_Sample[0].num_obs) + sizeof(rec_cg_scanline_Sample[0].category) +
              sizeof(rec_cg_scanline_Sample[0].t_confirmed);
    disp[8] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
              sizeof(rec_cg_scanline_Sample[0].t_break) + sizeof(rec_cg_scanline_Sample[0].pos) +
              sizeof(rec_cg_scanline_Sample[0].num_obs) + sizeof(rec_cg_scanline_Sample[0].category) +
              sizeof(rec_cg_scanline_Sample[0].t_confirmed) + sizeof(rec_cg_scanline_Sample[0].change_prob);
    disp[9] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
              sizeof(rec_cg_scanline_Sample[0].t_break) + sizeof(rec_cg_scanline_Sample[0].pos) +
              sizeof(rec_cg_scanline_Sample[0].num_obs) + sizeof(rec_cg_scanline_Sample[0].category) +
              sizeof(rec_cg_scanline_Sample[0].t_confirmed) + sizeof(rec_cg_scanline_Sample[0].change_prob) +
              sizeof(rec_cg_scanline_Sample[0].coefs);
    disp[10] = sizeof(rec_cg_scanline_Sample[0].t_end) + sizeof(rec_cg_scanline_Sample[0].t_start) +
               sizeof(rec_cg_scanline_Sample[0].t_break) + sizeof(rec_cg_scanline_Sample[0].pos) +
               sizeof(rec_cg_scanline_Sample[0].num_obs) + sizeof(rec_cg_scanline_Sample[0].category) +
               sizeof(rec_cg_scanline_Sample[0].t_confirmed) + sizeof(rec_cg_scanline_Sample[0].change_prob) +
               sizeof(rec_cg_scanline_Sample[0].coefs) + sizeof(rec_cg_scanline_Sample[0].rmse);

    disp_sccd[0] = 0;
    disp_sccd[1] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start);
    disp_sccd[2] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end);
    disp_sccd[3] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break);
    disp_sccd[4] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos);
    disp_sccd[5] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs);
    disp_sccd[6] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category);
    disp_sccd[7] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category) + sizeof(rec_cg_scanline_Sample_sccd[0].land_type);
    disp_sccd[8] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category) + sizeof(rec_cg_scanline_Sample_sccd[0].land_type) + sizeof(rec_cg_scanline_Sample_sccd[0].t_confirmed);
    disp_sccd[9] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category) + sizeof(rec_cg_scanline_Sample_sccd[0].land_type) + sizeof(rec_cg_scanline_Sample_sccd[0].t_confirmed) + sizeof(rec_cg_scanline_Sample_sccd[0].change_prob);
    disp_sccd[10] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category) + sizeof(rec_cg_scanline_Sample_sccd[0].land_type) + sizeof(rec_cg_scanline_Sample_sccd[0].t_confirmed) + sizeof(rec_cg_scanline_Sample_sccd[0].change_prob) + sizeof(rec_cg_scanline_Sample_sccd[0].coefs);
    disp_sccd[11] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category) + sizeof(rec_cg_scanline_Sample_sccd[0].land_type) + sizeof(rec_cg_scanline_Sample_sccd[0].t_confirmed) + sizeof(rec_cg_scanline_Sample_sccd[0].change_prob) + sizeof(rec_cg_scanline_Sample_sccd[0].coefs) + sizeof(rec_cg_scanline_Sample_sccd[0].obs_disturb);
    disp_sccd[12] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category) + sizeof(rec_cg_scanline_Sample_sccd[0].land_type) + sizeof(rec_cg_scanline_Sample_sccd[0].t_confirmed) + sizeof(rec_cg_scanline_Sample_sccd[0].change_prob) + sizeof(rec_cg_scanline_Sample_sccd[0].coefs) + sizeof(rec_cg_scanline_Sample_sccd[0].obs_disturb) + sizeof(rec_cg_scanline_Sample_sccd[0].state_disturb);
    disp_sccd[13] = sizeof(rec_cg_scanline_Sample_sccd[0].t_start) + sizeof(rec_cg_scanline_Sample_sccd[0].t_end) + sizeof(rec_cg_scanline_Sample_sccd[0].t_break) + sizeof(rec_cg_scanline_Sample_sccd[0].pos) + sizeof(rec_cg_scanline_Sample_sccd[0].num_obs) + sizeof(rec_cg_scanline_Sample_sccd[0].category) + sizeof(rec_cg_scanline_Sample_sccd[0].land_type) + sizeof(rec_cg_scanline_Sample_sccd[0].t_confirmed) + sizeof(rec_cg_scanline_Sample_sccd[0].change_prob) + sizeof(rec_cg_scanline_Sample_sccd[0].coefs) + sizeof(rec_cg_scanline_Sample_sccd[0].obs_disturb) + sizeof(rec_cg_scanline_Sample_sccd[0].state_disturb) + sizeof(rec_cg_scanline_Sample_sccd[0].rmse);

    MPI_Type_create_struct(14, blocklen_sccd, disp_sccd, mpi_elem_type_sccd, &mpi_reccg_type_sccd);
    MPI_Type_commit(&mpi_reccg_type_sccd);
    MPI_Type_create_struct(11, blocklen, disp, mpi_elem_type, &mpi_reccg_type);
    MPI_Type_commit(&mpi_reccg_type);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &n_process);

    /* get this process id */
    MPI_Comm_rank(comm, &process_id);

    // debug mode here:
    //    printf("argc = %d\n", argc);
    //    for(i = 0; i< argc; i++)
    //    {
    //        printf("%s\n",argv[i]);
    //    }
    //    printf("nprocess = %d\n", n_process);
    //    printf("process_id = %d\n", process_id);

    // master process
    scene_list_1d = (char *)malloc(MAX_SCENE_LIST * ARD_STR_LEN * sizeof(char));
    if (scene_list_1d == NULL)
    {
        RETURN_ERROR("ERROR allocating scene_list_1d",
                     FUNC_NAME, FAILURE);
    }

    meta = (Input_meta_t *)malloc(sizeof(Input_meta_t));
    if (meta == NULL)
    {
        RETURN_ERROR("ERROR allocating meta",
                     FUNC_NAME, FAILURE);
    }

    scene_list = (char **)allocate_2d_array(MAX_SCENE_LIST, ARD_STR_LEN,
                                            sizeof(char));
    if (scene_list == NULL)
    {
        RETURN_ERROR("ERROR allocating scene_list memory", FUNC_NAME, FAILURE);
    }

    if (process_id == 0)
    { /**************************************************************/
        /*                                                            */
        /*   record the start time of just the CDCD         */
        /*     algorithm.  Up until here, it has all just been        */
        /*     setting it up......                                    */
        /*                                                            */
        /**************************************************************/
        time(&now); /*     intermediate times.                   */
        snprintf(msg_str, sizeof(msg_str), "CCDC start_time\n", ctime(&now));

        // printf("debug: step1\n");
        LOG_MESSAGE(msg_str, FUNC_NAME);

        printf("The actual assigned process number is %d\n", n_process);

        // printf("debug: step2\n");
        /**************************************************************/
        /*                                                            */
        /*   read CCD variable                                        */
        /*                                                            */
        /**************************************************************/
        result = get_variables_hpc(argc, argv, in_path, out_path,
                                   &METHOD, mask_path, &probability_threshold,
                                   &min_days_conse);
        if (result == ERROR)
        {
            RETURN_ERROR("CCDC procedure fails. The program stops!", FUNC_NAME, FAILURE);
        }
        // printf("debug: step3\n");

        //        if(n_process != n_cores * n_nodes)
        //        {
        //            printf("Warning: the assigned core number is not equal to the request core "
        //                   "number (the assigned core number is %d, "
        //                   "the request process number is %d)", n_process, n_cores * n_nodes);
        //        }

        sprintf(scene_list_directory, "%s/%s", in_path, scene_list_filename);

        if (access(scene_list_directory, F_OK) != 0) /* File does not exist */
        {
            status = create_scene_list(in_path, &num_scenes, scene_list_filename);
            if (status != SUCCESS)
                RETURN_ERROR("Running create_scene_list file", FUNC_NAME, FAILURE);
        }
        else
        {
            num_scenes = MAX_SCENE_LIST;
        }

        /**************************************************************/
        /*                                                            */
        /* Fill the scene list array with full path names.            */
        /*                                                            */
        /**************************************************************/
        fd = fopen(scene_list_directory, "r");
        if (fd == NULL)
        {
            RETURN_ERROR("Opening scene_list file", FUNC_NAME, FAILURE);
        }

        if (scene_list == NULL)
        {
            RETURN_ERROR("ERROR allocating scene_list memory", FUNC_NAME, FAILURE);
        }

        for (i = 0; i < num_scenes; i++)
        {
            if (fscanf(fd, "%s", tmpstr) == EOF)
                break;
            strcpy(scene_list[i], tmpstr);
        }
        num_scenes = i;

        fclose(fd);

        sdate = (int *)malloc(num_scenes * sizeof(int));

        if (sdate == NULL)
        {
            RETURN_ERROR("ERROR allocating sdate memory", FUNC_NAME, FAILURE);
        }

        /**************************************************************/
        /*                                                            */
        /* Sort scene_list based on year & julian_day, then do the    */
        /* swath filter, but read it above first.                     */
        /*                                                            */
        /**************************************************************/

        status = sort_scene_based_on_year_doy_row(scene_list, num_scenes, sdate, ENVI_FORMAT);
        if (status != SUCCESS)
        {
            RETURN_ERROR("Calling sort_scene_based_on_year_jday",
                         FUNC_NAME, FAILURE);
        }

        /* replicate scene_list as 1-d array to get them transported by mpi*/

        for (i = 0; i < num_scenes; i++)
        {
            for (j = 0; j < ARD_STR_LEN; j++)
                scene_list_1d[i * ARD_STR_LEN + j] = scene_list[i][j];
        }

        status = read_envi_header(in_path, scene_list[0], meta);
        if (status != SUCCESS)
        {
            RETURN_ERROR("Calling read_envi_header",
                         FUNC_NAME, FAILURE);
        }

        // n_process = 48;

        n_cols = meta->samples;
        n_rows = meta->lines;
        if (n_process > 1)
        {
            n_first_block = (int)(meta->lines / n_process * 0.9);
            n_block = (int)((meta->lines - n_first_block) / (n_process - 1));
        }
        else
        {
            n_first_block = meta->lines;
            n_block = meta->lines;
        }

        n_remain_line = meta->lines - n_first_block - n_block * (n_process - 1);

        printf("The row number for primary node is %d; the standard row number per block is %d; the remaining row is %d \n", n_first_block, n_block, n_remain_line);

        interval = n_process - 1;

        /* Broadcasts variables from the master process  */
        int mpi_args[8] = {n_block, n_cols, n_rows, num_scenes, n_remain_line, n_first_block, interval, METHOD};
        // printf("debug: step2\n");
        /* broadcase mpi args */
        MPI_Bcast(mpi_args, 8, MPI_INT, 0, MPI_COMM_WORLD);
        // printf("debug: step3\n");

        MPI_Bcast(&probability_threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* broadcase scene list */
        MPI_Bcast(scene_list_1d, num_scenes * ARD_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
        // printf("debug: step4\n");

        MPI_Bcast(sdate, num_scenes, MPI_INT, 0, MPI_COMM_WORLD);

        // MPI_Bcast(&METHOD, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(in_path, MAX_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        MPI_Bcast(out_path, MAX_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        MPI_Bcast(mask_path, MAX_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        // printf("debug: step5\n");

        // MPI_Isend(sdate, num_scenes, MPI_INT, i, 9, MPI_COMM_WORLD, &reqs[0]);
        //        MPI_Ssend(buf, TOTAL_IMAGE_BANDS * meta->samples * num_scenes, MPI_INT16_T, i, 1, MPI_COMM_WORLD, &reqs[0]);
        //        MPI_Ssend(fmask_buf_scanline, meta->samples * num_scenes, MPI_INT16_T, i, 2, MPI_COMM_WORLD, &reqs[1]);
        //        MPI_Ssend(valid_scene_count_scanline, 1, MPI_INT, i, 3, MPI_COMM_WORLD, &reqs[2]);
        //        MPI_Ssend(num_scenes, 1, MPI_INT, i, 4, MPI_COMM_WORLD, &reqs[3]);
        //        MPI_Ssend(valid_date_array_scanline, meta->samples * num_scenes, MPI_INT, i, 5, MPI_COMM_WORLD, &reqs[4]);

        //        MPI_Waitall(5, reqs, MPI_STATUSES_IGNORE);
    }
    else
    {
        int mpi_args[8];
        MPI_Bcast(mpi_args, 8, MPI_INT, 0, MPI_COMM_WORLD);
        n_block = mpi_args[0];
        n_cols = mpi_args[1];
        n_rows = mpi_args[2];
        num_scenes = mpi_args[3];
        n_remain_line = mpi_args[4];
        n_first_block = mpi_args[5];
        interval = mpi_args[6];
        METHOD = mpi_args[7];
        MPI_Bcast(&probability_threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        sdate = (int *)malloc(num_scenes * sizeof(int));

        if (sdate == NULL)
        {
            RETURN_ERROR("ERROR allocating sdate memory", FUNC_NAME, FAILURE);
        }

        MPI_Bcast(scene_list_1d, num_scenes * ARD_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
        // printf("debug: step4\n");

        MPI_Bcast(sdate, num_scenes, MPI_INT, 0, MPI_COMM_WORLD);

        // MPI_Bcast(&METHOD, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(in_path, MAX_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        MPI_Bcast(out_path, MAX_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        MPI_Bcast(mask_path, MAX_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        for (i = 0; i < num_scenes; i++)
        {
            for (j = 0; j < ARD_STR_LEN; j++)
                scene_list[i][j] = scene_list_1d[i * ARD_STR_LEN + j];
        }
    }

    // for debug
    //    if(process_id == 1)
    //    {
    //        for (i = 0; i < num_scenes; i++)
    //        {
    //            for(j = 0; j < ARD_STR_LEN; j++)
    //                printf("%c", scene_list[i][j]);
    //            printf("\n");

    //        }

    //    }

    buf = (short int **)allocate_2d_array(TOTAL_IMAGE_BANDS, num_scenes * n_cols, sizeof(short int));
    if (buf == NULL)
    {
        RETURN_ERROR("Allocating buf", FUNC_NAME, FAILURE);
    }

    tmp_buf_2d = (short int **)allocate_2d_array(TOTAL_IMAGE_BANDS, num_scenes, sizeof(short int));
    if (tmp_buf_2d == NULL)
    {
        RETURN_ERROR("ERROR allocating tmp_buf_2d",
                     FUNC_NAME, FAILURE);
    }

    fmask_buf_scanline = (short int **)allocate_2d_array(n_cols, num_scenes, sizeof(short int));
    if (fmask_buf_scanline == NULL)
    {
        RETURN_ERROR("Allocating fmask_buf_scanline", FUNC_NAME, FAILURE);
    }

    valid_scene_count_scanline = (int *)malloc(n_cols * sizeof(int));
    if (valid_scene_count_scanline == NULL)
    {
        RETURN_ERROR("Allocating valid_scene_count_scanline", FUNC_NAME, FAILURE);
    }

    valid_date_array_scanline = (int **)allocate_2d_array(n_cols, num_scenes, sizeof(int));
    if (valid_date_array_scanline == NULL)
    {
        RETURN_ERROR("Allocating valid_date_array_scanline", FUNC_NAME, FAILURE);
    }

    sensor_buf = (short int *)malloc(num_scenes * sizeof(short int));
    if (sensor_buf == NULL)
    {
        RETURN_ERROR("Allocating sensor_buf fails", FUNC_NAME, FAILURE);
    }
    //    scene_list = (char **) allocate_2d_array (MAX_SCENE_LIST, ARD_STR_LEN,
    //                                                       sizeof (char));
    //    if(scene_list == NULL)
    //    {
    //        RETURN_ERROR ("Allocating scene_list", FUNC_NAME, FAILURE);
    //    }

    /**************************************************************/
    /*                                                            */
    /* Now that we know the actual number of scenes, allocate     */
    /* memory for date array.                                     */
    /*                                                            */
    /**************************************************************/

    num_fc_scanline = (int *)malloc(n_cols * sizeof(short int));
    if (num_fc_scanline == NULL)
    {
        RETURN_ERROR("ERROR allocating num_fc_scanline",
                     FUNC_NAME, FAILURE);
    }

    user_mask_scanline = (char *)malloc(n_cols * sizeof(char));
    if (user_mask_scanline == NULL)
    {
        RETURN_ERROR("ERROR allocating user_mask_scanline",
                     FUNC_NAME, FAILURE);
    }

    tmp_buf = malloc(sizeof(short int) * n_cols * TOTAL_BANDS);
    if (tmp_buf == NULL)
    {
        RETURN_ERROR("ERROR allocating tmp_buf",
                     FUNC_NAME, FAILURE);
    }

    // process_id = n_remain_line;

    if (process_id == 0)
    {
        new_n_block = n_first_block;
        starting_row = 0;
    }
    else if (process_id < n_remain_line + 1)
    {
        new_n_block = n_block + 1;
        starting_row = n_first_block + new_n_block * (process_id - 1);
        // prev_n_block = process_id * n_block;
    }
    else
    {
        new_n_block = n_block;
        starting_row = n_first_block + (n_block + 1) * n_remain_line + (process_id - 1 - n_remain_line) * n_block;
    }
    //    else
    //    {
    //        prev_n_block = n_remain_line * (n_block + 1) +
    //                (process_id - n_remain_line) * n_block;
    //    }

    //    n_block = 0;
    //    process_id = 0;
    //    prev_n_block = process_id * n_block;

    /*********************************************/
    /*              mpi loop                     */
    /*********************************************/
    // debug modes
    // printf("process_id = %d; n_block = %d; prev_n_block = %d \n", process_id, n_block, prev_n_block);
    // prev_n_block = 119;
    for (i = 0; i < num_scenes; i++)
    {
        sprintf(img_filename, "%s/%s/%s_MTLstack", in_path, scene_list[i], scene_list[i]);
        f_bip[i] = open_raw_binary(img_filename, "rb");
        // result = MPI_File_open(MPI_COMM_SELF, img_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh[i]);
        if (result != SUCCESS)
        {
            sprintf(errmsg, "error in mpi opening %s\n", scene_list[i]);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }
    }

    int current_row;
    // for(current_row = starting_row; current_row < starting_row + new_n_block; current_row++)
    for (j = 0; j < new_n_block; j++)
    {

        // for debug
        //        current_row = 58;
        // j = new_n_block - 1;
        if (process_id == 0)
            current_row = j;
        else
            current_row = n_first_block - 1 + process_id + j * interval;

        for (i = 0; i < n_cols; i++)
        {
            valid_scene_count_scanline[i] = 0;
        }

        /******************************************************************/
        /*                                                                */
        /*   first check if the row is not covered by mask                */
        /*                                                                */
        /******************************************************************/
        if (*mask_path == '\0')
        {
            isUserMaskExist = FALSE;
        }
        else
        {
            isUserMaskExist = TRUE;
        }

        if (isUserMaskExist)
        {
            MPI_File_open(MPI_COMM_SELF, mask_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fusermask_bip);
            // fusermask_bip = open_raw_binary(mask_path,"rb");

            if (fusermask_bip == NULL)
            {
                RETURN_ERROR("Opening user mask fails", FUNC_NAME, FAILURE);
            }

            MPI_File_seek(fusermask_bip, current_row * n_cols * sizeof(char), SEEK_SET);

            MPI_File_read(fusermask_bip, user_mask_scanline, n_cols, MPI_INT8_T, &mpi_status);

            MPI_File_close(&fusermask_bip);

            for (i = 0; i < n_cols; i++)
            {
                if (user_mask_scanline[i] != 0)
                {
                    user_mask_hasvalidcount = TRUE;
                    break;
                }
            }

            /* skip if there are not avaiable pixels in this scanline */
            if (user_mask_hasvalidcount == FALSE)
            {

                // free_2d_array ((void **) fp_bip);
                continue;
            }
        }

        /******************************************************************/
        /*                                                                */
        /*       if the row has pixels not masked out                     */
        /*                                                                */
        /******************************************************************/
        sprintf(out_filename, "record_change_row%d.dat", current_row + 1);

        snprintf(msg_str, sizeof(msg_str), "Row %d processing started\n", current_row + 1);
        LOG_MESSAGE(msg_str, FUNC_NAME)

        sprintf(out_fullpath, "%s/%s", out_path, out_filename);
        // MPI_File_open(MPI_COMM_SELF, out_fullpath, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fhoutput);
        fdoutput = fopen(out_fullpath, "w");

        /******************************************************************/
        /*                                                                */
        /* Read the image bands for this scene.                           */
        /*                                                                */
        /******************************************************************/

        for (i = 0; i < num_scenes; i++)
        {
            // printf("num_scenes is %d with proccess id %d\n", num_scenes, process_id);
            if (scene_list[i][2] == '4' || scene_list[i][2] == '5')
                tmp_sensor = LANDSAT45_TM;
            else if (scene_list[i][2] == '7')
                tmp_sensor = LANDSAT7_ETM;
            else if (scene_list[i][2] == '8')
                tmp_sensor = LANDSAT8_OLI;

            // MPI_File_seek(fh[i], (current_row * n_cols * TOTAL_BANDS) * sizeof(short int), MPI_SEEK_SET);
            // MPI_File_read(fh[i], tmp_buf, TOTAL_BANDS * n_cols, MPI_SHORT, &mpi_status);
            fseek(f_bip[i], (current_row * n_cols * TOTAL_BANDS) * sizeof(short int), SEEK_SET);
            fread(tmp_buf, sizeof(short int), TOTAL_BANDS * n_cols, f_bip[i]);

            if (tmp_buf == NULL)
            {
                sprintf(errmsg, "error reading %d scene, %d row\n", i, current_row + 1);
                RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
            }

            for (k = 0; k < n_cols; k++)
            {
                if (tmp_buf[k * TOTAL_BANDS + TOTAL_BANDS - 1] < CFMASK_FILL)
                {
                    for (b = 0; b < TOTAL_IMAGE_BANDS; b++)
                    {
                        // printf("%d\n", valid_scene_count[k]);
                        buf[b][k * num_scenes + valid_scene_count_scanline[k]] = tmp_buf[k * TOTAL_BANDS + b];
                    }
                    fmask_buf_scanline[k][valid_scene_count_scanline[k]] = (short int)tmp_buf[k * TOTAL_BANDS + TOTAL_BANDS - 1];
                    valid_date_array_scanline[k][valid_scene_count_scanline[k]] = sdate[i];
                    sensor_buf[valid_scene_count_scanline[k]] = tmp_sensor;
                    valid_scene_count_scanline[k] = valid_scene_count_scanline[k] + 1;
                }
            }
        }

        /******************************************************************/
        /*                                                                */
        /*           Begin CCDC scanline processing                       */
        /*                                                                */
        /******************************************************************/
        int offset = 0;
        int size;
        // MPI_Type_size(mpi_reccg_type, &size);
        for (i_col = 0; i_col < n_cols; i_col++)
        {
            // for debug
            // printf("i_col = %d with processid %d\n", i_col, process_id);
            if (isUserMaskExist)
            {
                if (user_mask_scanline[i_col] != 1)
                {
                    continue;
                }
            }

            num_fc = 0;
            for (b = 0; b < TOTAL_IMAGE_BANDS; b++)
            {
                tmp_buf_2d[b] = buf[b] + i_col * num_scenes;
            }

            //            printf("temporal success with processid %d\n", process_id);
            // tmp_valid_date_array = (int*)valid_date_array_scanline[i_col];
            // tmp_fmask_buf = (short int*)fmask_buf_scanline[i_col];

            //            if(i_col == 0)
            //            {

            //                for(i = 0; i < valid_scene_count_scanline[i_col]; i++)
            //                {
            //                    printf("%d ", tmp_valid_date_array[i]);
            //                    for(b = 0; b < TOTAL_IMAGE_BANDS; b++)
            //                    {
            //                        printf("%d ", tmp_buf_2d[b][i]);
            //                    }
            //                    printf("%d ", tmp_fmask_buf[i]);
            //                    printf("\n");
            //                }

            //           }

            if (METHOD == CCD)
            {
                rec_cg = malloc(NUM_FC * sizeof(Output_t));
                if (rec_cg == NULL)
                {
                    RETURN_ERROR("ERROR allocating rec_cg",
                                 FUNC_NAME, FAILURE);
                }

                //                printf("temporal success2 with processid %d\n", process_id);
                //                printf("valid_scene_count_scanline[i_col] is %d;i_col is %d; current_row is %d; probability threshold is %f\n",
                //                       valid_scene_count_scanline[i_col], i_col, current_row, probability_threshold);
                result = ccd(tmp_buf_2d, fmask_buf_scanline[i_col], valid_date_array_scanline[i_col],
                             valid_scene_count_scanline[i_col], rec_cg, &num_fc, n_cols,
                             i_col + 1, current_row + 1, probability_threshold);

                for (i = 0; i < num_fc; i++)
                {
                    //                printf("i for num_fc is %d\n", i);
                    //               MPI_File_write_at(fhoutput, offset, &rec_cg[i], 1, mpi_reccg_type, &mpi_status);
                    // MPI_File_write_all(fhoutput, &rec_cg[i], 1, mpi_reccg_type, &mpi_status);
                    result = write_output_binary(fdoutput, rec_cg[i]);
                    //                offset = offset + size;
                    //                MPI_Get_count( &mpi_status, MPI_INT, &count );
                    //                if (count !=0)
                    //                {
                    //                    printf("warning: writing fail at row_%d col_%d\n", current_row + 1, i_col + 1);
                    //                }
                }
                free(rec_cg);
                // for debug
                // printf("free stage 7 \n");
            }
            else
            {
                s_rec_cg = malloc(NUM_FC * sizeof(Output_t_sccd));
                if (s_rec_cg == NULL)
                {
                    RETURN_ERROR("ERROR allocating rec_cg",
                                 FUNC_NAME, FAILURE);
                }

                result = sccd_hpc(tmp_buf_2d, tmp_fmask_buf, tmp_valid_date_array, valid_scene_count_scanline[i_col], s_rec_cg, &num_fc,
                                  n_cols, i_col + 1, current_row + 1, probability_threshold, min_days_conse, sensor_buf);
                // printf("free stage 9 \n");
                for (i = 0; i < num_fc; i++)
                {
                    result = write_output_binary_sccd(fdoutput, s_rec_cg[i]);
                    // MPI_File_write_all(fhoutput, &s_rec_cg[i], 1, mpi_reccg_type_sccd, &mpi_status);
                }

                // printf("free stage 10 \n");
                free(s_rec_cg);
            }

            if (result != SUCCESS)
            {
                return ("ccd procedure fails at row %d and col %d \n", current_row + 1, i_col + 1);
            }
            //            MPI_Status mpi_status;
            //            memset( &mpi_status, 0xff, sizeof(MPI_Status) );
            //            int count;

        } // end for(i_col = 0; i_col < num_samples; i_col++)

        // MPI_File_close(&fhoutput);
        fclose(fdoutput);
        snprintf(msg_str, sizeof(msg_str), "Row %d finished\n", current_row + 1);
        LOG_MESSAGE(msg_str, FUNC_NAME);

    } // for(j = 0; j < n_block; j++)

    for (i = 0; i < num_scenes; i++)
    {
        // MPI_File_close(&fh[i]);
        fclose(f_bip[i]);
    }

    //        n_remain_line = meta->lines - n_block * n_working_process;

    /********************************************************/
    /*        dispatch jobs to processes                    */
    /********************************************************/

    //        for (j = 0; j < block_num; j ++)
    //        {
    //            for (i = 1; i < n_working_process+1; i++)
    //            {
    //                MPI_Request reqs[5];

    //                if (METHOD == SCCD)
    //                    sprintf(out_filename, "record_change_row%d_s.dat", current_row);
    //                else
    //                    sprintf(out_filename, "record_change_row%d.dat", current_row);

    //                for (i = 0 ; i < meta->samples; i++)
    //                {
    //                    valid_scene_count_scanline[i] = 0;
    //                }

    //                result = read_bip_lines(in_path, scene_list,
    //                         current_row, meta->samples, num_scenes, sdate, buf,
    //                               fmask_buf_scanline, valid_scene_count_scanline, valid_date_array_scanline);
    //                if (result != SUCCESS)
    //                {
    //                    sprintf(errmsg, "Error in reading Landsat data for row_%d for process_%d\n", row, i + 2);
    //                    RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
    //                }

    //                /* send data to process */
    //                // MPI_Isend(sdate, num_scenes, MPI_INT, i, 9, MPI_COMM_WORLD, &reqs[0]);
    //                MPI_Ssend(buf, TOTAL_IMAGE_BANDS * meta->samples * num_scenes, MPI_INT16_T, i, 1, MPI_COMM_WORLD, &reqs[0]);
    //                MPI_Ssend(fmask_buf_scanline, meta->samples * num_scenes, MPI_INT16_T, i, 2, MPI_COMM_WORLD, &reqs[1]);
    //                MPI_Ssend(valid_scene_count_scanline, 1, MPI_INT, i, 3, MPI_COMM_WORLD, &reqs[2]);
    //                MPI_Ssend(num_scenes, 1, MPI_INT, i, 4, MPI_COMM_WORLD, &reqs[3]);
    //                MPI_Ssend(valid_date_array_scanline, meta->samples * num_scenes, MPI_INT, i, 5, MPI_COMM_WORLD, &reqs[4]);

    //                MPI_Waitall(5, reqs, MPI_STATUSES_IGNORE);

    //            }

    //            for (i = 1; i < n_working_process + 1; i++)
    //            {
    //                current_row = j * n_working_process + i;
    //                sprintf(out_fullpath, "%s/%s", out_path, out_filename);
    //                fhoutput= fopen(out_fullpath, "w");

    //                MPI_Request reqs[2];
    //                MPI_Recv(rec_cg_scanline, meta->samples * NUM_FC, mpi_reccg_type, i, 9, MPI_COMM_WORLD, &reqs[0]);
    //                MPI_Recv(num_fc_scanline, meta->samples, MPI_INT, i, 9, MPI_COMM_WORLD, &reqs[1]);

    //                MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    //                for(i_col = 0; i_col < meta->samples; i_col++)
    //                {
    //                    for(i = 0; i < num_fc_scanline[i_col]; i++)
    //                    {
    //                        result = write_output_binary(fhoutput, rec_cg_scanline[i_col* NUM_FC + i]);
    //                    }
    //                }

    //                fclose(fhoutput);

    //            }
    //        }

    /********************************************************/
    /*                   free memory                        */
    /********************************************************/

    free(meta);
    free(scene_list_1d);

    status = free_2d_array((void **)scene_list);
    if (status != SUCCESS)
    {
        RETURN_ERROR("Freeing memory: scene_list\n", FUNC_NAME, FAILURE);
    }

    status = free_2d_array((void **)buf);
    if (status != SUCCESS)
    {
        RETURN_ERROR("Freeing memory: buf\n", FUNC_NAME, FAILURE);
    }

    status = free_2d_array((void **)tmp_buf_2d);
    if (status != SUCCESS)
    {
        RETURN_ERROR("Freeing memory: tmp_buf_2d\n", FUNC_NAME, FAILURE);
    }

    status = free_2d_array((void **)fmask_buf_scanline);
    if (status != SUCCESS)
    {
        RETURN_ERROR("Freeing memory: fmask_buf_scanline\n", FUNC_NAME, FAILURE);
    }

    free(valid_scene_count_scanline);

    status = free_2d_array((void **)valid_date_array_scanline);
    if (status != SUCCESS)
    {
        RETURN_ERROR("Freeing memory: valid_date_array_scanline\n", FUNC_NAME, FAILURE);
    }

    free(num_fc_scanline);

    free(sensor_buf);

    free(user_mask_scanline);
    free(tmp_buf);
    free(sdate);
    // free(fh);

    MPI_Finalize();
}
