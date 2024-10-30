#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
// #include "gdal/gdal.h"
// #include "/Library/Frameworks/GDAL.framework/unix/include/gdal.h"
#include "misc.h"
#include "input.h"
#include "utilities.h"
#include "defines.h"

FILE *open_raw_binary(
    char *infile,     /* I: name of the input file to be opened */
    char *access_type /* I: string for the access type for reading the
                            input file; use the raw_binary_format
                            array at the top of this file */
)
{
    FILE *rb_fptr = NULL;                 /* pointer to the raw binary file */
    char FUNC_NAME[] = "open_raw_binary"; /* function name */
    char error_message[512];
    /* Open the file with the specified access type */
    rb_fptr = fopen(infile, access_type);
    if (rb_fptr == NULL)
    {
        sprintf(error_message, "Opening raw binary: %s", infile);
        ERROR_MESSAGE(error_message, FUNC_NAME);
        return NULL;
    }

    /* Return the file pointer */
    return rb_fptr;
}

void close_raw_binary(
    FILE *fptr /* I: pointer to raw binary file to be closed */
)
{
    fclose(fptr);
}

/************************************************************************
FUNCTION: is_leap_year

PURPOSE:
Test if year given is a leap year.

RETURN VALUE:
Type = int
Value    Description
-----    -----------
TRUE     the year is a leap year
FALSE    the year is NOT a leap year

**************************************************************************/
int is_leap_year(
    int year /*I: Year to test         */
)
{
    if (((year % 4) != 0) || (((year % 100) == 0) && ((year % 400) != 0)))
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}

/******************************************************************************
MODULE:  convert_year_doy_to_ordinal

PURPOSE:  convert day of year in a year to julian day counted from year 0000

RETURN VALUE: int
ERROR           Error for year less than 1973 as input
SUCCESS         No errors encountered

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/23/2015   Song Guo         Original Development
20160513    Brian Davis      At some point, a fix was made to add doy only
                             after the last year.  Previously, (v04.03) it
                             was erroneously being added in the loop for each
                             of the years, resulting in creating future dates
                             and time travel......

NOTES:
******************************************************************************/
int convert_year_doy_to_ordinal(
    int year, /* I: year */
    int doy,  /* I: day of the year */
    int *jday /* O: julian date since year 0001 */
)
{
    char FUNC_NAME[] = "convert_year_doy_to_ordinal";
    int i;
    int status;

    if (year < 1973)
    {
        RETURN_ERROR("Landsat data starts from 1973", FUNC_NAME, ERROR);
    }

    if (year != LANDSAT_START_YEAR)
    {
        *jday = ORDINAL_DATE_LAST_DAY_1972;
        for (i = LANDSAT_START_YEAR; i < year; i++)
        {
            status = is_leap_year(i);
            if (status == TRUE)
            {
                *jday += LEAP_YEAR_DAYS;
            }
            else
            {
                *jday += NON_LEAP_YEAR_DAYS;
            }
        }
    }
    *jday += doy;

    return (SUCCESS);
}

/******************************************************************************
MODULE:  sort_scene_based_on_year_doy_row

PURPOSE:  Sort scene list based on year and julian day of year and row number

RETURN VALUE:
Type = int
Value           Description
-----           -----------
ERROR           error return
SUCCESS         No errors encountered

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/23/2015   Song Guo         Original Development
2/1/2016    Song Guo         Added row number

NOTES:
******************************************************************************/
int sort_scene_based_on_year_doy_row(
    char **scene_list, /* I/O: scene_list, sorted as output             */
    int num_scenes,    /* I: number of scenes in the scene list         */
    int *sdate,        /* O: year plus date since 0000                  */
    int format)
{
    int i, month, day;                                     /* loop counter                                  */
    int status;                                            /* return of function calls for errror handling  */
    int year, doy;                                         /* to keep track of year, and day within year    */
    int *yeardoy;                                          /* combined year day of year as one string       */
    int *row;                                              /* row of path/row for ordering from same swath  */
    char temp_string[8];                                   /* for string manipulation                       */
    char temp_string2[5];                                  /* for string manipulation                       */
    char temp_string3[4];                                  /* for string manipulation                       */
    char temp_string4[4];                                  /* for string manipulation                       */
    char temp_string5[3];                                  /* for string manipulation                       */
    char errmsg[MAX_STR_LEN];                              /* for printing error messages                 */
    char FUNC_NAME[] = "sort_scene_based_on_year_doy_row"; /* function name  */
    int len;                                               /* length of string returned from strlen for string manipulation*/
    char temp[MAX_STR_LEN];                                /* for string manipuulation                      */

    /******************************************************************/
    /*                                                                */
    /* Allocate memory for yeardoy                                    */
    /*                                                                */
    /******************************************************************/

    yeardoy = malloc(num_scenes * sizeof(int));
    if (yeardoy == NULL)
    {
        RETURN_ERROR("Allocating yeardoy memory", FUNC_NAME, ERROR);
    }

    row = malloc(num_scenes * sizeof(int));
    if (row == NULL)
    {
        RETURN_ERROR("Allocating row memory", FUNC_NAME, ERROR);
    }

    /******************************************************************/
    /*                                                                */
    /* Get year plus doy from scene name                              */
    /*                                                                */
    /******************************************************************/
    // printf("num_scenes = %d \n", num_scenes);
    for (i = 0; i < num_scenes; i++)
    {
        if (format == TIFF_FORMAT)
        {
            len = strlen(scene_list[i]);
            strncpy(temp_string2, scene_list[i] + 15, 4);
            temp_string2[4] = '\0';
            year = atoi(temp_string2);

            strncpy(temp_string5, scene_list[i] + 19, 2);
            temp_string5[2] = '\0';
            month = atoi(temp_string5);

            strncpy(temp_string5, scene_list[i] + 21, 2);
            temp_string5[2] = '\0';
            day = atoi(temp_string5);
            doy = yearmonth2doy(year, month, day);
            yeardoy[i] = year * 1000 + doy;
            status = convert_year_doy_to_ordinal(year, doy,
                                                 &sdate[i]);

            if (status != SUCCESS)
            {
                sprintf(errmsg, "Converting year %d month %d day %d", year, month, day);
                RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
            }
        }
        else if (format == ENVI_FORMAT)
        {
            len = strlen(scene_list[i]);
            // printf("%s\n",scene_list[i]);
            strncpy(temp_string, scene_list[i] + (len - 12), 7);
            temp_string[7] = '\0';
            yeardoy[i] = atoi(temp_string);
            strncpy(temp_string2, scene_list[i] + (len - 12), 4);
            temp_string2[4] = '\0';
            year = atoi(temp_string2);
            strncpy(temp_string3, scene_list[i] + (len - 8), 3);
            doy = atoi(temp_string3);
            temp_string3[3] = '\0';
            strncpy(temp_string4, scene_list[i] + (len - 15), 3);
            temp_string4[3] = '\0';
            row[i] = atoi(temp_string4);

            status = convert_year_doy_to_ordinal(year, doy, &sdate[i]);

            if (status != SUCCESS)
            {
                sprintf(errmsg, "Converting year %d doy %d", year, doy);
                RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
            }
        }
    }

    /******************************************************************/
    /*                                                                */
    /* Sort the scene_list & sdate based on yeardoy                   */
    /*                                                                */
    /******************************************************************/

    // printf("convert_year_doy_to_ordinal is finished \n");
    quick_sort(yeardoy, scene_list, sdate, 0, num_scenes - 1);
    // printf("quick_sort is finished \n");

    /******************************************************************/
    /*                                                                */
    /* If two identical date data exist (this is only the case        */
    /* for ARD data, then put the smaller row number data in          */
    /* the front.                                                     */
    /*                                                                */
    /******************************************************************/

    for (i = 0; i < num_scenes - 1; i++)
    {
        // printf("sdate[i] = %d \n", sdate[i]);
        // printf("scene_list[i] = %s \n", scene_list[i]);
        if (sdate[i + 1] == sdate[i])
        {
            if (row[i] > row[i + 1])
            {
                strcpy(temp, scene_list[i]);
                strcpy(scene_list[i], scene_list[i + 1]);
                strcpy(scene_list[i + 1], temp);
            }
        }
    }

    /******************************************************************/
    /*                                                                */
    /* Free memory                                                    */
    /*                                                                */
    /******************************************************************/

    free(yeardoy);
    free(row);

    return (SUCCESS);
}
/*****************************************************************************
MODULE:  get_scenename

PURPOSE:  get scene name based on full filename even with path

RETURN VALUE: None

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/21/2015   Song Guo         Original Development

NOTES:
*****************************************************************************/

void get_scenename(
    const char *filename, /* I: Name of file to split                       */
    char *directory,      /* O: Directory portion of file name              */
    char *scene_name,     /* O: Scene name portion of the file name.        */
    char *appendix        /* O: Appendix portion of the file name           */
)
{
    char file_name[MAX_STR_LEN]; /* Local copy of filename                   */
    char *ptr;                   /* String pointer                           */

    /******************************************************************/
    /*                                                                */
    /* Make a local copy of filename so it is not destroyed           */
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
        strcpy(directory, file_name);
        ptr = (char *)strrchr(directory, '/');
        ptr++;
        strcpy(file_name, ptr);
        *ptr = '\0';
    }
    else
    {
        strcpy(directory, "");
    }

    /******************************************************************/
    /*                                                                */
    /* Check for the first "_"                                        */
    /*                                                                */
    /******************************************************************/

    ptr = (char *)strchr(file_name, '_');
    if (ptr != NULL)
    {
        *(ptr++) = '\0';
        strcpy(scene_name, file_name);
        strcpy(appendix, ptr);
    }
    else
    {
        strcpy(scene_name, file_name);
        strcpy(appendix, "");
    }
}

/*****************************************************************************
MODULE:  create_scene_list

PURPOSE:  Create scene list from existing files under working data
          directory and pop them into scene_list string array

RETURN VALUE:
Type = int
Value           Description
-----           -----------
ERROR           Error getting the command-line arguments or a command-line
                argument and associated value were not specified
SUCCESS         No errors encountered

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/21/2015   Song Guo         Original Development
20151203    Brian Davis      Added argument for scene list file name.
09/08/2018  Su Ye            Modification

NOTES:
*****************************************************************************/

int create_scene_list(
    const char *in_path,      /* I: string of ARD image directory          */
    int *num_scenes,          /* O: number of scenes                      */
    char *scene_list_filename /* I: file name of list of scene IDs        */
)
{
    DIR *dirp;
    struct dirent *dp;                      /* structure for directory entries            */
    FILE *fd;                               /* file descriptor for scene list file        */
    char FUNC_NAME[] = "create_scene_list"; /* function name for messages   */
    int scene_counter = 0;                  /* to record number of scenes                */
    char scene_list_directory[MAX_STR_LEN]; /* full directory of the scene list file*/

    sprintf(scene_list_directory, "%s/%s", in_path, scene_list_filename);

    fd = fopen(scene_list_directory, "w");
    if (fd == NULL)
    {
        RETURN_ERROR("Opening scene_list file", FUNC_NAME, ERROR);
    }

    dirp = opendir(in_path);
    if (dirp != NULL)
    {
        while ((dp = readdir(dirp)) != NULL)
        {
            if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0 && strcmp(dp->d_name, scene_list_filename) != 0)
            {
                // printf("%s\n", dp->d_name);
                fprintf(fd, "%s\n", dp->d_name);
                scene_counter++;
            }
        }
    }
    (void)closedir(dirp);
    fclose(fd);
    *num_scenes = scene_counter;

    return (SUCCESS);

    //    DIR *d= opendir(in_path);;
    //    struct dirent *dir;
    //    if (d)
    //    {
    //        int i = 0;
    //        while ((dir = readdir(d)) != NULL)
    //        {
    //            if (i<2){
    //               i++;
    //            }
    //            else{
    //               scene_list[i-2] = dir->d_name;
    //               printf("dir %s\n", dir->d_name);
    //               i++;
    //            }

    //        }
    //        *num_scenes = i - 2;
    //        closedir(d);
    //    }
}

/*****************************************************************************
MODULE:  save_scene_list

PURPOSE:  Create scene list from existing files under working data
          directory and pop them into scene_list string array

RETURN VALUE:
Type = int
Value           Description
-----           -----------
ERROR           Error getting the command-line arguments or a command-line
                argument and associated value were not specified
SUCCESS         No errors encountered

HISTORY:
Date        Programmer       Reason
--------    ---------------  -------------------------------------
1/21/2015   Song Guo         Original Development
20151203    Brian Davis      Added argument for scene list file name.
09/08/2018  Su Ye            Modification

NOTES:
*****************************************************************************/

int save_scene_list(
    const char *scene_list_directory,
    int num_scenes, /* O: number of scenes                      */
    char **scene_list)
{
    FILE *fd;                               /* file descriptor for scene list file        */
    char FUNC_NAME[] = "create_scene_list"; /* function name for messages   */
    int i;

    fd = fopen(scene_list_directory, "w");
    if (fd == NULL)
    {
        RETURN_ERROR("Opening scene_list file", FUNC_NAME, ERROR);
    }

    for (i = 0; i < num_scenes; i++)
        fprintf(fd, "%s\n", scene_list[i]);
    fclose(fd);

    return (SUCCESS);
}

int read_raw_binary(
    FILE *rb_fptr,  /* I: pointer to the raw binary file */
    int nlines,     /* I: number of lines to read from the file */
    int nsamps,     /* I: number of samples to read from the file */
    int size,       /* I: number of bytes per pixel (ex. sizeof(uint8)) */
    void *img_array /* O: array of nlines * nsamps * size to be read from
                          the raw binary file (sufficient space should
                          already have been allocated) */
)
{
    int nvals;                            /* number of values read from the file */
    char FUNC_NAME[] = "read_raw_binary"; /* function name */

    /* Read the data from the raw binary file */
    nvals = fread(img_array, size, nlines * nsamps, rb_fptr);
    if (nvals != nlines * nsamps)
    {
        RETURN_ERROR("Incorrect amount of data read", FUNC_NAME, ERROR);
    }

    return (SUCCESS);
}

/******************************************************************************
MODULE: trimwhitespace

PURPOSE: Trim leading spaces of a sting

RETURN VALUE:
Type = string without trailing space

HISTORY:
Date         Programmer       Reason
----------   --------------   -------------------------------------
1/16/2015    Song Guo         Modified from online code

NOTES:
*****************************************************************************/
char *trimwhitespace(char *str)
{
    char *end;

    /* Trim leading space */
    while (isspace(*str))
        str++;

    if (*str == 0)
        return str;

    /* Trim trailing space */
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end))
        end--;

    /* Write new null terminator */
    *(end + 1) = 0;

    return str;
}

/******************************************************************************
MODULE: read_envi_header

PURPOSE: Reads envi header info into input structure

RETURN VALUE:
Type = None
NOTES:
*****************************************************************************/

int read_envi_header(
    char *in_path,     /* I: Landsat ARD directory  */
    char *scene_name,  /* I: scene name             */
    Input_meta_t *meta /* O: saved header file info */
)
{
    char buffer[MAX_STR_LEN] = "\0";       /* for retrieving fields        */
    char *label = NULL;                    /* for retrieving string tokens */
    char *tokenptr = NULL;                 /* for retrieving string tokens */
    char *tokenptr2 = NULL;                /* for retrieving string tokens */
    char *seperator = "=";                 /* for retrieving string tokens */
    char *seperator2 = ",";                /* for retrieving string tokens */
    FILE *in;                              /* file ptr to input hdr file   */
    int ib;                                /* loop index                   */
    char map_info[10][MAX_STR_LEN];        /* projection information fields*/
    char FUNC_NAME[] = "read_envi_header"; /* function name           */
    char filename[MAX_STR_LEN];            /* scene name                   */
    // char directory[MAX_STR_LEN];      /* for constucting path/file names */
    // char tmpstr[MAX_STR_LEN];         /* char string for text manipulation */
    // char scene_list_name[MAX_STR_LEN];/* char string for text manipulation */

    /******************************************************************/
    /*                                                                */
    /* Determine the file name.                                       */
    /*                                                                */
    /******************************************************************/

    sprintf(filename, "%s/%s/%s_MTLstack.hdr", in_path, scene_name, scene_name);

    in = fopen(filename, "r");
    if (in == NULL)
    {
        RETURN_ERROR("opening header file", FUNC_NAME, FAILURE);
    }

    /* process line by line */
    while (fgets(buffer, MAX_STR_LEN, in) != NULL)
    {

        char *s;
        s = strchr(buffer, '=');
        if (s != NULL)
        {
            /* get string token */
            tokenptr = strtok(buffer, seperator);
            label = trimwhitespace(tokenptr);

            if (strcmp(label, "lines") == 0)
            {
                tokenptr = trimwhitespace(strtok(NULL, seperator));
                meta->lines = atoi(tokenptr);
            }

            if (strcmp(label, "data type") == 0)
            {
                tokenptr = trimwhitespace(strtok(NULL, seperator));
                meta->data_type = atoi(tokenptr);
            }

            if (strcmp(label, "byte order") == 0)
            {
                tokenptr = trimwhitespace(strtok(NULL, seperator));
                meta->byte_order = atoi(tokenptr);
            }

            if (strcmp(label, "samples") == 0)
            {
                tokenptr = trimwhitespace(strtok(NULL, seperator));
                meta->samples = atoi(tokenptr);
            }

            if (strcmp(label, "interleave") == 0)
            {
                tokenptr = trimwhitespace(strtok(NULL, seperator));
                strcpy(meta->interleave, tokenptr);
            }

            if (strcmp(label, "UPPER_LEFT_CORNER") == 0)
            {
                tokenptr = trimwhitespace(strtok(NULL, seperator));
            }

            if (strcmp(label, "map info") == 0)
            {
                tokenptr = trimwhitespace(strtok(NULL, seperator));
            }

            if (strcmp(label, "map info") == 0)
            {
                tokenptr2 = strtok(tokenptr, seperator2);
                ib = 0;
                while (tokenptr2 != NULL)
                {
                    strcpy(map_info[ib], tokenptr2);
                    if (ib == 3)
                        meta->upper_left_x = atoi(map_info[ib]);
                    if (ib == 4)
                        meta->upper_left_y = atoi(map_info[ib]);
                    if (ib == 5)
                        meta->pixel_size = atoi(map_info[ib]);
                    if (ib == 7)
                        meta->utm_zone = atoi(map_info[ib]);
                    tokenptr2 = strtok(NULL, seperator2);
                    ib++;
                }
            }
        }
    }
    fclose(in);

    return (SUCCESS);
}

/******************************************************************************
MODULE: read_envi_header

PURPOSE: Reads envi header info into input structure

RETURN VALUE:
Type = None
NOTES:
*****************************************************************************/

// int read_tif_header
//(
//     char *in_path,       /* I: Landsat ARD directory  */
//     char *scene_name,      /* I: scene name             */
//     Input_meta_t *meta     /* O: saved header file info */
//)
//{

//    /******************************************************************/
//    /*                                                                */
//    /* Determine the file name.                                       */
//    /*                                                                */
//    /******************************************************************/
//    char filename[MAX_STR_LEN];       /* scene name                   */
//    GDALDatasetH  srsDataset;
//    GDALDriverH   hDriver;
//    double        adfGeoTransform[6];
//    GDALRasterBandH hBand;

//    sprintf(filename, "%s/%s_SR/%s_SRB1.tif", in_path, scene_name, scene_name);

//    GDALAllRegister();
//    srsDataset = GDALOpen( filename, GA_ReadOnly );
//    hBand = GDALGetRasterBand(srsDataset, 1 );

//    // printf( "Projection is %s\n", GDALGetProjectionRef(srsDataset ) );
//    meta->lines = GDALGetRasterYSize(srsDataset);
//    meta->data_type = GDT_Int16;
//    meta->byte_order = 0;
//    meta->samples = GDALGetRasterXSize(srsDataset);

//    GDALGetGeoTransform( srsDataset, adfGeoTransform );
//    meta->upper_left_x = adfGeoTransform[0];
//    meta->upper_left_y = adfGeoTransform[3];
//    meta->pixel_size = adfGeoTransform[1];

//    GDALClose(srsDataset);
//    return (SUCCESS);
//}

// int read_bip
// (
//     char *in_path,       /* I: Landsat ARD directory  */
//     char **scene_list,   /* I:   current scene name in list of sceneIDs       */
//     FILE **fp_bip,            /* I/O: file pointer array for BIP  file names */
//     int  curr_scene_num,      /* I:   current num. in list of scenes to read */
//     int  row,                 /* I:   the row (Y) location within img/grid   */
//     int  col,                 /* I:   the col (X) location within img/grid   */
//     int  num_samples,         /* I:   number of image samples (X width)      */
//     int  *sdate,              /* I:   Original array of julian date values         */
//     short int  **image_buf,          /* I/O:   pointer to 2-D image band values array */
//     short int  *fmask_buf,            /* I/O:   pointer to 1-D mask array */
//     short int  *sensor_buf,            /* I/O:   pointer to 1-D mask array */
//     int  *valid_scene_count,   /* I/O: x/y is not always valid for gridded data,  */
//     char **valid_scene_list,  /* I/O: 2-D array for list of filtered            */
//     int  *updated_sdate_array /* I/O: new buf of valid date values            */
// )
// {

//     int  k;                     /* band loop counter.                   */
//     char filename[MAX_STR_LEN]; /* file name constructed from sceneID   */
//     // char shorter_name[MAX_STR_LEN]; /* file name constructed from sceneID*/
//     // char directory[MAX_STR_LEN];
//     // char tmpstr[MAX_STR_LEN];   /* for string manipulation              */
//     char errmsg[MAX_STR_LEN];   /* for printing error text to the log.  */
//     char curr_scene_name[MAX_STR_LEN]; /* current scene name */
//     short int* qa_val;           /* qa value*/
//     bool debug = FALSE;          /* for debug printing                   */
//     char FUNC_NAME[] = "read_bip"; /* function name */

//     /******************************************************************/
//     /*                                                                */
//     /* Determine the BIP file name, open, fseek.                      */
//     /*                                                                */
//     /******************************************************************/
//     sprintf(curr_scene_name, "%s", scene_list[curr_scene_num]);

//     sprintf(filename, "%s/%s/%s_MTLstack", in_path, curr_scene_name, curr_scene_name);

//     fp_bip[curr_scene_num] = open_raw_binary(filename,"rb");
//     if (fp_bip[curr_scene_num] == NULL)
//     {
//         sprintf(errmsg,  "Opening %d scene files\n", curr_scene_num);
//         RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
//     }

//     /******************************************************************/
//     /*                                                                */
//     /* Read the image bands for this scene.                           */
//     /*                                                                */
//     /******************************************************************/

//      /* read quality band, and check if the pixel is valid */
//      fseek(fp_bip[curr_scene_num], (((row - 1)* num_samples + col - 1) *
//          TOTAL_BANDS + TOTAL_BANDS - 1) * sizeof(short int), SEEK_SET);

//      qa_val = malloc(sizeof(short int));
//      read_raw_binary(fp_bip[curr_scene_num], 1, 1,
//                             sizeof(short int), qa_val);

//      if (*qa_val < CFMASK_FILL){
//          if(curr_scene_name[2] == '4' || curr_scene_name[2] == '5')
//              sensor_buf[*valid_scene_count] = LANDSAT45_TM;
//          else if(curr_scene_name[2] == '7')
//              sensor_buf[*valid_scene_count] = LANDSAT7_ETM;
//          else if(curr_scene_name[2] == '8')
//              sensor_buf[*valid_scene_count] = LANDSAT8_OLI;

//          fseek(fp_bip[curr_scene_num], ((row - 1)* num_samples + col - 1) *
//                        TOTAL_BANDS * sizeof(short int), SEEK_SET);
//          for (k = 0; k < TOTAL_BANDS; k++)
//          {

//              if(k < TOTAL_BANDS - 1){
//                  if (read_raw_binary(fp_bip[curr_scene_num], 1, 1,
//                          sizeof(short int), &image_buf[k][*valid_scene_count]) != 0)
//                  {
//                      sprintf(errmsg, "error reading %d scene, %d bands\n", curr_scene_num, k+1);
//                      RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
//                  }
//                  if (debug)
//                  {
//                      printf("%d\n", (short int)image_buf[k][*valid_scene_count]);
//                  }
//              }
//              else{
//                    fmask_buf[*valid_scene_count] = (short int)(*qa_val);
// //                 if (read_raw_binary(fp_bip[curr_scene_num], 1, 1,
// //                         sizeof(short int), &fmask_buf[*valid_scene_count]) != 0)
// //                 {
// //                     sprintf(errmsg, "error reading %d scene, mask bands\n", curr_scene_num);
// //                     printf(errmsg);
// //                     return (FAILURE);
// //                 }
//                  if (debug)
//                  {
//                      printf("%d\n", (short int)fmask_buf[*valid_scene_count]);
//                  }
//              }

//          }
//          strcpy(valid_scene_list[*valid_scene_count], scene_list[curr_scene_num]);
//          updated_sdate_array[*valid_scene_count] = sdate[curr_scene_num];
//          (*valid_scene_count)++;
//      }

//     free(qa_val);

//     close_raw_binary(fp_bip[curr_scene_num]);

//     return (SUCCESS);
// }

// int read_tif
//(
//     char *in_path,       /* I: Landsat ARD directory  */
//     char **scene_list,   /* I:   current scene name in list of sceneIDs       */
//     FILE **fp_bip,            /* I/O: file pointer array for BIP  file names */
//     int  curr_scene_num,      /* I:   current num. in list of scenes to read */
//     int  row,                 /* I:   the row (Y) location within img/grid   */
//     int  col,                 /* I:   the col (X) location within img/grid   */
//     int  num_samples,         /* I:   number of image samples (X width)      */
//     int  *sdate,              /* I:   Original array of julian date values         */
//     short int  **image_buf,          /* I/O:   pointer to 2-D image band values array */
//     short int  *fmask_buf,            /* I/O:   pointer to 1-D mask array */
//     short int  *sensor_buf,            /* I/O:   pointer to 1-D mask array */
//     int  *valid_scene_count,   /* I/O: x/y is not always valid for gridded data,  */
//     char **valid_scene_list,  /* I/O: 2-D array for list of filtered            */
//     int  *updated_sdate_array /* I/O: new buf of valid date values            */
//)
//{

//    int  k;                     /* band loop counter.                   */
//    char filename[MAX_STR_LEN]; /* file name constructed from sceneID   */
//    char errmsg[MAX_STR_LEN];   /* for printing error text to the log.  */
//    char curr_scene_name[MAX_STR_LEN]; /* current scene name */
//    //short int qa_val;           /* qa value*/
//    //short int* pixel_val;
//    char FUNC_NAME[] = "read_tif"; /* function name */
//    GDALDriverH   hDriver;
//    GDALDatasetH  hDataset;
//    GDALRasterBandH hBand;
//    short tmp1, tmp2;
//    short int pixel_qa;

//    /******************************************************************/
//    /*                                                                */
//    /* Determine the BIP file name, open, fseek.                      */
//    /*                                                                */
//    /******************************************************************/
//    sprintf(curr_scene_name, "%s", scene_list[curr_scene_num]);

//    if(curr_scene_name[3] == '4' || curr_scene_name[3] == '5')
//        sensor_buf[*valid_scene_count] = LANDSAT45_TM;
//    else if(curr_scene_name[3] == '7')
//        sensor_buf[*valid_scene_count] = LANDSAT7_ETM;
//    else if(curr_scene_name[3] == '8')
//        sensor_buf[*valid_scene_count] = LANDSAT8_OLI;

//    GDALAllRegister();
//    sprintf(filename, "%s/%s_SR/%s_PIXELQA.tif", in_path, curr_scene_name, curr_scene_name);
//    // printf("%s\n", filename);
//    hDataset = GDALOpen(filename, GA_ReadOnly);
//    hBand = GDALGetRasterBand(hDataset, 1);
//    GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                 &pixel_qa, 1, 1, GDT_Int16,
//                 0, 0 );
//    GDALClose(hDataset);

//    if (qabitval(pixel_qa) < CFMASK_FILL){
//        if ((sensor_buf[*valid_scene_count] == LANDSAT45_TM)||(sensor_buf[*valid_scene_count] == LANDSAT7_ETM)){
//            sprintf(filename, "%s/%s_SR/%s_SRB1.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[0][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB2.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[1][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB3.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[2][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB4.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[3][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB5.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[4][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB7.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[5][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_BT/%s_BTB6.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[6][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//        }else if(sensor_buf[*valid_scene_count] == LANDSAT8_OLI){
//            sprintf(filename, "%s/%s_SR/%s_SRB2.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[0][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB3.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[1][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB4.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[2][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB5.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[3][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB6.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[4][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB7.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[5][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_BT/%s_BTB10.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, col - 1, row - 1, 1, 1,
//                         &image_buf[6][*valid_scene_count], 1, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//        }
//        fmask_buf[*valid_scene_count] = (short int)qabitval(pixel_qa);
//        strcpy(valid_scene_list[*valid_scene_count], scene_list[curr_scene_num]);
//        updated_sdate_array[*valid_scene_count] = sdate[curr_scene_num];
//        (*valid_scene_count)++;
//    }

//    //free(pixel_qa);
//    return (SUCCESS);
//}

int read_bip_lines(
    char *in_path,     /* I: Landsat ARD directory  */
    char **scene_list, /* I:   current scene name in list of sceneIDs       */
    // FILE **f_bip,            /* I/O: file pointer array for BIP  file names */
    int row,                   /* I: row number, The beginning number is 1, not 0!*/
    int num_samples,           /* I:   number of image samples (X width)      */
    int num_scenes,            /* I:   current num. in list of scenes to read */
    int *sdate,                /* I:   Original array of julian date values         */
    short int **image_buf,     /* O:   pointer to a scanline for 2-D image band values array */
    short int **fmask_buf,     /* O:   pointer to a scanline 1-D mask values array */
    int *valid_scene_count,    /* I/O: x/y is not always valid for gridded data,  */
    int **updated_sdate_array, /* I/O: new buf of valid date values for each pixel */
    short int *sensor_buf)
{
    int i, j, k;                /* band loop counter.                   */
    char filename[MAX_STR_LEN]; /* file name constructed from sceneID   */
    char errmsg[MAX_STR_LEN];   /* for printing error text to the log.  */
    short int *tmp_buf;
    FILE *f_bip; /* I/O: file pointer array for BIP  file names */
    char FUNC_NAME[] = "read_bip_lines";

    tmp_buf = malloc(sizeof(short int) * TOTAL_BANDS);
    int tmp_sensor;
    /******************************************************************/
    /*                                                                */
    /* Determine the BIP file name, open, fseek.                      */
    /*                                                                */
    /******************************************************************/

    /******************************************************************/
    /*                                                                */
    /* Read the image bands for this scene.                           */
    /*                                                                */
    /******************************************************************/

    /* read quality band, and check if the pixel is valid */

    for (i = 0; i < num_scenes; i++)
    {

        if (scene_list[i][2] == '4' || scene_list[i][2] == '5')
            tmp_sensor = LANDSAT45_TM;
        else if (scene_list[i][2] == '7')
            tmp_sensor = LANDSAT7_ETM;
        else if (scene_list[i][2] == '8')
            tmp_sensor = LANDSAT8_OLI;

        sprintf(filename, "%s/%s/%s_MTLstack", in_path, scene_list[i], scene_list[i]);

        f_bip = open_raw_binary(filename, "rb");
        if (f_bip == NULL)
        {
            sprintf(errmsg, "Opening %d scene files\n", i);
            RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
        }

        fseek(f_bip, (((row - 1) * num_samples) * TOTAL_BANDS) * sizeof(short int), SEEK_SET);

        for (k = 0; k < num_samples; k++)
        {
            if (read_raw_binary(f_bip, 1, TOTAL_BANDS,
                                sizeof(short int), tmp_buf) != 0)
            {
                sprintf(errmsg, "error reading %s scene, %d row, %d col\n", scene_list[i], row, k);
                RETURN_ERROR(errmsg, FUNC_NAME, ERROR);
            }
            if (tmp_buf[TOTAL_BANDS - 1] < CFMASK_FILL)
            {
                for (j = 0; j < TOTAL_IMAGE_BANDS; j++)
                {
                    // printf("%d\n", valid_scene_count[k]);
                    image_buf[j][k * num_scenes + valid_scene_count[k]] = tmp_buf[j];
                }
                fmask_buf[k][valid_scene_count[k]] = (short int)tmp_buf[TOTAL_BANDS - 1];
                updated_sdate_array[k][valid_scene_count[k]] = sdate[i];
                sensor_buf[valid_scene_count[k]] = tmp_sensor;
                valid_scene_count[k] = valid_scene_count[k] + 1;
            }
        }
        close_raw_binary(f_bip);
    }

    free(tmp_buf);

    return (SUCCESS);
}

// int read_tif_lines
//(
//     char *in_path,       /* I: Landsat ARD directory  */
//     char **scene_list,   /* I:   current scene name in list of sceneIDs       */
//     //FILE **f_bip,            /* I/O: file pointer array for BIP  file names */
//     int row,             /* I: row number, The beginning number is 1, not 0!*/
//     int  num_samples,         /* I:   number of image samples (X width)      */
//     int  num_scenes,      /* I:   current num. in list of scenes to read */
//     int *sdate,              /* I:   Original array of julian date values         */
//     short int  **image_buf,          /* O:   pointer to a scanline for 2-D image band values array */
//     short int **fmask_buf,            /* O:   pointer to a scanline 1-D mask values array */
//     int *valid_scene_count,           /* I/O: x/y is not always valid for gridded data,  */
//     int **updated_sdate_array,         /* I/O: new buf of valid date values for each pixel */
//     short int *sensor_buf
//)
//{
//     int  k, i;                     /* band loop counter.                   */
//     char filename[MAX_STR_LEN]; /* file name constructed from sceneID   */
//     char errmsg[MAX_STR_LEN];   /* for printing error text to the log.  */
//     char curr_scene_name[MAX_STR_LEN]; /* current scene name */
//     //short int qa_val;           /* qa value*/
//     //short int* pixel_val;
//     char FUNC_NAME[] = "read_tif"; /* function name */
//     GDALDriverH   hDriver;
//     GDALDatasetH  hDataset;
//     GDALRasterBandH hBand;
//     short int* tmp_qa;
//     short int* tmp_spectral_B1;
//     short int* tmp_spectral_B2;
//     short int* tmp_spectral_B3;
//     short int* tmp_spectral_B4;
//     short int* tmp_spectral_B5;
//     short int* tmp_spectral_B6;
//     short int* tmp_spectral_B7;
//     short int pixel_qa;
//     int tmp_sensor;

//    /******************************************************************/
//    /*                                                                */
//    /* Determine the BIP file name, open, fseek.                      */
//    /*                                                                */
//    /******************************************************************/
//    /* read quality band, and check if the pixel is valid */
//    tmp_qa = malloc(sizeof(short int) * num_samples);
//    tmp_spectral_B1 = malloc(sizeof(short int) * num_samples);
//    tmp_spectral_B2 = malloc(sizeof(short int) * num_samples);
//    tmp_spectral_B3 = malloc(sizeof(short int) * num_samples);
//    tmp_spectral_B4 = malloc(sizeof(short int) * num_samples);
//    tmp_spectral_B5 = malloc(sizeof(short int) * num_samples);
//    tmp_spectral_B6 = malloc(sizeof(short int) * num_samples);
//    tmp_spectral_B7 = malloc(sizeof(short int) * num_samples);

//    for (i = 0; i < num_scenes; i++)
//    {
//        sprintf(curr_scene_name, "%s", scene_list[i]);
//        GDALAllRegister();
//        sprintf(filename, "%s/%s_SR/%s_PIXELQA.tif", in_path, curr_scene_name, curr_scene_name);
//        // printf("%s\n", filename);

//        hDataset = GDALOpen(filename, GA_ReadOnly);
//        hBand = GDALGetRasterBand(hDataset, 1);
//        GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                     tmp_qa, num_samples, 1,  GDT_Int16,
//                     0, 0 );
//        GDALClose(hDataset);

//        for(k = 0; k < num_samples; k++){
//            tmp_qa[k] =  qabitval(tmp_qa[k]);
//            //printf("%d\n", tmp_qa[k]);
//        }

//        if(curr_scene_name[3] == '4' || curr_scene_name[3] == '5')
//            tmp_sensor = LANDSAT45_TM;
//        else if(curr_scene_name[3] == '7')
//            tmp_sensor = LANDSAT7_ETM;
//        else if(curr_scene_name[3] == '8')
//            tmp_sensor = LANDSAT8_OLI;

//        if ((tmp_sensor == LANDSAT45_TM)||(tmp_sensor == LANDSAT7_ETM)){
//            sprintf(filename, "%s/%s_SR/%s_SRB1.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand(hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B1, num_samples, 1, GDT_Int16,
//                         0, 0 );

//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB2.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B2, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB3.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B3, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB4.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B4, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB5.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B5, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB7.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B6, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_BT/%s_BTB6.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand(hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B7, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//        }else if(tmp_sensor == LANDSAT8_OLI){
//            sprintf(filename, "%s/%s_SR/%s_SRB2.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand(hDataset, 1);
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B1, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB3.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B2, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB4.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B3, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB5.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B4, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB6.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B5, num_samples, 1, GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_SR/%s_SRB7.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B6, num_samples, 1,GDT_Int16,
//                         0, 0 );
//            GDALClose(hDataset);

//            sprintf(filename, "%s/%s_BT/%s_BTB10.tif", in_path, curr_scene_name, curr_scene_name);
//            hDataset = GDALOpen(filename, GA_ReadOnly);
//            hBand = GDALGetRasterBand( hDataset, 1 );
//            GDALRasterIO(hBand, GF_Read, 0, row - 1, num_samples, 1,
//                         tmp_spectral_B7, 1, num_samples, GDT_Int16,
//                         0, 0);
//            GDALClose(hDataset);

//        }

//        for(k = 0; k < num_samples; k++){
//            if(tmp_qa[k] < CFMASK_FILL){
//                 image_buf[0][k * num_scenes + valid_scene_count[k]] = tmp_spectral_B1[k];
//                 image_buf[1][k * num_scenes + valid_scene_count[k]] = tmp_spectral_B2[k];
//                 image_buf[2][k * num_scenes + valid_scene_count[k]] = tmp_spectral_B3[k];
//                 image_buf[3][k * num_scenes + valid_scene_count[k]] = tmp_spectral_B4[k];
//                 image_buf[4][k * num_scenes + valid_scene_count[k]] = tmp_spectral_B5[k];
//                 image_buf[5][k * num_scenes + valid_scene_count[k]] = tmp_spectral_B6[k];
//                 image_buf[6][k * num_scenes + valid_scene_count[k]] = tmp_spectral_B7[k];

//                 fmask_buf[k][valid_scene_count[k]] = (short int)tmp_qa[k];
//                 updated_sdate_array[k][valid_scene_count[k]] = sdate[i];
//                 sensor_buf[valid_scene_count[k]] = tmp_sensor;
//                 valid_scene_count[k] = valid_scene_count[k] + 1;
//            }

//        }
//    }

//    free(tmp_qa);
//    free(tmp_spectral_B1);
//    free(tmp_spectral_B2);
//    free(tmp_spectral_B3);
//    free(tmp_spectral_B4);
//    free(tmp_spectral_B5);
//    free(tmp_spectral_B6);
//    free(tmp_spectral_B7);
//    return (SUCCESS);

//}
