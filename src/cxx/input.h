#ifndef INPUTS_H
#define INPUTS_H

#include <stdbool.h>
#include <stdio.h>
#include "const.h"

typedef struct
{
    int lines;                    /* number of lines in a scene */
    int samples;                  /* number of samples in a scene */
    int data_type;                /* envi data type */
    int byte_order;               /* envi byte order */
    int utm_zone;                 /* UTM zone; use a negative number if this is a
                                     southern zone */
    int pixel_size;               /* pixel size */
    char interleave[MAX_STR_LEN]; /* envi save format */
    int upper_left_x;             /* upper left x coordinates */
    int upper_left_y;             /* upper left y coordinates */
} Input_meta_t;

FILE *open_raw_binary(
    char *infile,     /* I: name of the input file to be opened */
    char *access_type /* I: string for the access type for reading the
                            input file; use the raw_binary_format
                            array at the top of this file */
);

void close_raw_binary(
    FILE *fptr /* I: pointer to raw binary file to be closed */
);

int is_leap_year(
    int year /*I: Year to test         */
);

int convert_year_doy_to_ordinal(
    int year, /* I: year */
    int doy,  /* I: day of the year */
    int *jday /* O: julian date since year 0000 */
);

void get_scenename(
    const char *filename, /* I: Name of file to split                       */
    char *directory,      /* O: Directory portion of file name              */
    char *scene_name,     /* O: Scene name portion of the file name.        */
    char *appendix        /* O: Appendix portion of the file name           */
);

int sort_scene_based_on_year_doy_row(
    char **scene_list, /* I/O: scene_list, sorted as output             */
    int num_scenes,    /* I: number of scenes in the scene list         */
    int *sdate,        /* O: year plus date since 0000                  */
    int format);

int create_scene_list(
    const char *in_path,      /* I: string of ARD image directory          */
    int *num_scenes,          /* O: number of scenes                      */
    char *scene_list_filename /* I: file name of list of scene IDs        */
);

int read_raw_binary(
    FILE *rb_fptr,  /* I: pointer to the raw binary file */
    int nlines,     /* I: number of lines to read from the file */
    int nsamps,     /* I: number of samples to read from the file */
    int size,       /* I: number of bytes per pixel (ex. sizeof(uint8)) */
    void *img_array /* O: array of nlines * nsamps * size to be read from
                          the raw binary file (sufficient space should
                          already have been allocated) */
);

int read_envi_header(
    char *image_dir,   /* I: Landsat ARD directory  */
    char *scene_name,  /* I: scene name             */
    Input_meta_t *meta /* O: saved header file info */
);

// int read_tif_header
//(
//     char *in_path,       /* I: Landsat ARD directory  */
//     char *scene_name,      /* I: scene name             */
//     Input_meta_t *meta     /* O: saved header file info */
//);

// int read_bip
// (
//     char *in_path,       /* I: Landsat ARD directory  */
//     char **scene_list,   /* I:   current scene name in list of sceneIDs       */
//     FILE **f_bip,            /* I/O: file pointer array for BIP  file names */
//     int  curr_scene_num,      /* I:   current num. in list of scenes to read */
//     int  row,                 /* I:   the row (Y) location within img/grid   */
//     int  col,                 /* I:   the col (X) location within img/grid   */
//     int  num_samples,         /* I:   number of image samples (X width)      */
//     int *sdate,              /* I:   Original array of julian date values         */
//     short int  **image_buf,          /* O:   pointer to 2-D image band values array */
//     short int *fmask_buf,            /* O:   pointer to 1-D mask values array */
//     short int  *sensor_buf,
//     int *valid_scene_count,   /* I/O: x/y is not always valid for gridded data,  */
//     char **valid_scene_list,  /* I/O: 2-D array for list of filtered            */
//     int *updated_sdate_array/* I/O: new buf of valid date values            */
// );

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
//);

int read_bip_lines(
    char *in_path,     /* I: Landsat ARD directory  */
    char **scene_list, /* I:   current scene name in list of sceneIDs       */
    // FILE **fp_bip,            /* I/O: file pointer array for BIP  file names */
    int row,                   /* I: row number*/
    int num_samples,           /* I:   number of image samples (X width)      */
    int num_scenes,            /* I:   current num. in list of scenes to read */
    int *sdate,                /* I:   Original array of julian date values         */
    short int **image_buf,     /* O:   pointer to a scanline for 2-D image band values array */
    short int **fmask_buf,     /* O:   pointer to a scanline 1-D mask values array */
    int *valid_scene_count,    /* I/O: x/y is not always valid for gridded data,  */
    int **updated_sdate_array, /* I/O: new buf of valid date values for each pixel */
    short int *sensor_buf);

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
//);

int save_scene_list(
    const char *scene_list_directory,
    int num_scenes, /* O: number of scenes                      */
    char **scene_list);

#endif // INPUTS_H
