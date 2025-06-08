from datetime import datetime
import datetime as dt
from os.path import join
from dataclasses import fields
import os

from scipy import stats

import numpy as np
import pandas as pd

# from osgeo import gdal
import rasterio
from rasterio.plot import reshape_as_image
from .app import defaults
from .common import SccdOutput, nrtqueue_dt, sccd_dt, nrtmodel_dt, DatasetInfo


def rio_loaddata(path: str) -> np.ndarray:
    """load raster dataset as numpy array

    Parameters
    ----------
    path : str
        path of the input raster
    Returns
    -------
    np.ndarray
    """
    with rasterio.open(path, "r") as ds:
        arr = ds.read()
        if arr.shape[0] == 1:
            return arr[0, :, :]
        else:
            return reshape_as_image(arr)


def get_block_y(block_id: int, n_block_x: int) -> int:
    """get block pos in y axis (started from 1)

    Parameters
    ----------
    block_id: int
        The id of the block to be processed
    n_block_x: int
        Total number of blocks at x axis

    Returns
    -------
    int
        current block id at y axis (start from 1)
    """
    return int((block_id - 1) / n_block_x) + 1


def get_block_x(block_id: int, n_block_x: int) -> int:
    """get block pos at x axis (started from 1)

    Parameters
    ----------
    block_id: int
        The id of the block to be processed
    n_block_x: int
        Total number of blocks at x axis

    Returns
    -------
    int
        current block pos at x axis (start from 1)
    """
    return (block_id - 1) % n_block_x + 1


def get_col_index(pos: int, n_cols, current_block_x, block_width) -> int:
    """get column index in a block

    Parameters
    ----------
    pos: int
        The position of a pixel, i.e., i_row * n_cols + i_col + 1
    n_cols: int
        The number of columns
    current_block_x: int
        The current block id at y axis
    block_width: int
        block width
    Returns
    -------

    """
    return int((pos - 1) % n_cols) - (current_block_x - 1) * block_width


def get_row_index(pos, n_cols, current_block_y, block_height) -> int:
    """
    Parameters
    ----------
    pos: int
        The position of a pixel, i.e., i_row * n_cols + i_col + 1
    n_cols: int
        The number of columns
    current_block_y: int
        The current block id at y axis
    block_height: int
        block height
    Returns
    -------

    """
    return int((pos - 1) / n_cols) - (current_block_y - 1) * block_height


def assemble_cmmaps(
    dataset_info: DatasetInfo,
    result_path: str,
    cmmap_path: str,
    starting_date: int,
    n_cm_maps: int,
    prefix: str,
    cm_output_interval: int,
    clean: bool = True,
):
    """reorganized block-based change magnitudes into a series of maps

    Parameters
    ----------
    config: dict
        pyxccd config dict
    result_path: str
        the path where block-based CM intermediate files are
    cmmap_path: str
        the path to save the new map-based output
    starting_date: int
        the starting date of the dataset
    n_cm_maps: int
        the number of change magnitude outputted per pixel
    prefix: str
        choose from 'CM', 'CM_date', 'CM_direction'
    clean: bool
        if True, clean tmp files
    Returns
    -------

    """
    # anchor_dates_list = None
    if prefix == "CM":
        output_type = np.int16  # type: ignore
    elif prefix == "CM_date":
        output_type = np.int32  # type: ignore
        # cuz the date is produced as byte to squeeze the storage size, need to expand
        # anchor_dates_list_single = np.arange(start=starting_date,
        #                                      stop=starting_date + config['CM_OUTPUT_INTERVAL'] * n_cm_maps,
        #                                      step=config['CM_OUTPUT_INTERVAL'])
        # anchor_dates_list = np.tile(anchor_dates_list_single, config['block_width'] * config['block_height'])

    elif prefix == "CM_direction":
        output_type = np.uint8  # type: ignore
    else:
        raise ValueError("prefix has been among CM, CM_direction, CM_direction")

    cm_map_list = [
        np.full(
            (dataset_info.n_rows, dataset_info.n_cols),
            defaults["COMMON"]["NAN_VAL"],
            dtype=output_type,
        )
        for x in range(n_cm_maps)
    ]
    for iblock in range(dataset_info.nblocks):
        current_block_y = int(np.floor(iblock / dataset_info.n_block_x)) + 1
        current_block_x = iblock % dataset_info.n_block_y + 1
        try:
            cm_block = np.load(
                join(
                    result_path,
                    "{}_x{}_y{}.npy".format(prefix, current_block_x, current_block_y),
                )
            ).astype(output_type)
        except OSError as e:
            print("Reading CM files fails: {}".format(e))
        #    continue

        # if prefix == "CM_date":
        #    cm_block_copy = cm_block.copy()
        #    cm_block = cm_block + defaults["COMMON"]["JULIAN_LANDSAT4_LAUNCH"]
        # we assign an extremely large value to original NA value (255)
        #    cm_block[cm_block_copy == -9999] = -9999

        cm_block_reshape = np.reshape(
            cm_block, (dataset_info.block_width * dataset_info.block_height, n_cm_maps)
        )
        hori_profile = np.hsplit(cm_block_reshape, n_cm_maps)
        for count, maps in enumerate(cm_map_list):
            maps[
                (current_block_y - 1)
                * dataset_info.block_height : current_block_y
                * dataset_info.block_height,
                (current_block_x - 1)
                * dataset_info.block_width : current_block_x
                * dataset_info.block_width,
            ] = hori_profile[count].reshape(
                dataset_info.block_height, dataset_info.block_width
            )

    # output cm images
    for count, cm_map in enumerate(cm_map_list):
        ordinal_date = starting_date + count * cm_output_interval
        outfile = join(
            cmmap_path,
            "{}_maps_{}_{}{}.npy".format(
                prefix,
                str(ordinal_date),
                pd.Timestamp.fromordinal(ordinal_date).year,
                get_doy(ordinal_date),
            ),
        )
        np.save(outfile, cm_map)

    if clean is True:
        tmp_filenames = [
            file for file in os.listdir(result_path) if file.startswith(prefix + "_x")
        ]
        for file in tmp_filenames:
            os.remove(join(result_path, file))


def get_rowcol_intile(
    id: int, block_width: int, block_height: int, block_x: int, block_y: int
):
    """Calculate row and col in original images based on pos index and block location

    Parameters
    ----------
    id: int
        id of the pixel (i.e., i_row * n_cols + i_col). Note starting from 0
    block_width: int
        the width of each block
    block_height: int
        the height of each block
    block_x:int
        block location at x direction
    block_y:int
        block location at y direction
    Returns
    -------
    (int, int)
        return (original_row, original_col), i.e., row and col number (starting from 1) in original image (e.g., Landsat ARD 5000*5000)
    """
    original_row = int(id / block_width + (block_y - 1) * block_height + 1)
    original_col = int(id % block_width + (block_x - 1) * block_width + 1)
    return original_row, original_col


def get_id_inblock(pos: int, block_width: int, block_height: int, n_cols: int):
    """pixel id in the block, starting from 0

    Parameters
    ----------
    pos : int
        position id of a pixel, i.e., i_row * n_cols + i_col + 1
    block_width : int
        the width of the block
    block_height : int
        the width of the height
    n_cols : int
        the number of the culimn

    Returns
    -------
    int
        pixel id, i.e., i_row * n_cols + i_col
    """

    row_inblock = int(int((pos - 1) / n_cols) % block_height)
    col_inblock = (pos - 1) % n_cols % block_width
    return row_inblock * block_width + col_inblock


# def gdal_save_file_1band(out_path, array, gdal_type, trans, proj, cols, rows, image_format='GTiff'):
#     """
#     save array as tiff format
#     Parameters
#     ----------
#     out_path : full outputted path
#     array : numpy array to be saved
#     gdal_type: gdal type
#     trans: transform coefficients
#     proj: projection
#     rows: the row number
#     cols: the col number
#     image_format: default is GTiff
#     Returns
#     -------
#     TRUE OR FALSE
#     """
#     outdriver = gdal.GetDriverByName(image_format)
#     outdata = outdriver.Create(out_path, cols, rows, 1, gdal_type)
#     if outdata == None:
#         return False
#     outdata.GetRasterBand(1).WriteArray(array)
#     outdata.FlushCache()
#     outdata.SetGeoTransform(trans)
#     outdata.FlushCache()
#     outdata.SetProjection(proj)
#     outdata.FlushCache()
#     return True


def get_time_now(tz):
    """get datetime for now

    Parameters
    ----------
    tz
        The input time zone

    Returns
    -------
    datetime
        datatime format of current time
    """
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


# def get_ymd_now(tz):
#     """get datetime for now

#     Parameters
#     ----------
#     tz: str
#         The input time zone

#     Returns
#     -------
#     datetime
#         datatime format of current time
#     """
#     return datetime.now(tz).strftime("%Y-%m-%d")


def get_doy(ordinal_date: int) -> str:
    """get doy from ordinal date

    Parameters
    ----------
    ordinal_date: int
        a ordinal date (MATLAB-format ordinal date)

    Returns:
    -------
    str
        day of year
    """
    return str(pd.Timestamp.fromordinal(ordinal_date).timetuple().tm_yday).zfill(3)


# def get_anchor_days(starting_day: int, n_cm_maps: int, interval: int):
#     """
#     get a list of starting days for each change magnitude time slices

#     Parameters
#     ----------
#     starting_days:int
#         Starting days
#     n_cm_maps: int
#         The number of change magnitudes
#     interval:int
#         The interval of change magnitudes

#     Returns
#     -------
#         A list of starting day
#     """
#     return np.arange(
#         start=starting_day, stop=starting_day + n_cm_maps * interval, step=interval
#     )


def assemble_array(array_list: list, n_block_x: int) -> np.ndarray:
    """
    Assemble a list of block-based array to a bigger array that aligns with the dimension of an ARD tile

    Parameters
    ----------
    array_list: list
        a list of np.ndarray
    n_block_x: int
        block number at x axis

    Returns
    -------
    np.ndarray
        an array [nrows, ncols]
    """
    full_feature_array = np.hstack(array_list)
    full_feature_array = np.vstack(
        np.hsplit(full_feature_array, n_block_x)
    )  # (nrows, ncols, nfeatures)
    return full_feature_array


def read_data(path: str) -> np.ndarray:
    """Load a sample file containing acquisition days and spectral values.
    The first column is assumed to be the day number, subsequent columns
    correspond to the day number. This improves readability of large datasets.

    Parameters
    ----------
    path : str
        the path of csv

    Returns
    -------
    np.ndarray

    """
    return np.genfromtxt(path, delimiter=",", dtype=np.int64, encoding="utf-8").T


# def date2matordinal(year:int, month:int, day:int):
#     """

#     Parameters
#     ----------
#     year : int
#         _description_
#     month : int
#         _descriptdate2matordinalion_
#     day : int
#         _description_

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     return pd.Timestamp.toordinal(dt.date(year, month, day))


# def matordinal2date(ordinal):
#     return pd.Timestamp.fromordinal(ordinal)


def save_nrtfiles(
    out_folder: str, outfile_prefix: str, sccd_pack: SccdOutput, data_ext: pd.DataFrame
):
    """save nrt files into local for debug purpose

    Parameters
    ----------
    out_folder : str
        The folder to svae results
    outfile_prefix : str
        prefix for saved files
    sccd_pack : SccdOutput
        SCCD output
    data_ext : pd.DataFrame
        observation as pandas dataframe
    """
    """
    save all files for C debug
    :param out_folder: the outputted folder
    :param outfile_prefix: the prefix of outputted files
    :param sccd_pack:
    :param data_ext:
    :return:
    """
    data_ext.to_csv(
        join(out_folder, "spectral_{}_extension.csv").format(outfile_prefix),
        index=False,
        header=False,
    )
    # data_ini_current.to_csv(join(out_path, 'spectral_{}_ini.csv').format(pid), index=False, header=False)
    np.asarray(sccd_pack.nrt_mode).tofile(
        join(out_folder, "sccd_pack{}_nrt_mode").format(outfile_prefix)
    )
    sccd_pack.rec_cg.tofile(
        join(out_folder, "sccd_pack{}_rec_cg").format(outfile_prefix)
    )
    sccd_pack.nrt_model.tofile(
        join(out_folder, "sccd_pack{}_nrt_model").format(outfile_prefix)
    )
    sccd_pack.nrt_queue.tofile(
        join(out_folder, "sccd_pack{}_nrt_queue").format(outfile_prefix)
    )
    sccd_pack.min_rmse.tofile(
        join(out_folder, "sccd_pack{}_min_rmse").format(outfile_prefix)
    )


def save_obs2csv(out_path: str, data: pd.DataFrame):
    """save observation dataframe to a local csv

    Parameters
    ----------
    out_path : str
        the path for saving csv
    data : pd.DataFrame
        data to be outputed
    """
    data.to_csv(out_path, index=False, header=False)


def unindex_sccdpack(sccd_pack_single: SccdOutput) -> list:
    """remove index of sccdpack to save memory

    Parameters
    ----------
    sccd_pack_single : SccdOutput
        sccd output

    Returns
    -------
    list
        a list for five SccdOutput components
    """

    sccd_pack_single = sccd_pack_single._replace(
        rec_cg=sccd_pack_single.rec_cg.tolist()
    )

    if len(sccd_pack_single.nrt_model) > 0:
        sccd_pack_single = sccd_pack_single._replace(
            nrt_model=sccd_pack_single.nrt_model.tolist()
        )

    if len(sccd_pack_single.nrt_queue) > 0:
        sccd_pack_single = sccd_pack_single._replace(
            nrt_queue=sccd_pack_single.nrt_queue.tolist()
        )

    return list(sccd_pack_single)


def index_sccdpack(sccd_pack_single) -> SccdOutput:
    """convert a list of SccdOutput components back to namedtuple SccdOutput

    Parameters
    ----------
    sccd_pack_single : list
        A list of SccdOutput components

    Returns
    -------
    SccdOutput
        sccd output

    Raises
    ------
    Exception
       The element number of sccd_pack_single is not five
    """

    if len(sccd_pack_single) != defaults["SCCD"]["PACK_ITEM"]:
        raise Exception(
            "the element number of sccd_pack_single must be {}".format(
                defaults["SCCD"]["PACK_ITEM"]
            )
        )

    # convert to named tuple
    sccd_pack_single = SccdOutput(*sccd_pack_single)

    # replace the element to structured array
    if len(sccd_pack_single.rec_cg) == 0:
        sccd_pack_single = sccd_pack_single._replace(
            rec_cg=np.asarray(sccd_pack_single.rec_cg, dtype=np.float64)
        )
    else:
        sccd_pack_single = sccd_pack_single._replace(
            rec_cg=np.asarray(sccd_pack_single.rec_cg, dtype=sccd_dt)
        )
    if len(sccd_pack_single.nrt_model) > 0:
        sccd_pack_single = sccd_pack_single._replace(
            nrt_model=np.asarray(sccd_pack_single.nrt_model, dtype=nrtmodel_dt)
        )
    if len(sccd_pack_single.nrt_queue) > 0:
        sccd_pack_single = sccd_pack_single._replace(
            nrt_queue=np.asarray(sccd_pack_single.nrt_queue, dtype=nrtqueue_dt)
        )
    return sccd_pack_single


def save_1band_fromrefimage(
    array: np.ndarray, out_path: str, ref_image_path=None, dtype=None
):
    """save an array into the local as tif, using the georeference from a refimage

    Parameters
    ----------
    array : np.ndarray
        Inputted array to be converted
    out_path : str
        Path for the output
    ref_image_path : str, optional
        Path for the reference image to copy georeference info, by default None
    dtype : _type_, optional
       str or numpy.dtype, optional. The data type for bands. For example: uint8 or rasterio.uint16.
    """
    if dtype == None:
        dtype = np.int16

    if ref_image_path is None:
        profile = {
            "driver": "GTiff",
            "height": array.shape[0],
            "width": array.shape[1],
            "count": 1,
            "dtype": dtype,
            "compress": "lzw",
        }
    else:
        with rasterio.open(ref_image_path, "r") as ds:
            profile = ds.profile
            profile.update(dtype=array.dtype, count=1, compress="lzw")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)


def coefficient_matrix(date: int, num_coefficients: int) -> np.ndarray:
    """Generate cos and sin variables for Fourier transform function

    Parameters
    ----------
    date : int
        Inputted ordinal date
    num_coefficients : int
        Number of variables to be outputted, i.e., the length of the matrix as return

    Returns
    -------
    np.ndarray
    """

    slope_scale = 10000
    w23 = 2 * np.pi / 365.25
    matrix = np.zeros(shape=(num_coefficients), order="F")

    # lookup optimizations
    # Before optimization - 12.53% of total runtime
    # After optimization  - 10.57% of total runtime
    cos = np.cos
    sin = np.sin

    matrix[0] = 1
    matrix[1] = date / slope_scale
    matrix[2] = cos(w23 * date)
    matrix[3] = sin(w23 * date)

    if num_coefficients >= 6:
        w45 = 2 * w23
        matrix[4] = cos(w45 * date)
        matrix[5] = sin(w45 * date)

    if num_coefficients >= 8:
        w67 = 3 * w23
        matrix[6] = cos(w67 * date)
        matrix[7] = sin(w67 * date)

    return matrix


def predict_ref(coefs: np.ndarray, date: int, num_coefficients: int = 6) -> int:
    """predicting a single-band reflectance using harmonic coefficients for a date

    Parameters
    ----------
    coefs : np.ndarray
        1-d array for harmonic coefficients
    date : int
        ordinal date for the inputted date
    num_coefficients : int, optional
        the number of coefs, by default 6

    Returns
    -------
    int
        the predicted reflectance
    """
    coef_matrix = coefficient_matrix(date, num_coefficients)
    return np.dot(coef_matrix, coefs.T)


def generate_rowcolimage(ref_image_path: str, out_path: str):
    """a function to convert the reference image to index image (starting from 1, e.g., the first pixel is 100001),
    which has the same rows and columns as the reference image

    Parameters
    ----------
    ref_image_path : str
        Path of the reference image
    out_path : str
        Path of the outputted image
    """
    # ref_image_path = '/home/coloury/Dropbox/UCONN/HLS/HLS.L30.T18TYM.2022074T153249.v2.0.B10.tif'
    # ref_image = gdal.Open(ref_image_path, gdal.GA_ReadOnly)
    # trans = ref_image.GetGeoTransform()
    # proj = ref_image.GetProjection()
    # cols = ref_image.RasterXSize
    # rows = ref_image.RasterYSize
    ref_image = rio_loaddata(ref_image_path)

    i, j = np.indices(ref_image.shape[:2])
    index = (i + 1) * 100000 + j + 1

    with rasterio.open(ref_image_path, "r") as ds:
        profile = ds.profile
        profile.update(dtype="int32", count=1, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(index, 1)
    save_1band_fromrefimage(index, out_path, ref_image_path, dtype=np.int32)


def calculate_sccd_cm(sccd_pack: SccdOutput) -> float:
    """compute median change magnitude for the current anomalies at the tail

    Parameters
    ----------
    sccd_pack : SccdOutput
        sccd output

    Returns
    -------
    float
        computed as the median values for the current conse_last change magnitudes
    """
    start_index = defaults["SCCD"]["NRT_BAND"] - sccd_pack.nrt_model[0]["conse_last"]
    pred_ref = np.asarray(
        [
            [
                predict_ref(
                    sccd_pack.nrt_model[0]["nrt_coefs"][b],
                    sccd_pack.nrt_model[0]["obs_date_since1982"][i_conse + start_index]
                    + defaults["COMMON"]["JULIAN_LANDSAT4_LAUNCH"],
                )
                for i_conse in range(sccd_pack.nrt_model[0]["conse_last"])
            ]
            for b in range(defaults["SCCD"]["NRT_BAND"])
        ]
    )
    cm = (
        sccd_pack.nrt_model[0]["obs"][
            :, start_index : defaults["SCCD"]["DEFAULT_CONSE"]
        ]
        - pred_ref
    )
    return np.median(cm, axis=1)


# from typing import Callable, Any
# data_class_type = DataclassInstance | type[DataclassInstance]
# Callable[..., Any],


def class_from_dict(data_class, dict_var: dict):
    """convert dictionary to dataclass following the declaration of the dataclass

    Parameters
    ----------
    data_class :
        Declare for dataclass
    dict_var : dict
        Inputted dictionary

    Returns
    -------
    dataclass
    """
    fieldSet = {f.name for f in fields(data_class) if f.init}
    filteredArgDict = {k: v for k, v in dict_var.items() if k in fieldSet}
    return data_class(**filteredArgDict)


def rio_warp(input_path: str, output_path: str, template_path: str):
    """warp a raster from a template file

    Parameters
    ----------
    input_path : str
        path of inputted raster
    output_path : str
        path of outputted path
    template_path : str
        path of template path
    """
    cmd = f"rio warp {input_path} {output_path} --like {template_path} --overwrite"
    os.system(cmd)


def modeby(input_array: np.ndarray, index_array: np.ndarray) -> list:
    """calculate modes of input_array groupped by index_array.

    Parameters
    ----------
    input_array : np.ndarray
        input array to calculate
    index_array : np.ndarray
        the object array, where the same id indicates the pixel for the same object

    Returns
    -------
    list
        a list of mode value for each object, following ascending order of unique id. modified from: https://stackoverflow.com/questions/49372918/group-numpy-into-multiple-sub-arrays-using-an-array-of-values
    """

    # Get argsort indices, to be used to sort a and b in the next steps
    # input_array = classification_map
    # index_array = object_map
    sidx = index_array.argsort(kind="mergesort")
    a_sorted = input_array[sidx]
    b_sorted = index_array[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    split = [a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])]
    mode_list = [stats.mode(x, keepdims=True)[0][0] for x in split]
    return mode_list


def mode_median_by(
    input_array_mode: np.ndarray,
    input_array_median: np.ndarray,
    index_array: np.ndarray,
) -> tuple:
    """_summary_

    Parameters
    ----------
    input_array_mode : np.ndarray
        input array for calculating mode
    input_array_median : np.ndarray
        input array for calculating median
    index_array : np.ndarray
        the array for indicating objects. The pixels in the same object have the same ids

    Returns
    -------
    (list, list)
        a list of mode value and a list of median value for each object
    """

    sidx = index_array.argsort(kind="mergesort")
    a1_sorted = input_array_mode[sidx]
    a2_sorted = input_array_median[sidx]
    b_sorted = index_array[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    split_mode = [a1_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])]
    split_median = [a2_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])]
    mode_list = [stats.mode(x)[0][0] for x in split_mode]
    median_list = [np.median(x[~np.isnan(x)]) for x in split_median]
    return mode_list, median_list


def getcategory_cold(cold_plot: np.ndarray, i_curve: int, t_c: float = -200.0) -> int:
    """an empirical way to get break category for COLD algorithm

    Parameters
    ----------
    cold_plot : np.ndarray
        Cold result for a single pixel, a structured array of dtype = :py:type:`~pyxccd.common.cold_rec_cg`
    i_curve : int
        Curve number to be classified
    t_c : float, optional
        The threshold to be used, by default -200.0

    Returns
    -------
    int
        break category

            1 - land disturbance

            2 - regrowth

            3 - aforestation

    """
    if (
        cold_plot[i_curve]["magnitude"][3] > t_c
        and cold_plot[i_curve]["magnitude"][2] < -t_c
        and cold_plot[i_curve]["magnitude"][4] < -t_c
    ):
        if (
            cold_plot[i_curve + 1]["coefs"][3, 1]
            > np.abs(cold_plot[i_curve]["coefs"][3, 1])
            and cold_plot[i_curve + 1]["coefs"][2, 1]
            < -np.abs(cold_plot[i_curve]["coefs"][2, 1])
            and cold_plot[i_curve + 1]["coefs"][4, 1]
            < -np.abs(cold_plot[i_curve]["coefs"][4, 1])
        ):
            return 3  # aforestation
        else:
            return 2  # regrowth
    else:
        return 1  # land disturbance


def getcategory_sccd(sccd_plot: np.ndarray, i_curve: int, t_c: float = -200.0) -> int:
    """an empirical way to get break category for COLD algorithm

    Parameters
    ----------
    sccd_plot : np.ndarray
        sccd offline result for a single pixel, a structured array of dtype = :py:type:`~pyxccd.common.rec_cg`
    i_curve : int
        Curve number to be classified
    t_c : float, optional
        The threshold to be used, by default -200.0

    Returns
    -------
    int
        break category

            1 - land disturbance

            2 - regrowth

            3 - aforestation
    """
    if (
        sccd_plot[i_curve]["magnitude"][3] > t_c
        and sccd_plot[i_curve]["magnitude"][2] < -t_c
        and sccd_plot[i_curve]["magnitude"][4] < -t_c
    ):
        return 2  # regrowth
    else:
        return 1  # land disturbance


def extract_features(
    cold_plot: np.ndarray,
    band: int,
    ordinal_day_list: list,
    nan_val: int = 0,
    feature_outputs: list = ["a0", "a1", "b1"],
) -> list:
    """generate features for classification based on a plot-based rec_cg and a list of days to be predicted

    Parameters
    ----------
    cold_plot : np.ndarray
        A structured array of :py:type:`~pyxccd.common.cold_rec_cg`
    band : int
        Band index, started from 0, i.e., index 0 is band 1, index 1 is band 2, etc
    ordinal_day_list : list
        A list of ordinal day to extract a0, a0 = intercept + slope * a1 / 10000
    nan_val : int
        The default values assigned to feature output, by default 0
    feature_outputs : list, optional
        Indicate which features to be outputted.  They must be within [a0, c1, a1, b1,a2, b2, a3, b3, rmse],
        by default ["a0", "a1", "b1"]

    Returns
    -------
    A list
        a list of 1-d array. The length of list is len(feature_outputs); the length of 1-d array is len(ordinal_day_list)

    Raises
    ------
    ValueError
        The outputted feature must be in [a0, c1, a1, b1,a2, b2, a3, b3, cv, rmse]
    """

    features = [
        np.full(len(ordinal_day_list), nan_val, dtype=np.double)
        for x in range(len(feature_outputs))
    ]
    for index, ordinal_day in enumerate(ordinal_day_list):
        # print(index)
        for idx, cold_curve in enumerate(cold_plot):
            if idx == len(cold_plot) - 1:
                max_days = dt.datetime(
                    pd.Timestamp.fromordinal(cold_plot[idx]["t_end"]).year, 12, 31, 0, 0
                ).toordinal()
            else:
                max_days = cold_plot[idx + 1]["t_start"]

            min_day = (
                dt.datetime(1985, 12, 31, 0, 0).toordinal()
                if idx == 0
                else cold_curve["t_start"]
            )
            if min_day <= ordinal_day < max_days:
                for n, feature in enumerate(feature_outputs):
                    if int(cold_curve["category"] / 10) == 5:  # permanent snow
                        features[n][index] = 0
                        continue
                    if feature == "a0":
                        features[n][index] = (
                            cold_curve["coefs"][band][0]
                            + cold_curve["coefs"][band][1]
                            * ordinal_day
                            / defaults["COMMON"]["SLOPE_SCALE"]
                        )
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "c1":
                        features[n][index] = (
                            cold_curve["coefs"][band][1]
                            / defaults["COMMON"]["SLOPE_SCALE"]
                        )
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "a1":
                        features[n][index] = cold_curve["coefs"][band][2]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "b1":
                        features[n][index] = cold_curve["coefs"][band][3]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "a2":
                        features[n][index] = cold_curve["coefs"][band][4]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "b2":
                        features[n][index] = cold_curve["coefs"][band][5]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "a3":
                        features[n][index] = cold_curve["coefs"][band][6]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "b3":
                        features[n][index] = cold_curve["coefs"][band][7]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == "rmse":
                        features[n][index] = cold_curve["rmse"][band]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    else:
                        raise ValueError(
                            "the outputted feature must be in [a0, c1, a1, b1,a2, b2, a3, b3, rmse, cv]"
                        )
                break

    if "cv" in feature_outputs:
        ordinal_day_years = [
            pd.Timestamp.fromordinal(day).year for day in ordinal_day_list
        ]
        for index, ordinal_year in enumerate(ordinal_day_years):
            for cold_curve in cold_plot:
                if (cold_curve["t_break"] == 0) or (cold_curve["change_prob"] != 100):
                    continue
                break_year = pd.Timestamp.fromordinal(cold_curve["t_break"]).year
                if break_year == ordinal_year:
                    features[feature_outputs.index("cv")][index] = cold_curve[
                        "magnitude"
                    ][band]
                    break

    return features
