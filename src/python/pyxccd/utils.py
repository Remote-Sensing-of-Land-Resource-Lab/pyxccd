import numpy as np
from datetime import datetime
import pandas as pd
from os.path import join
from dataclasses import fields
import os
import datetime as dt

# from osgeo import gdal
import rasterio
from rasterio.plot import reshape_as_image
from .app import defaults
from .common import SccdOutput, nrtqueue_dt, sccd_dt, nrtmodel_dt, DatasetInfo


def rio_loaddata(path: str) -> np.ndarray:
    """_summary_

    Args:
        path (str): the path of the input raster

    Returns:
        np.ndarray: _description_
    """
    with rasterio.open(path, "r") as ds:
        arr = ds.read()
        if arr.shape[0] == 1:
            return arr[0, :, :]
        else:
            return reshape_as_image(arr)


def get_block_y(block_id, n_block_x):
    """
    Parameters
    ----------
    block_id: integer
    n_block_x: integer, number of blocks at x xis

    Returns
    -------
    current block id at y axis
    """
    return int((block_id - 1) / n_block_x) + 1


def get_block_x(block_id, n_block_x):
    """
    Parameters
    ----------
    block_id: integer
    n_block_x: integer, number of blocks at x xis

    Returns
    -------
    current block id at x axis
    """
    return (block_id - 1) % n_block_x + 1


def get_col_index(pos, n_cols, current_block_x, block_width):
    """
    Parameters
    ----------
    pos
    n_cols
    current_block_x
    block_width

    Returns
    -------

    """
    return int((pos - 1) % n_cols) - (current_block_x - 1) * block_width


def get_row_index(pos, n_cols, current_block_y, block_height):
    """
    Parameters
    ----------
    pos: start from 1
    n_cols
    current_block_y
    block_height

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
    """
    this function reorganized block-based fix-interval CM intermediate files into map-based output (one map per interval)
    Parameters
    ----------
    config: dictionary
        pyxccd config dictionary
    result_path: string
        the path where block-based CM intermediate files are
    cmmap_path: string
        the path to save the new map-based output
    starting_date: integer
        the starting date of the dataset
    n_cm_maps: integer
        the number of change magnitude outputted per pixel
    prefix: {'CM', 'CM_date', 'CM_direction'}
    clean: True -> clean tmp files
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


def get_rowcol_intile(index, block_width, block_height, block_x, block_y):
    """
    calculate row and col in original images based on pos index and block location
    Parameters
    ----------
    id: integer
        position id of the pixel (i.e., i_row * n_cols + i_col)
    block_width: integer
        the width of each block
    block_height: integer
        the height of each block
    block_x:integer
        block location at x direction
    block_y:integer
        block location at y direction
    Returns
    -------
    (original_row, original_col)
    row and col number (starting from 1) in original image (e.g., Landsat ARD 5000*5000)
    """
    original_row = int(index / block_width + (block_y - 1) * block_height + 1)
    original_col = int(index % block_width + (block_x - 1) * block_width + 1)
    return original_row, original_col


def get_id_inblock(pos, block_width, block_height, n_cols):
    """
    :param pos:
    :param block_width:
    :param block_height:
    :param block_x:
    :param block_y:
    :return: pixel id in the bloack, starting from 0
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
    """
    Parameters
    ----------
    tz: string

    Returns
    -------
    datatime format of current time
    """
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def get_ymd_now(tz):
    """
    Parameters
    ----------
    tz: string

    Returns
    -------
    datatime format of current time
    """
    return datetime.now(tz).strftime("%Y-%m-%d")


def get_doy(ordinal_date):
    """
    Parameters
    ----------
    ordinal_date: int
    a ordinal date (MATLAB-format ordinal date)

    Returns: string
    -------
    doy
    """
    return str(pd.Timestamp.fromordinal(ordinal_date).timetuple().tm_yday).zfill(3)


def get_anchor_days(starting_day, n_cm_maps, interval):
    """
    get a list of starting days for each change magnitude time slices
    Parameters
    ----------
    starting_days
    n_cm_maps
    interval

    Returns
    -------

    """
    return np.arange(
        start=starting_day, stop=starting_day + n_cm_maps * interval, step=interval
    )


def assemble_array(array_list, n_block_x):
    """
    assemble a list of block-based array to a bigger array that aligns with the dimension of an ARD tile
    Parameters
    ----------
    array_list
    n_block_x: block number at x axis

    Returns
    -------
    an array [nrows, ncols]
    """
    full_feature_array = np.hstack(array_list)
    full_feature_array = np.vstack(
        np.hsplit(full_feature_array, n_block_x)
    )  # (nrows, ncols, nfeatures)
    return full_feature_array


def read_blockdata(block_folder, total_pixels, total_bands):
    img_files = [f for f in os.listdir(block_folder) if f.startswith("L")]

    # sort image files by dates
    img_dates = [
        pd.Timestamp.toordinal(
            dt.datetime(int(folder_name[9:13]), 1, 1)
            + dt.timedelta(int(folder_name[13:16]) - 1)
        )
        for folder_name in img_files
    ]
    files_date_zip = sorted(zip(img_dates, img_files))
    img_files_sorted = [x[1] for x in files_date_zip]
    img_dates_sorted = np.asarray([x[0] for x in files_date_zip])
    img_stack = [
        np.load(join(block_folder, f)).reshape(total_pixels, total_bands)
        for f in img_files_sorted
    ]
    img_stack = np.dstack(img_stack)
    return img_stack, img_dates_sorted


def read_data(path):
    """Load a sample file containing acquisition days and spectral values.
    The first column is assumed to be the day number, subsequent columns
    correspond to the day number. This improves readability of large datasets.
    Args:
        path: location of CSV containing test data
    Returns:
        A 2D numpy array.
    """
    return np.genfromtxt(path, delimiter=",", dtype=np.int64, encoding="utf-8").T


def date2matordinal(year, month, day):
    return pd.Timestamp.toordinal(dt.date(year, month, day))


def matordinal2date(ordinal):
    return pd.Timestamp.fromordinal(ordinal)


def save_nrtfiles(out_folder, outfile_prefix, sccd_pack, data_ext):
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


def save_obs2csv(out_path, data):
    data.to_csv(out_path, index=False, header=False)


def unindex_sccdpack(sccd_pack_single):
    """
    remove index of sccdpack to save memory
    sccd_pack_single: a namedtuple SccdOutput
    :return: a nested list
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


def index_sccdpack(sccd_pack_single):
    """
    convert list of sccdpack to namedtuple to facilitate parse,
    :param sccd_pack_single: a nested list
    :return: a namedtuple SccdOutput
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


def save_1band_fromrefimage(array: np.ndarray, out_path: str, ref_image_path=None):
    """_summary_
    Args:
        array (np.ndarray): input array
        out_path (str): the path for saving output
        ref_image_path (_type_, optional): _description_. Defaults to None.
        gtype (_type_, optional): _description_. Defaults to rasterio.int16.
    """
    if ref_image_path is None:
        profile = {
            "driver": "GTiff",
            "height": array.shape[0],
            "width": array.shape[1],
            "count": 1,
            "dtype": array.dtype,
            "compress": "lzw",
        }
    else:
        with rasterio.open(ref_image_path, "r") as ds:
            profile = ds.profile
            profile.update(dtype=array.dtype, count=1, compress="lzw")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)


def coefficient_matrix(dates, num_coefficients):
    """
    Fourier transform function to be used for the matrix of inputs for
    model fitting
    Args:
        dates: list of ordinal dates
        num_coefficients: how many coefficients to use to build the matrix
    Returns:
        Populated numpy array with coefficient values
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
    matrix[1] = dates / slope_scale
    matrix[2] = cos(w23 * dates)
    matrix[3] = sin(w23 * dates)

    if num_coefficients >= 6:
        w45 = 2 * w23
        matrix[4] = cos(w45 * dates)
        matrix[5] = sin(w45 * dates)

    if num_coefficients >= 8:
        w67 = 3 * w23
        matrix[6] = cos(w67 * dates)
        matrix[7] = sin(w67 * dates)

    return matrix


def predict_ref(model, dates, num_coefficients=6):
    coef_matrix = coefficient_matrix(dates, num_coefficients)
    return np.dot(coef_matrix, model.T)


def generate_rowcolimage(ref_image_path, out_path):
    """
    a function to convert source image to index image (starting from 1, e.g., the first pixel is 10001)
    :param ref_image_path: the path for reference images
    :param out_path:  the path for output images
    :return:
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
    save_1band_fromrefimage(index, out_path, ref_image_path)


def calculate_sccd_cm(sccd_pack):
    """
    Parameters
    ----------
    sccd_pack

    Returns
    -------

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
    """convert dictionary to dataclas

    Args:
        dataclass (DataclassInstance | type[DataclassInstance]): _description_
        dict_var (dict): _description_

    Returns:
        _type_: _description_
    """
    fieldSet = {f.name for f in fields(data_class) if f.init}
    filteredArgDict = {k: v for k, v in dict_var.items() if k in fieldSet}
    return data_class(**filteredArgDict)


def rio_warp(input: str, output: str, template: str):
    """warp image using rasterio cmd
    Args:
        input (str): the directory of input file
        output (str): the directory of output file
        template (str): the directory of template file that define the output georeference informations
    """
    cmd = f"rio warp {input} {output} --like {template} --overwrite"
    os.system(cmd)
