# Author: Su Ye
# This script is an example for generating block-based stack files from original ARD zip as
# intermediate inputs to the COLD algorithm in a HPC environment. As preparation, you need to
# download '_BT' and '_SR' for all Landsat ARD collection 1.
# This script has 4 steps: 1) warp single-path array to limit the observation inputs for each pixel
# if single_path is set True; 2) unzip all images and unpack bit-based QA bands; 3)
# partition each 5000*5000 temporal images to blocks and eliminate those image blocks if no clear,
# water or snow pixel were in it (so to save disk space and enable independent IO for individual
# block in later time-series analysis); 4) save each image block to python-native binary format
# (.npy) into its block folders

# For a 42-year Landsat ARD C1 tile (~3000 images), this script averagely produces ~350 G intermediate disk
# files, and takes ~12 mins to finish if 200 EPYC 7452 cores are used.
# running example:
# source /home/colory666/env_collect/py310_geo/bin/activate
# python prepare_ard.py --source_dir=/data/landsat_c2/204_22 --out_dir=/data/results/204_22_stack --yaml_path=/home/colory666/pyxccd/src/python/pyxccd/imagetool/config.yaml  --n_cores=10
import os
import shutil
import tarfile
import logging
import time
from typing import Optional
import datetime as dt
import xml.etree.ElementTree as ET
from logging import Logger
import yaml
import click
from pytz import timezone
from datetime import datetime
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path
from importlib import resources as importlib_resources
from multiprocessing import Pool
import functools

import pandas as pd
import numpy as np
from glob import glob
from dateutil.parser import parse
from pyxccd.common import DatasetInfo
from pyxccd.utils import class_from_dict, rio_loaddata, rio_warp


# define constant here
QA_CLEAR = 0
QA_WATER = 1
QA_SHADOW = 2
QA_SNOW = 3
QA_CLOUD = 4
QA_FILL = 255

QA_CIRRUS_HLS = 0
QA_WATER_HLS = 5
QA_SHADOW_HLS = 3
QA_SNOW_HLS = 4
QA_CLOUDADJACENT_HLS = 2
QA_CLOUD_HLS = 1
res = 30


s2_stack_bands = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
l8_stack_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]


def mask_value(vector, val):
    """
    Build a boolean mask around a certain value in the vector.

    Args:
        vector: 1-d ndarray of values
        val: values to mask on
    Returns:
        1-d boolean ndarray
    """
    return vector == val


def qabitval_array_HLS(packedint_array: np.ndarray):
    """
    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value.

    fill > cloud > shadow > snow > water > clear

    Args:
        packedint: int value to bit check
    Returns:
        offset value to use
    """
    unpacked = np.full(packedint_array.shape, 0)
    QA_CLOUD_unpacked = np.bitwise_and(packedint_array, 1 << 1)
    QA_CLOUD_ADJ = np.bitwise_and(packedint_array, 1 << 2)
    QA_SHADOW_unpacked = np.bitwise_and(packedint_array, 1 << 3)
    QA_SNOW_unpacked = np.bitwise_and(packedint_array, 1 << 4)
    QA_WATER_unpacked = np.bitwise_and(packedint_array, 1 << 5)

    unpacked[QA_WATER_unpacked > 0] = QA_WATER
    unpacked[QA_SNOW_unpacked > 0] = QA_SNOW
    unpacked[QA_SHADOW_unpacked > 0] = QA_SHADOW
    unpacked[QA_CLOUD_ADJ > 0] = QA_CLOUD
    unpacked[QA_CLOUD_unpacked > 0] = QA_CLOUD
    unpacked[packedint_array == QA_FILL] = QA_FILL

    return unpacked


def qabitval_array(packedint_array: np.ndarray):
    """
    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value.
    fill > cloud > shadow > snow > water > clear
    Args:
        packedint: int value to bit check
    Returns:
        offset value to use
    """
    unpacked = np.full(packedint_array.shape, QA_FILL)
    QA_CLOUD_unpacked = np.bitwise_and(packedint_array, 1 << (QA_CLOUD + 1))
    QA_SHADOW_unpacked = np.bitwise_and(packedint_array, 1 << (QA_SHADOW + 1))
    QA_SNOW_unpacked = np.bitwise_and(packedint_array, 1 << (QA_SNOW + 1))
    QA_WATER_unpacked = np.bitwise_and(packedint_array, 1 << (QA_WATER + 1))
    QA_CLEAR_unpacked = np.bitwise_and(packedint_array, 1 << (QA_CLEAR + 1))

    unpacked[QA_CLEAR_unpacked > 0] = QA_CLEAR
    unpacked[QA_WATER_unpacked > 0] = QA_WATER
    unpacked[QA_SNOW_unpacked > 0] = QA_SNOW
    unpacked[QA_SHADOW_unpacked > 0] = QA_SHADOW
    unpacked[QA_CLOUD_unpacked > 0] = QA_CLOUD
    return unpacked


def qabitval_array_c2(packedint_array: np.ndarray):
    """
    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value for c2

    fill > cloud > shadow > snow > water > clear

    Args:
        packedint: int value to bit check
    Returns:
        offset value to use
    """
    unpacked = np.full(packedint_array.shape, QA_FILL)
    QA_CLEAR_unpacked = np.bitwise_and(packedint_array, 1 << 6)
    QA_SHADOW_unpacked = np.bitwise_and(packedint_array, 1 << 4)
    QA_CLOUD_unpacked = np.bitwise_and(packedint_array, 1 << 3)
    QA_DILATED_unpacked = np.bitwise_and(packedint_array, 1 << 1)
    QA_SNOW_unpacked = np.bitwise_and(packedint_array, 1 << 5)
    QA_WATER_unpacked = np.bitwise_and(packedint_array, 1 << 7)

    unpacked[QA_CLEAR_unpacked > 0] = QA_CLEAR
    unpacked[QA_WATER_unpacked > 0] = QA_WATER
    unpacked[QA_SNOW_unpacked > 0] = QA_SNOW
    unpacked[QA_SHADOW_unpacked > 0] = QA_SHADOW
    unpacked[QA_CLOUD_unpacked > 0] = QA_CLOUD
    unpacked[QA_DILATED_unpacked > 0] = QA_CLOUD

    return unpacked


def single_image_stacking_hls(
    source_dir: str,
    out_dir: str,
    logger: Logger,
    dataset_info: DatasetInfo,
    is_partition: bool,
    clear_threshold: float,
    low_date_bound: str,
    upp_date_bound: str,
    folder: str,
):
    """
    unzip single image, convert bit-pack qa to byte value, and save as numpy
    :param source_dir: the parent folder to save image 'folder'
    :param out_dir: the folder to save result
    :param logger: the handler of logger file
    :param data_info: data info data class
    :param is_partition: True, partition each image into blocks; False, save original size of image
    :param clear_threshold: threshold of clear pixel percentage, if lower than threshold, won't be processed
    :param low_date_bound: the lower date of user interested date range
    :param upp_date_bound: the upper date of user interested date range
    :param folder: the folder name of image
    :return:
    """
    try:
        QA_band = rio_loaddata(join(source_dir, folder, f"{folder}.Fmask.tif"))
    except Exception as e:
        # logger.error('Cannot open QA band for {}: {}'.format(folder, e))
        logger.error(f"Cannot open QA band for {folder}: {e}")
        return False

    # convertQA = np.vectorize(qabitval)
    QA_band_unpacked = qabitval_array_HLS(QA_band).astype(np.short)
    if clear_threshold > 0:
        clear_ratio = np.sum(
            np.logical_or(QA_band_unpacked == QA_CLEAR, QA_band_unpacked == QA_WATER)
        ) / np.sum(QA_band_unpacked != QA_FILL)
    else:
        clear_ratio = 1

    if clear_ratio > clear_threshold:
        [collection, sensor, tile_id, imagetime, version1, version2] = folder.rsplit(
            "."
        )
        year = imagetime[0:4]
        doy = imagetime[4:7]
        file_name = sensor + tile_id + year + doy + collection + version1
        if low_date_bound is not None:
            if (dt.datetime(int(year), 1, 1) + dt.timedelta(int(doy) - 1)) < parse(
                low_date_bound
            ):
                return True
        if upp_date_bound is not None:
            if (dt.datetime(int(year), 1, 1) + dt.timedelta(int(doy) - 1)) > parse(
                upp_date_bound
            ):
                return True

        if sensor == "L30":
            try:
                b1 = rio_loaddata(join(source_dir, folder, "{}.B02.tif".format(folder)))
                b2 = rio_loaddata(join(source_dir, folder, "{}.B03.tif".format(folder)))
                b3 = rio_loaddata(join(source_dir, folder, "{}.B04.tif".format(folder)))
                b4 = rio_loaddata(join(source_dir, folder, "{}.B05.tif".format(folder)))
                b5 = rio_loaddata(join(source_dir, folder, "{}.B06.tif".format(folder)))
                b6 = rio_loaddata(join(source_dir, folder, "{}.B07.tif".format(folder)))
                b7 = np.full(b6.shape, 0)  # assign zero

            except Exception as e:
                # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                logger.error("Cannot open Landsat bands for {}: {}".format(folder, e))
                return False
        else:
            try:
                b1 = rio_loaddata(join(source_dir, folder, "{}.B02.tif".format(folder)))
                b2 = rio_loaddata(join(source_dir, folder, "{}.B03.tif".format(folder)))
                b3 = rio_loaddata(join(source_dir, folder, "{}.B04.tif".format(folder)))
                b4 = rio_loaddata(join(source_dir, folder, "{}.B8A.tif".format(folder)))
                b5 = rio_loaddata(join(source_dir, folder, "{}.B11.tif".format(folder)))
                b6 = rio_loaddata(join(source_dir, folder, "{}.B12.tif".format(folder)))
                b7 = np.full(b6.shape, 0)

            except Exception as e:
                # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                logger.error("Cannot open Landsat bands for {}: {}".format(folder, e))
                return False

        if (
            (b1 is None)
            or (b2 is None)
            or (b3 is None)
            or (b4 is None)
            or (b5 is None)
            or (b6 is None)
        ):
            logger.error("Reading Landsat band fails for {}".format(folder))
            return False

        if is_partition is True:
            # width of a block
            bytesize = 2  # short16 = 2 * byte
            # source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
            B1_blocks = np.lib.stride_tricks.as_strided(
                b1,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B2_blocks = np.lib.stride_tricks.as_strided(
                b2,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B3_blocks = np.lib.stride_tricks.as_strided(
                b3,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B4_blocks = np.lib.stride_tricks.as_strided(
                b4,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B5_blocks = np.lib.stride_tricks.as_strided(
                b5,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B6_blocks = np.lib.stride_tricks.as_strided(
                b6,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B7_blocks = np.lib.stride_tricks.as_strided(
                b7,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            QA_blocks = np.lib.stride_tricks.as_strided(
                QA_band_unpacked,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            for i in range(dataset_info.n_block_y):
                for j in range(dataset_info.n_block_x):
                    # check if no valid pixels in the chip, then eliminate
                    qa_unique = np.unique(QA_blocks[i][j])

                    # skip blocks are all cloud, shadow or filled values
                    # in DHTC, we also don't need to save pixel that has qa value of 'QA_CLOUD',
                    # 'QA_SHADOW', or FILLED value (255)
                    if (
                        QA_CLEAR not in qa_unique
                        and QA_WATER not in qa_unique
                        and QA_SNOW not in qa_unique
                    ):
                        continue

                    block_folder = "block_x{}_y{}".format(j + 1, i + 1)
                    np.save(
                        join(join(out_dir, block_folder), file_name),
                        np.dstack(
                            [
                                B1_blocks[i][j],
                                B2_blocks[i][j],
                                B3_blocks[i][j],
                                B4_blocks[i][j],
                                B5_blocks[i][j],
                                B6_blocks[i][j],
                                B7_blocks[i][j],
                                QA_blocks[i][j],
                            ]
                        ).astype(np.int16),
                    )

        else:
            np.save(
                join(out_dir, file_name),
                np.dstack([b1, b2, b3, b4, b5, b6, b7, QA_band_unpacked]).astype(
                    np.int16
                ),
            )
        # scene_list.append(folder_name)
    else:
        # logger.info('Not enough clear observations for {}'.format(folder[0:len(folder) - 3]))
        logger.warn("Not enough clear observations for {}".format(folder))

    return True


def single_image_stacking(
    tmp_path: str,
    source_dir: str,
    out_dir: str,
    clear_threshold: float,
    path_array: np.ndarray,
    logger: Logger,
    dataset_info: DatasetInfo,
    is_partition: bool,
    low_date_bound: str,
    upp_date_bound: str,
    b_c2: bool,
    folder: str,
):
    """
    unzip single image, convert bit-pack qa to byte value, and save as numpy
    :param tmp_path: tmp folder to save unzip image
    :param source_dir: image folder save source zipped files
    :param out_dir: the folder to save result
    :param clear_threshold: threshold of clear pixel percentage, if lower than threshold, won't be processed
    :param path_array: path array has the same dimension of inputted image, and the pixel value indicates
                      the path which the pixel belongs to; if path_array == none, we will use all path
    :param logger: the handler of logger file
    :param dataset_info: dataset information
    :param is_partition: True, partition each image into blocks; False, save original size of image
    :param low_date_bound: the lower bound of user interested year range
    :param upp_date_bound: the upper bound of user interested year range
    :param b_c2: False
    :param folder: the folder name of image
    :return:
    """
    # unzip SR
    if os.path.exists(join(tmp_path, folder)):
        shutil.rmtree(join(tmp_path, folder), ignore_errors=True)
    if os.path.exists(join(tmp_path, folder.replace("SR", "BT"))):
        shutil.rmtree(join(tmp_path, folder.replace("SR", "BT")), ignore_errors=True)

    try:
        with tarfile.open(join(source_dir, folder + ".tar")) as tar_ref:
            try:
                tar_ref.extractall(join(tmp_path, folder))
            except Exception:
                # logger.warning('Unzip fails for {}'.format(folder))
                logger.error("Unzip fails for {}".format(folder))
                return
    except IOError as e:
        logger.error("Unzip fails for {}: {}".format(folder, e))
        # return

    # unzip BT
    try:
        with tarfile.open(
            join(source_dir, folder.replace("SR", "BT") + ".tar")
        ) as tar_ref:
            try:
                tar_ref.extractall(join(tmp_path, folder.replace("SR", "BT")))
            except Exception:
                # logger.warning('Unzip fails for {}'.format(folder.replace("SR", "BT")))
                logger.error("Unzip fails for {}".format(folder.replace("SR", "BT")))
                return
    except IOError as e:
        logger.error("Unzip fails for {}: {}".format(folder.replace("SR", "BT"), e))
        return

    if not isdir(join(tmp_path, folder.replace("SR", "BT"))):
        logger.error("Fail to locate BT folder for {}".format(folder))
        return

    try:
        if b_c2 is False:
            QA_band = rio_loaddata(
                join(
                    join(tmp_path, folder),
                    "{}_PIXELQA.tif".format(folder[0 : len(folder) - 3]),
                )
            )
        else:
            QA_band = rio_loaddata(
                join(
                    join(tmp_path, folder),
                    "{}_QA_PIXEL.TIF".format(folder[0 : len(folder) - 3]),
                )
            )
    except Exception as e:
        # logger.error('Cannot open QA band for {}: {}'.format(folder, e))
        logger.error("Cannot open QA band for {}: {}".format(folder, e))
        return

    # convertQA = np.vectorize(qabitval)
    if b_c2 is False:
        QA_band_unpacked = qabitval_array(QA_band).astype(np.short)
    else:
        QA_band_unpacked = qabitval_array_c2(QA_band).astype(np.short)
    if clear_threshold > 0:
        clear_ratio = np.sum(
            np.logical_or(QA_band_unpacked == QA_CLEAR, QA_band_unpacked == QA_WATER)
        ) / np.sum(QA_band_unpacked != QA_FILL)
    else:
        clear_ratio = 1

    if clear_ratio > clear_threshold:
        if folder[3] == "5":
            sensor = "LT5"
        elif folder[3] == "7":
            sensor = "LE7"
        elif folder[3] == "8":
            sensor = "LC8"
        elif folder[3] == "4":
            sensor = "LT4"
        elif folder[3] == "9":
            sensor = "LC9"
        else:
            logger.error(
                "Sensor is not correctly formatted for the scene {}".format(folder)
            )

        col = folder[8:11]
        row = folder[11:14]
        year = folder[15:19]
        doy = datetime(int(year), int(folder[19:21]), int(folder[21:23])).strftime("%j")
        collection = "C{}".format(folder[35:36])
        if b_c2 is False:
            version = folder[37:40]
        else:
            version = folder[34:35]
        file_name = sensor + col + row + year + doy + collection + version
        if low_date_bound is not None:
            if (dt.datetime(int(year), 1, 1) + dt.timedelta(int(doy) - 1)) < parse(
                low_date_bound
            ):
                return
        if upp_date_bound is not None:
            if (dt.datetime(int(year), 1, 1) + dt.timedelta(int(doy) - 1)) > parse(
                upp_date_bound
            ):
                return
        if b_c2 is False:
            if sensor == "LT5" or sensor == "LE7" or sensor == "LT4":
                try:
                    b1 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b1.tif".format(folder))
                    )
                    b2 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b2.tif".format(folder))
                    )
                    b3 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b3.tif".format(folder))
                    )
                    b4 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b4.tif".format(folder))
                    )
                    b5 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b5.tif".format(folder))
                    )
                    b6 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b7.tif".format(folder))
                    )
                    b7 = rio_loaddata(
                        join(
                            join(tmp_path, "{}_BT".format(folder[0 : len(folder) - 3])),
                            "{}_BTB6.tif".format(folder[0 : len(folder) - 3]),
                        )
                    )
                except Exception as e:
                    # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                    logger.error(
                        "Cannot open Landsat bands for {}: {}".format(folder, e)
                    )
                    return
            elif sensor == "LC8" or "LC9":
                try:
                    b1 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b2.tif".format(folder))
                    )
                    b2 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b3.tif".format(folder))
                    )
                    b3 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b4.tif".format(folder))
                    )
                    b4 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b5.tif".format(folder))
                    )
                    b5 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b6.tif".format(folder))
                    )
                    b6 = rio_loaddata(
                        join(join(tmp_path, folder), "{}b7.tif".format(folder))
                    )
                    b7 = rio_loaddata(
                        join(
                            join(tmp_path, "{}_BT".format(folder[0 : len(folder) - 3])),
                            "{}_BTB10.tif".format(folder[0 : len(folder) - 3]),
                        )
                    )
                except Exception as e:
                    # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                    logger.error(
                        "Cannot open Landsat bands for {}: {}".format(folder, e)
                    )
                    return
        else:
            if sensor == "LT5" or sensor == "LE7" or sensor == "LT4":
                try:
                    b1 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B1.TIF".format(folder))
                    )
                    b2 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B2.TIF".format(folder))
                    )
                    b3 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B3.TIF".format(folder))
                    )
                    b4 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B4.TIF".format(folder))
                    )
                    b5 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B5.TIF".format(folder))
                    )
                    b6 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B7.TIF".format(folder))
                    )
                    b7 = rio_loaddata(
                        join(
                            join(tmp_path, "{}_BT".format(folder[0 : len(folder) - 3])),
                            "{}_BT_B6.TIF".format(folder[0 : len(folder) - 3]),
                        )
                    )
                except Exception as e:
                    # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                    logger.error(
                        "Cannot open Landsat bands for {}: {}".format(folder, e)
                    )
                    return
            elif sensor == "LC8" or "LC9":
                try:
                    b1 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B2.TIF".format(folder))
                    )
                    b2 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B3.TIF".format(folder))
                    )
                    b3 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B4.TIF".format(folder))
                    )
                    b4 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B5.TIF".format(folder))
                    )
                    b5 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B6.TIF".format(folder))
                    )
                    b6 = rio_loaddata(
                        join(join(tmp_path, folder), "{}_B7.TIF".format(folder))
                    )
                    b7 = rio_loaddata(
                        join(
                            tmp_path,
                            "{}_BT".format(folder[0 : len(folder) - 3]),
                            "{}_BT_B10.TIF".format(folder[0 : len(folder) - 3]),
                        )
                    )
                except Exception as e:
                    # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                    logger.error(
                        "Cannot open Landsat bands for {}: {}".format(folder, e)
                    )
                    return

        if (
            (b1 is None)
            or (b2 is None)
            or (b3 is None)
            or (b4 is None)
            or (b5 is None)
            or (b6 is None)
            or (b7 is None)
        ):
            logger.error("Reading Landsat band fails for {}".format(folder))
            return

        if b_c2 is True:
            b1 = (10000 * (b1 * 2.75e-05 - 0.2)).astype(np.int16)
            b2 = (10000 * (b2 * 2.75e-05 - 0.2)).astype(np.int16)
            b3 = (10000 * (b3 * 2.75e-05 - 0.2)).astype(np.int16)
            b4 = (10000 * (b4 * 2.75e-05 - 0.2)).astype(np.int16)
            b5 = (10000 * (b5 * 2.75e-05 - 0.2)).astype(np.int16)
            b7 = (10000 * (b7 * 2.75e-05 - 0.2)).astype(np.int16)
            b6 = (10 * (b6 * 0.00341802 + 149)).astype(np.int16)

        # if path_array is not None, we will eliminate those observation that has different path with its assigned path
        if path_array is not None:  # meaning that single-path processing
            if not os.path.exists(
                join(join(tmp_path, folder), folder.replace("_SR", ".xml"))
            ):
                logger.error(
                    "Cannot find xml file for {}".format(
                        join(join(tmp_path, folder), folder.replace("_SR", ".xml"))
                    )
                )
                return
            tree = ET.parse(join(join(tmp_path, folder), folder.replace("_SR", ".xml")))

            # get root element
            root = tree.getroot()
            if b_c2 is False:
                elements = root.findall(
                    "./{https://landsat.usgs.gov/ard/v1}scene_metadata/{https://landsat.usgs.gov/"
                    "ard/v1}global_metadata/{https://landsat.usgs.gov/ard/v1}wrs"
                )
            else:
                elements = root.findall("SCENE_METADATA/IMAGE_ATTRIBUTES")
            if len(elements) == 0:
                logger.error("Parsing xml fails for {}".format(folder))
                return
            if b_c2 is False:
                pathid = int(elements[0].attrib["path"])
            else:
                if elements[0][3].text is not None:
                    pathid = int(elements[0][3].text)
                else:
                    raise ValueError("path id could not be located.")
            # print(pathid)

            # assign filled value to the pixels has different path id so won't be processed
            QA_band_unpacked[path_array != pathid] = QA_FILL

        if is_partition is True:
            # width of a block
            bytesize = 2  # short16 = 2 * byte
            # source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
            B1_blocks = np.lib.stride_tricks.as_strided(
                b1,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B2_blocks = np.lib.stride_tricks.as_strided(
                b2,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B3_blocks = np.lib.stride_tricks.as_strided(
                b3,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B4_blocks = np.lib.stride_tricks.as_strided(
                b4,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B5_blocks = np.lib.stride_tricks.as_strided(
                b5,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B6_blocks = np.lib.stride_tricks.as_strided(
                b6,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B7_blocks = np.lib.stride_tricks.as_strided(
                b7,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            QA_blocks = np.lib.stride_tricks.as_strided(
                QA_band_unpacked,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            for i in range(dataset_info.n_block_y):
                for j in range(dataset_info.n_block_x):
                    # check if no valid pixels in the chip, then eliminate
                    qa_unique = np.unique(QA_blocks[i][j])

                    # skip blocks are all cloud, shadow or filled values
                    # in DHTC, we also don't need to save pixel that has qa value of 'QA_CLOUD',
                    # 'QA_SHADOW', or FILLED value (255)
                    if (
                        QA_CLEAR not in qa_unique
                        and QA_WATER not in qa_unique
                        and QA_SNOW not in qa_unique
                    ):
                        continue

                    block_folder = "block_x{}_y{}".format(j + 1, i + 1)
                    np.save(
                        join(join(out_dir, block_folder), file_name),
                        np.dstack(
                            [
                                B1_blocks[i][j],
                                B2_blocks[i][j],
                                B3_blocks[i][j],
                                B4_blocks[i][j],
                                B5_blocks[i][j],
                                B6_blocks[i][j],
                                B7_blocks[i][j],
                                QA_blocks[i][j],
                            ]
                        ).astype(np.int16),
                    )

        else:
            np.save(
                join(out_dir, file_name),
                np.dstack([b1, b2, b3, b4, b5, b6, b7, QA_band_unpacked]).astype(
                    np.int16
                ),
            )
        # scene_list.append(folder_name)
    else:
        # logger.info('Not enough clear observations for {}'.format(folder[0:len(folder) - 3]))
        logger.warn(
            "Not enough clear observations for {}".format(folder[0 : len(folder) - 3])
        )

    # delete unzip folder
    shutil.rmtree(join(tmp_path, folder), ignore_errors=True)
    shutil.rmtree(join(tmp_path, folder.replace("SR", "BT")), ignore_errors=True)


def single_image_stacking_collection2(
    tmp_path: str,
    source_dir: str,
    out_dir: str,
    clear_threshold: float,
    logger: Logger,
    dataset_info: DatasetInfo,
    reference_path: str,
    is_partition: bool,
    low_date_bound: str,
    upp_date_bound: str,
    folder: str,
):
    """
    for collection 2
    :param tmp_path: tmp folder to save unzip image
    :param source_dir: image folder save source zipped files
    :param out_dir: the folder to save result
    :param clear_threshold: threshold of clear pixel percentage, if lower than threshold, won't be processed
    :param logger: the handler of logger file
    :param dataset_info:DatasetInfo
    :param is_partition: True, partition each image into blocks; False, save original size of image
    :param low_date_bound: the lower bound of user interested date range
    :param upp_date_bound: the upper bound of user interested date range
    :param bounds
    :param folder: the folder name of image
    :return:
    """
    # unzip SR
    if os.path.exists(join(tmp_path, folder)):
        shutil.rmtree(join(tmp_path, folder), ignore_errors=True)

    try:
        with tarfile.open(join(source_dir, folder + ".tar")) as tar_ref:
            try:
                tar_ref.extractall(join(tmp_path, folder))
            except Exception:
                # logger.warning('Unzip fails for {}'.format(folder))
                logger.error("Unzip fails for {}".format(folder))
                return
    except IOError as e:
        logger.error("Unzip fails for {}: {}".format(folder, e))
        # return

    try:
        rio_warp(
            join(tmp_path, folder, f"{folder}_QA_PIXEL.TIF"),
            join(tmp_path, folder, "_tmp_img.tif"),
            reference_path,
        )
        QA_band = rio_loaddata(join(tmp_path, folder, f"{folder}_QA_PIXEL.TIF"))
    except Exception as e:
        # logger.error('Cannot open QA band for {}: {}'.format(folder, e))
        logger.error("Cannot open QA band for {}: {}".format(folder, e))
        return

    # convertQA = np.vectorize(qabitval)
    QA_band_unpacked = qabitval_array_c2(QA_band).astype(np.short)
    if clear_threshold > 0:
        clear_ratio = np.sum(
            np.logical_or(QA_band_unpacked == QA_CLEAR, QA_band_unpacked == QA_WATER)
        ) / np.sum(QA_band_unpacked != QA_FILL)
    else:
        clear_ratio = 1

    if clear_ratio > clear_threshold:
        if folder[3] == "5":
            sensor = "LT5"
        elif folder[3] == "7":
            sensor = "LE7"
        elif folder[3] == "8":
            sensor = "LC8"
        elif folder[3] == "9":
            sensor = "LC9"
        elif folder[3] == "4":
            sensor = "LT4"
        else:
            logger.error(
                "Sensor is not correctly formatted for the scene {}".format(folder)
            )

        path = folder[10:13]
        row = folder[13:16]
        year = folder[17:21]
        doy = datetime(int(year), int(folder[21:23]), int(folder[23:25])).strftime("%j")
        collection = "C2"
        version = folder[len(folder) - 2 : len(folder)]
        file_name = sensor + path + row + year + doy + collection + version
        if low_date_bound is not None:
            if (dt.datetime(int(year), 1, 1) + dt.timedelta(int(doy) - 1)) < parse(
                low_date_bound
            ):
                return True

        if upp_date_bound is not None:
            if (dt.datetime(int(year), 1, 1) + dt.timedelta(int(doy) - 1)) > parse(
                upp_date_bound
            ):
                return True

        if sensor == "LT5" or sensor == "LE7" or sensor == "LT4":
            try:
                # b1 = rio_loaddata(join(join(tmp_path, folder),
                #                               "{}_SR_B1.TIF".format(folder)))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B1.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b1 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B2.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b2 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B3.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b3 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B4.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b4 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B5.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b5 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B7.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b6 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))

                rio_warp(
                    join(tmp_path, folder, f"{folder}_ST_B6.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b7 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
            except Exception as e:
                # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                logger.error("Cannot open Landsat bands for {}: {}".format(folder, e))
                return
        elif sensor == "LC8" or "LC9":
            try:
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B2.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b1 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B3.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b2 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B4.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b3 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B5.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b4 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B6.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b5 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
                rio_warp(
                    join(tmp_path, folder, f"{folder}_SR_B7.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b6 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))

                rio_warp(
                    join(tmp_path, folder, f"{folder}_ST_B10.TIF"),
                    join(tmp_path, folder, "_tmp_img.tif"),
                    reference_path,
                )
                b7 = rio_loaddata(join(tmp_path, folder, "_tmp_img.tif"))
            except Exception as e:
                # logger.error('Cannot open spectral bands for {}: {}'.format(folder, e))
                logger.error(f"Cannot open Landsat bands for {folder}: {e}")
                return

        if (
            (b1 is None)
            or (b2 is None)
            or (b3 is None)
            or (b4 is None)
            or (b5 is None)
            or (b6 is None)
            or (b7 is None)
        ):
            logger.error("Reading Landsat band fails for {}".format(folder))
            return

        # source: https://www.usgs.gov/faqs/how-do-i-use-scale-factor-landsat-level-2-science-products?qt-
        # news_science_products=0#qt-news_science_products recommended by yongquan
        B1_t = (10000 * (b1 * 2.75e-05 - 0.2)).astype(np.int16)
        B2_t = (10000 * (b2 * 2.75e-05 - 0.2)).astype(np.int16)
        B3_t = (10000 * (b3 * 2.75e-05 - 0.2)).astype(np.int16)
        B4_t = (10000 * (b4 * 2.75e-05 - 0.2)).astype(np.int16)
        B5_t = (10000 * (b5 * 2.75e-05 - 0.2)).astype(np.int16)
        B7_t = (10000 * (b7 * 2.75e-05 - 0.2)).astype(np.int16)
        B6_t = (10 * (b6 * 0.00341802 + 149)).astype(np.int16)

        # padding zeros
        b1 = np.zeros([dataset_info.n_rows, dataset_info.n_cols]).astype(np.int16)
        b1[0 : B1_t.shape[0], 0 : B1_t.shape[1]] = B1_t
        b2 = np.zeros([dataset_info.n_rows, dataset_info.n_cols]).astype(np.int16)
        b2[0 : B1_t.shape[0], 0 : B1_t.shape[1]] = B2_t
        b3 = np.zeros([dataset_info.n_rows, dataset_info.n_cols]).astype(np.int16)
        b3[0 : B1_t.shape[0], 0 : B1_t.shape[1]] = B3_t
        b4 = np.zeros([dataset_info.n_rows, dataset_info.n_cols]).astype(np.int16)
        b4[0 : B1_t.shape[0], 0 : B1_t.shape[1]] = B4_t
        b5 = np.zeros([dataset_info.n_rows, dataset_info.n_cols]).astype(np.int16)
        b5[0 : B1_t.shape[0], 0 : B1_t.shape[1]] = B5_t
        b6 = np.zeros([dataset_info.n_rows, dataset_info.n_cols]).astype(np.int16)
        b6[0 : B1_t.shape[0], 0 : B1_t.shape[1]] = B6_t
        b7 = np.zeros([dataset_info.n_rows, dataset_info.n_cols]).astype(np.int16)
        b7[0 : B1_t.shape[0], 0 : B1_t.shape[1]] = B7_t

        if is_partition is True:
            # width of a block
            bytesize = 2  # short16 = 2 * byte
            # source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-
            # in-python-using-numpy-d1bf0dd7b6f7
            B1_blocks = np.lib.stride_tricks.as_strided(
                b1,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B2_blocks = np.lib.stride_tricks.as_strided(
                b2,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B3_blocks = np.lib.stride_tricks.as_strided(
                b3,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B4_blocks = np.lib.stride_tricks.as_strided(
                b4,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B5_blocks = np.lib.stride_tricks.as_strided(
                b5,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B6_blocks = np.lib.stride_tricks.as_strided(
                b6,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            B7_blocks = np.lib.stride_tricks.as_strided(
                b7,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            QA_blocks = np.lib.stride_tricks.as_strided(
                QA_band_unpacked,
                shape=(
                    dataset_info.n_block_y,
                    dataset_info.n_block_x,
                    dataset_info.block_height,
                    dataset_info.block_width,
                ),
                strides=(
                    dataset_info.n_cols * dataset_info.block_height * bytesize,
                    dataset_info.block_width * bytesize,
                    dataset_info.n_cols * bytesize,
                    bytesize,
                ),
            )
            for i in range(dataset_info.n_block_y):
                for j in range(dataset_info.n_block_x):
                    # check if no valid pixels in the chip, then eliminate
                    qa_unique = np.unique(QA_blocks[i][j])

                    # skip blocks are all cloud, shadow or filled values
                    # in DHTC, we also don't need to save pixel that has qa value of 'QA_CLOUD',
                    # 'QA_SHADOW', or FILLED value (255)
                    if (
                        QA_CLEAR not in qa_unique
                        and QA_WATER not in qa_unique
                        and QA_SNOW not in qa_unique
                    ):
                        continue

                    block_folder = "block_x{}_y{}".format(j + 1, i + 1)
                    np.save(
                        join(join(out_dir, block_folder), file_name),
                        np.dstack(
                            [
                                B1_blocks[i][j],
                                B2_blocks[i][j],
                                B3_blocks[i][j],
                                B4_blocks[i][j],
                                B5_blocks[i][j],
                                B6_blocks[i][j],
                                B7_blocks[i][j],
                                QA_blocks[i][j],
                            ]
                        ).astype(np.int16),
                    )

        else:
            np.save(
                join(out_dir, file_name),
                np.dstack([b1, b2, b3, b4, b5, b6, b7, QA_band_unpacked]).astype(
                    np.int16
                ),
            )
        # scene_list.append(folder_name)
    else:
        # logger.info('Not enough clear observations for {}'.format(folder[0:len(folder) - 3]))
        logger.warn(
            "Not enough clear observations for {}".format(folder[0 : len(folder) - 3])
        )

    # delete unzip folder
    shutil.rmtree(join(tmp_path, folder), ignore_errors=True)
    shutil.rmtree(join(tmp_path, folder.replace("SR", "BT")), ignore_errors=True)


def checkfinished_step1(tmp_path):
    """
    :param tmp_path:
    :return:
    """
    if not os.path.exists(tmp_path):
        return False
    return True


def checkfinished_step2(out_dir, n_cores):
    """
    :param out_dir:
    :param n_cores:
    :return:
    """
    for i in range(n_cores):
        if not os.path.exists(join(out_dir, "rank{}_finished.txt".format(i + 1))):
            return False
    return True


def checkfinished_step3_partition(out_dir):
    """
    :param out_dir:
    :return:
    """
    if not os.path.exists(join(out_dir, "starting_last_dates.txt")):
        return False
    else:
        return True


def checkfinished_step3_nopartition(out_dir):
    """
    :param out_dir:
    :return:
    """
    if not os.path.exists(join(out_dir, "scene_list.txt")):
        return False
    return True


def get_extent(extent_geojson, res, buf=0):
    """
    read shapefile of a tile from an S3 bucket, and convert projection to be aligned with sample image.
    arg:
        'extent_geojson': sharply geojson object
        res: planet resolution
    return:
        (float, float, float, float), (int, int)) tuple
    """
    # txmin = min([row[0] for row in extent_geojson['coordinates'][0]]) - res / 2.0
    # txmax = max([row[0] for row in extent_geojson['coordinates'][0]]) + res / 2.0
    # tymin = min([row[1] for row in extent_geojson['coordinates'][0]]) - res / 2.0
    # tymax = max([row[1] for row in extent_geojson['coordinates'][0]]) + res / 2.0
    txmin = extent_geojson["bbox"][0] - res * buf
    txmax = extent_geojson["bbox"][2] + res * buf
    tymin = extent_geojson["bbox"][1] - res * buf
    tymax = extent_geojson["bbox"][3] + res * buf
    n_row = np.ceil((tymax - tymin) / res)
    n_col = np.ceil((txmax - txmin) / res)
    txmin_new = (txmin + txmax) / 2 - n_col / 2 * res
    txmax_new = (txmin + txmax) / 2 + n_col / 2 * res
    tymin_new = (tymin + tymax) / 2 - n_row / 2 * res
    tymax_new = (tymin + tymax) / 2 + n_row / 2 * res
    return (txmin_new, txmax_new, tymin_new, tymax_new), (n_row, n_col)


def get_feature(shp, id):
    for feature in shp:
        if feature["properties"]["id"] == id:
            return feature


def explode(coords):
    """Explode a GeoJSON geometry's coordinates object and yield coordinate tuples.
    As long as the input is conforming, the type of the geometry doesn't matter."""
    for e in coords:
        if isinstance(e, (float, int)):
            yield coords
            break
        else:
            for f in explode(e):
                yield f


def bbox(f):
    x, y = zip(*list(explode(f["geometry"]["coordinates"])))
    return min(x), min(y), max(x), max(y)


@click.command()
@click.option(
    "--source_dir",
    type=str,
    default=None,
    help="the folder directory of Landsat tar files " "downloaded from USGS website",
)
@click.option(
    "--out_dir", type=str, default=None, help="the folder directory for storing stacks"
)
@click.option(
    "--clear_threshold",
    type=float,
    default=0,
    help="threshold of clear pixel percentage for a tile, if lower than threshold, won't be processed",
)
@click.option(
    "--single_path",
    type=bool,
    default=True,
    help="indicate if using single_path or sidelap",
)
@click.option("--rank", type=int, default=1, help="the rank id")
@click.option("--n_cores", type=int, default=1, help="the total cores assigned")
@click.option(
    "--is_partition/--continuous", default=True, help="partition the output to blocks"
)
@click.option("--yaml_path", type=str, help="yaml file path")
@click.option(
    "--hpc/--dhtc", default=True, help="if it is set for HPC or DHTC environment"
)
@click.option(
    "--low_date_bound",
    type=str,
    default=None,
    help="the lower bound of the date range of user interest",
)
@click.option(
    "--upp_date_bound",
    type=str,
    default=None,
    help="the upper bound of the date range of user interest",
)
@click.option(
    "--collection",
    type=click.Choice(["ARD", "C2", "HLS", "HLS14", "ARD-C2"]),
    default="C2",
    help="image source category",
)
def main(
    source_dir,
    out_dir,
    clear_threshold,
    single_path,
    rank,
    n_cores,
    is_partition,
    yaml_path,
    hpc,
    low_date_bound,
    upp_date_bound,
    collection,
):
    if not os.path.exists(source_dir):
        print("Source directory not exists!")

    # select only _SR
    if collection == "ARD" or collection == "ARD-C2":
        folder_list = [
            f[0 : len(f) - 4]
            for f in listdir(source_dir)
            if (
                isfile(join(source_dir, f))
                and f.endswith(".tar")
                and f[len(f) - 6 : len(f) - 4] == "SR"
            )
        ]
    elif collection == "C2":
        folder_list = [
            f[0 : len(f) - 4]
            for f in listdir(source_dir)
            if (isfile(join(source_dir, f)) and f.endswith(".tar"))
        ]
    elif collection == "HLS":
        folder_list = [f for f in listdir(source_dir) if f.startswith("HLS")]
    elif collection == "HLS14":
        folder_list = [
            y for x in os.walk(source_dir) for y in glob(join(x[0], "*.hdf"))
        ]

    tmp_path = join(out_dir, "tmp{}".format(rank))

    tz = timezone("US/Eastern")
    with open(yaml_path) as yaml_obj:
        config_general = yaml.safe_load(yaml_obj)

    dataset_info = class_from_dict(DatasetInfo, config_general["DATASETINFO"])

    logger = None

    # adjust row and columns if they are not divisible by block height and width
    if collection == "C2":
        if os.path.exists(join(tmp_path, folder_list[0])):
            shutil.rmtree(join(tmp_path, folder_list[0]), ignore_errors=True)
        if not os.path.exists(join(tmp_path, "ref_folder")):
            with tarfile.open(join(source_dir, folder_list[0] + ".tar")) as tar_ref:
                tar_ref.extractall(join(tmp_path, "ref_folder"))
        ref_path = join(tmp_path, "ref_folder", f"{folder_list[0]}_SR_B1.TIF")
        ref_image = rio_loaddata(ref_path)
        if ref_image.shape[0] % dataset_info.block_height > 0:
            dataset_info.n_block_y = (
                int(ref_image.shape[0] / dataset_info.block_height) + 1
            )
            dataset_info.n_rows = dataset_info.block_height * dataset_info.n_block_y

        if ref_image.shape[1] % dataset_info.block_width > 0:
            dataset_info.n_block_x = (
                int(ref_image.shape[1] / dataset_info.block_width) + 1
            )
            dataset_info.n_cols = dataset_info.block_width * dataset_info.n_block_x
        ref_image = None

    # create needed folders
    if rank == 1:
        # mode = w enables the log file to be overwritten
        logging.basicConfig(
            filename=join(os.getcwd(), "prepare_ard.log"),
            filemode="w",
            level=logging.INFO,
        )
        logger = logging.getLogger(__name__)

        logger.info(
            "AutoPrepareDataARD starts: {}".format(
                datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            )
        )

        # create folder
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(tmp_path).mkdir(parents=True, exist_ok=True)

        # if the nrows are divisible by block_height, need to adjust dataset info
        if collection == "C2":
            ref_image = rio_loaddata(ref_path)
            if ref_image.shape[0] % dataset_info.block_height > 0:
                dataset_info.n_block_y = (
                    int(ref_image.shape[0] / dataset_info.block_height) + 1
                )
                dataset_info.n_rows = dataset_info.block_height * dataset_info.n_block_y

            if ref_image.shape[1] % dataset_info.block_width > 0:
                dataset_info.n_block_x = (
                    int(ref_image.shape[1] / dataset_info.block_width) + 1
                )
                dataset_info.n_cols = dataset_info.block_width * dataset_info.n_block_x
            ref_image = None

        if is_partition is True:
            for i in range(dataset_info.n_block_y):
                for j in range(dataset_info.n_block_x):
                    Path(join(out_dir, f"block_x{j + 1}_y{i + 1}")).mkdir(
                        parents=True, exist_ok=True
                    )

        if collection == "ARD" or collection == "ARD-C2":
            if hpc is True:
                if os.path.exists(join(tmp_path, folder_list[0])):
                    shutil.rmtree(join(tmp_path, folder_list[0]), ignore_errors=True)
                with tarfile.open(join(source_dir, folder_list[0] + ".tar")) as tar_ref:
                    try:
                        tar_ref.extractall(join(tmp_path, folder_list[0]))
                    except Exception:
                        logger.error("Unzip fails for {}".format(folder_list[0]))
                if collection == "ARD":
                    ref_path = join(
                        tmp_path, folder_list[0], "{}b1.tif".format(folder_list[0])
                    )
                else:
                    ref_path = join(
                        tmp_path, folder_list[0], "{}_B1.TIF".format(folder_list[0])
                    )

                # warp a tile-based single path tif
                with importlib_resources.path(
                    "pyxccd.imagetool", "singlepath_landsat_conus.tif"
                ) as conus_image_fpath:
                    # conus_image_fpath = (Path(__file__).parent / 'singlepath_landsat_conus.tif').resolve()
                    rio_warp(
                        conus_image_fpath,
                        join(out_dir, "singlepath_landsat_tile.tif"),
                        ref_path,
                    )
                shutil.rmtree(join(tmp_path, folder_list[0]), ignore_errors=True)
                ordinal_dates = [
                    pd.Timestamp.toordinal(
                        dt.datetime(
                            int(folder[15:19]), int(folder[19:21]), int(folder[21:23])
                        )
                    )
                    for folder in folder_list
                ]
        elif collection == "C2":
            ordinal_dates = [
                pd.Timestamp.toordinal(
                    dt.datetime(
                        int(folder[17:21]), int(folder[21:23]), int(folder[23:25])
                    )
                )
                for folder in folder_list
            ]
        elif collection == "HLS":
            ordinal_dates = [
                pd.Timestamp.toordinal(dt.datetime(int(folder[15:19]), 1, 1))
                + int(folder[19:22])
                - 1
                for folder in folder_list
            ]
        elif collection == "HLS14":
            ordinal_dates = [
                pd.Timestamp.toordinal(
                    dt.datetime(int(folder.split("/")[-1][15:19]), 1, 1)
                )
                + int(folder.split("/")[-1][19:22])
                - 1
                for folder in folder_list
            ]
        ordinal_dates.sort()
        file = open(
            join(out_dir, "starting_last_dates.txt"), "w+"
        )  # need to save out starting and
        # lasting date for this tile
        if low_date_bound is not None:
            tmp = pd.Timestamp.toordinal(parse(low_date_bound))
        else:
            tmp = 0
        file.writelines("{}\n".format(str(np.max([ordinal_dates[0], tmp]))))
        if upp_date_bound is not None:
            tmp = pd.Timestamp.toordinal(parse(upp_date_bound))
        else:
            tmp = 999999
        file.writelines("{}\n".format(str(np.min([ordinal_dates[-1], tmp]))))
        file.close()
    else:
        # mode = w enables the log file to be overwritten
        logging.basicConfig(
            filename=join(os.getcwd(), "prepare_ard.log"),
            filemode="a",
            level=logging.INFO,
        )
        logger = logging.getLogger(__name__)

    # use starting_last_dates.txt to indicate all folder have been created. Very stupid way. Need improve it in future.
    while not checkfinished_step1(join(out_dir, "starting_last_dates.txt")):
        time.sleep(5)

    # read a general path file which can indicate which pixel is assigned to which path
    path_array = None
    if single_path is True:
        if collection == "ARD" or collection == "ARD-C2":
            # parse tile h and v from folder name
            folder_name = os.path.basename(source_dir)
            if hpc is True:
                path_array = rio_loaddata(join(out_dir, "singlepath_landsat_tile.tif"))
            else:
                try:
                    tile_h = int(
                        folder_name[
                            folder_name.find("h") + 1 : folder_name.find("h") + 4
                        ]
                    )
                    tile_v = int(
                        folder_name[
                            folder_name.find("v") + 1 : folder_name.find("v") + 4
                        ]
                    )
                    path_array = rio_loaddata(
                        join(
                            Path(os.path.realpath(__file__)).parent,
                            "singlepath_landsat_tile_crop_compress.tif",
                        ),
                        xoff=tile_h * dataset_info.n_cols,
                        yoff=tile_v * dataset_info.n_rows,
                        xsize=dataset_info.n_cols,
                        ysize=dataset_info.n_rows,
                    )  # read a partial array
                    # from a large file
                except IOError as e:
                    logger.error(e)
                    exit()

    if collection == "C2":
        # gpd_tile = gpd.read_file(shapefile_path)

        pool = Pool(n_cores)
        single_image_stacking_collection2_partial = functools.partial(
            single_image_stacking_collection2,
            tmp_path,
            source_dir,
            out_dir,
            clear_threshold,
            logger,
            dataset_info,
            ref_path,
            is_partition,
            low_date_bound,
            upp_date_bound,
        )
        pool.map(single_image_stacking_collection2_partial, folder_list)

        # for i in range(int(np.ceil(len(folder_list) / n_cores))):
        #     new_rank = rank - 1 + i * n_cores
        #     # means that all folder has been processed
        #     if new_rank > (len(folder_list) - 1):
        #         break
        #     folder = folder_list[new_rank]
        #     single_image_stacking_collection2(
        #         tmp_path,
        #         source_dir,
        #         out_dir,
        #         clear_threshold,
        #         logger,
        #         dataset_info,
        #         ref_path,
        #         is_partition=is_partition,
        #         low_date_bound=low_date_bound,
        #         upp_date_bound=upp_date_bound,
        #         folder
        #     )
    elif collection == "ARD" or collection == "ARD-C2":
        pool = Pool(n_cores)
        single_image_stacking_partial = functools.partial(
            single_image_stacking,
            tmp_path,
            source_dir,
            out_dir,
            clear_threshold,
            path_array,
            logger,
            dataset_info,
            is_partition,
            low_date_bound,
            upp_date_bound,
        )
        pool.map(single_image_stacking_partial, folder_list)

        # assign files to each core
        # for i in range(int(np.ceil(len(folder_list) / n_cores))):
        #     new_rank = rank - 1 + i * n_cores
        #     # means that all folder has been processed
        #     if new_rank > (len(folder_list) - 1):
        #         break
        #     folder = folder_list[new_rank]
        #     single_image_stacking(
        #         tmp_path,
        #         source_dir,
        #         out_dir,
        #         folder,
        #         clear_threshold,
        #         path_array,
        #         logger,
        #         dataset_info,
        #         is_partition=is_partition,
        #         low_date_bound=low_date_bound,
        #         upp_date_bound=upp_date_bound,
        #     )
    elif collection == "HLS":
        pool = Pool(n_cores)
        single_image_stacking_hls_partial = functools.partial(
            single_image_stacking_hls,
            source_dir,
            out_dir,
            logger,
            dataset_info,
            clear_threshold,
            is_partition,
            low_date_bound,
            upp_date_bound,
        )
        pool.map(single_image_stacking_hls_partial, folder_list)
        # assign files to each core
        # for i in range(int(np.ceil(len(folder_list) / n_cores))):
        #     new_rank = rank - 1 + i * n_cores
        #     # means that all folder has been processed
        #     if new_rank > (len(folder_list) - 1):
        #         break
        #     folder = folder_list[new_rank]
        #     single_image_stacking_hls(
        #         source_dir,
        #         out_dir,
        #         logger,
        #         dataset_info,
        #         folder,
        #         clear_threshold=clear_threshold,
        #         is_partition=is_partition,
        #         low_date_bound=low_date_bound,
        #         upp_date_bound=upp_date_bound,
        #     )
    # elif collection == "HLS14":
    #     pool = Pool(n_cores)
    #     single_image_stacking_hls14_partial = functools.partial(
    #         single_image_stacking_hls, out_dir,
    #             logger,
    #             dataset_info,
    #             clear_threshold,
    #             is_partition,
    #             low_date_bound,
    #             upp_date_bound
    #     )
    #     pool.map(single_image_stacking_hls14_partial,  folder_list)
    # assign files to each core
    # for i in range(int(np.ceil(len(folder_list) / n_cores))):
    #     new_rank = rank - 1 + i * n_cores
    #     # means that all folder has been processed
    #     if new_rank > (len(folder_list) - 1):
    #         break
    #     folder = folder_list[new_rank]
    #     single_image_stacking_hls14(
    #         out_dir,
    #         logger,
    #         dataset_info,
    #         folder,
    #         clear_threshold=clear_threshold,
    #         is_partition=is_partition,
    #         low_date_bound=low_date_bound,
    #         upp_date_bound=upp_date_bound,
    #     )
    # create an empty file for signaling the core that has been finished
    with open(os.path.join(out_dir, "rank{}_finished.txt".format(rank)), "w"):
        pass

    # wait for other cores assigned
    # while not checkfinished_step2(out_dir, n_cores):
    #     time.sleep(5)
    shutil.rmtree(tmp_path, ignore_errors=True)
    # create scene list after stacking is finished and remove folders. Not need it in DHTC
    if rank == 1:
        # remove tmp folder
        logger.info(
            "Stacking procedure finished: {}".format(
                datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            )
        )


if __name__ == "__main__":
    main()
