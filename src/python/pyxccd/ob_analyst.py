import numpy as np
import pandas as pd
import os
from typing import Optional, List, Any, Union
import datetime as dt

# import datetime
from cv2 import floodFill
import cv2 as cv2
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from scipy import stats
from scipy.stats import chi2
from os.path import join, exists
from logging import Logger
from pyxccd.utils import (
    get_block_x,
    get_block_y,
    get_rowcol_intile,
    modeby,
    mode_median_by,
)
from pyxccd.app import defaults
from pyxccd import obcold_reconstruct
from skimage.segmentation import slic
from skimage.measure import label as sklabel
from skimage.segmentation import watershed
import logging
import sys
from pyxccd.common import DatasetInfo


NAN_VAL = -9999


def _read_blockdata(block_folder, total_pixels, total_bands):
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


def _cmname_fromdate(ordinal_date):
    """
    get file name from ordinate date
    Args:
        ordinal_date: the inputted ordinate date
    Returns:
        CM file name
    """
    return "CM_maps_{}_{}{}".format(
        str(ordinal_date),
        pd.Timestamp.fromordinal(ordinal_date).year,
        str(pd.Timestamp.fromordinal(ordinal_date).timetuple().tm_yday).zfill(3),
    )


def _cmdatename_fromdate(ordinal_date):
    """
    get date file name from ordinate date
    Args:
        ordinal_date: the inputted ordinate date
    Returns:
        CM date file name
    """
    return "CM_date_maps_{}_{}{}".format(
        str(ordinal_date),
        pd.Timestamp.fromordinal(ordinal_date).year,
        str(pd.Timestamp.fromordinal(ordinal_date).timetuple().tm_yday).zfill(3),
    )


def _obiaresname_fromdate(ordinal_date):
    """
    get OBIA result file name from ordinate date
    Args:
        ordinal_date: the inputted ordinate date
    Returns:
        CM file name
    """
    return "obiaresult_{}_{}{}".format(
        str(ordinal_date),
        pd.Timestamp.fromordinal(ordinal_date).year,
        str(pd.Timestamp.fromordinal(ordinal_date).timetuple().tm_yday).zfill(3),
    )


def _is_change_object(
    stats_lut_row: dict,
    keyword: str,
    classification_map: np.ndarray,
    uniform_threshold=None,
    uniform_sizeslope=None,
    parameters=None,
):
    """_summary_

    Parameters
    ----------
    stats_lut_row : dict
        a dictionary that has 'npixels', 'keyword', 'mode'

    keyword : str
        keyword to calculate change magnitude, 'cm_average' or 'cm_median'
    classification_map : np.ndarray
        labek map
    parameters : _type_, optional
        define size slope and intercept, by default None

    Returns
    -------
    bool
        _description_
    """
    # LCMAP category id
    # developed_id: 1
    # cropland_id: 2
    # grass_id: 3
    # tree_id: 4
    # water_id: 5
    # wetland_id: 6
    # ice: 7
    # barren: 8
    if parameters is None:
        parameters = defaults["OBCOLD"]

    log10_size = np.log10(int(stats_lut_row["npixels"]))
    intercept = 0.95
    if classification_map is None:
        if log10_size * parameters["default_sizeslope"] + intercept < 2:
            scale = log10_size * parameters["default_sizeslope"] + intercept
        else:
            scale = 2
        if np.double(stats_lut_row[keyword]) * scale > parameters["default_threshold"]:
            return True
        else:
            return False
    else:
        if uniform_threshold is not None and uniform_sizeslope is not None:
            # intercept = 1 - uniform_sizeslope
            if log10_size * uniform_sizeslope + intercept < 2:
                scale = log10_size * uniform_sizeslope + intercept
            else:
                scale = 2
            if np.double(stats_lut_row[keyword]) * scale > uniform_threshold:
                return True
            else:
                return False
        else:
            if int(stats_lut_row["mode"]) == 255:
                # intercept = 1 - config['default_sizeslope']
                if log10_size * parameters["default_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["default_sizeslope"] + intercept
                else:
                    scale = 2
                if (
                    float(stats_lut_row[keyword]) * scale
                    > parameters["default_threshold"]
                ):
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 1:  # develop in LCMAP
                # intercept = 1 - config['C1_sizeslope']
                if log10_size * parameters["C1_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C1_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C1_threshold"]:
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 2:  # cropland in LCMAP
                # intercept = 1 - config['C2_sizeslope']
                if log10_size * parameters["C2_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C2_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C2_threshold"]:
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 3:  # grassland in LCMAP
                # intercept = 1 - config['C3_sizeslope']
                if log10_size * parameters["C3_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C3_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C3_threshold"]:
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 4:  # forest in LCMAP
                # intercept = 1 - config['C4_sizeslope']
                if log10_size * parameters["C4_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C4_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C4_threshold"]:
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 5:  # water in LCMAP
                # intercept = 1 - config['C5_sizeslope']
                if log10_size * parameters["C5_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C5_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C5_threshold"]:
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 6:  # wetland in LCMAP
                # intercept = 1 - config['C6_sizeslope']
                if log10_size * parameters["C6_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C6_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C6_threshold"]:
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 7:  # ice in LCMAP
                # intercept = 1 - config['C7_sizeslope']
                if log10_size * parameters["C7_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C7_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C7_threshold"]:
                    return True
                else:
                    return False
            elif int(stats_lut_row["mode"]) == 8:  # barren in LCMAP
                # intercept = 1 - config['C8_sizeslope']
                if log10_size * parameters["C8_sizeslope"] + intercept < 2:
                    scale = log10_size * parameters["C8_sizeslope"] + intercept
                else:
                    scale = 2
                if float(stats_lut_row[keyword]) * scale > parameters["C8_threshold"]:
                    return True
                else:
                    return False


def segmentation_floodfill(
    cm_array: np.ndarray,
    cm_date_array: np.ndarray,
    cm_array_l1: Optional[np.ndarray] = None,
    cm_array_l1_date: Optional[np.ndarray] = None,
    floodfill_ratio: Optional[float] = None,
    parameters: Optional[dict] = None,
    b_dist_prob_map: Optional[bool] = False,
    date_interval: int = 60,
    peak_threshold: Optional[float] = None,
):
    """hierachical segmentation based on floodfill

    Parameters
    ----------
    cm_array: np.ndarray
        2-d numpy array, change magnitude array
    cm_date_array: 2-d numpy array, change date array
    cm_array_l1: np.ndarray
        2-d numpy array
    cm_array_l1_date: np.ndarray
        2-d numpy array
    floodfill_ratio: float
        the change magnitude ratio of the considered pixel over the seed pixel to be included into the cluster
    parameters: dict
        OBCOLD Parameters
    b_dist_prob_map: boolean
        if true, cm_array is probability
    date_interval: int
        the date interval for cm_date_array to connect adjacent pixels within the floodfill process
    peak_threshold:float
        the threshold defining local peaks

    Returns
    -------
    tuple
    [object_map_s1, cm_date_array, object_map_s2, s1_info]:
        object_map_s1: object map for superpixel,
        cm_date_array: change date maps which is updated by filling na values from the previous change date map,
                       it is generated as an input for saving out the new break date map (we need to know the date for
                       the new break to generate the new temporal segments and harmonic coefficients)
        object_map_s2: object map for object level
        s1_info: a zipped list of id and average change magnitude in the superpixel level
    """
    # cm_array = cm_tile
    # cm_date_array = cmdate_tile
    # cm_array_l1 = None
    # cm_array_l1_date = None
    # floodfill_ratio = None
    if parameters is None:
        parameters = defaults["OBCOLD"]

    [n_rows, n_cols] = cm_array.shape
    if floodfill_ratio is None:
        floodfill_ratio = parameters["floodfill_thres"]
    if cm_array_l1_date is None:
        cm_array_l1_date = np.full((n_rows, n_cols), NAN_VAL, dtype=np.int32)

    # assign valid CM values in the stack into current cm array where NA values are
    if b_dist_prob_map:
        if peak_threshold is None:
            peak_threshold = 0.5
        if cm_array_l1 is None:
            cm_array_l1 = np.full((n_rows, n_cols), 0, dtype=np.float32)
        cm_array_l1[cm_array_l1 < peak_threshold] = 0
        cm_array[cm_array == 0] = cm_array_l1[cm_array == 0]
    else:
        if peak_threshold is None:
            peak_threshold = chi2.ppf(0.90, 5)
        if cm_array_l1 is None:
            cm_array_l1 = np.full((n_rows, n_cols), NAN_VAL, dtype=np.int16)
        cm_array_l1[cm_array_l1 < (chi2.ppf(0.90, 5) * parameters["cm_scale"])] = (
            NAN_VAL
        )
        cm_array[cm_array == NAN_VAL] = cm_array_l1[cm_array == NAN_VAL]
        cm_array = cm_array.astype(float) / parameters["cm_scale"]

    cm_date_array[cm_date_array == NAN_VAL] = cm_array_l1_date[cm_date_array == NAN_VAL]

    # free memory
    del cm_array_l1
    del cm_array_l1_date

    #######################################################################################
    #                               Scale 1: change superpixel                            #
    #######################################################################################
    cm_array[cm_array < peak_threshold * (1 - floodfill_ratio)] = np.nan
    # cm_array[cm_array < chi2.ppf(0.7, 5)] = np.nan
    bandwidth = 1

    # using gaussian kernel ( with 1 sigma value) to smooth images in hpc_preparation for floodfill
    kernel = Gaussian2DKernel(x_stddev=bandwidth, y_stddev=bandwidth)
    cm_array_gaussian_s1 = convolve(
        cm_array, kernel, boundary="extend", preserve_nan=True
    )
    if b_dist_prob_map:
        cm_array_gaussian_s1[np.isnan(cm_array)] = 0
    else:
        cm_array_gaussian_s1[np.isnan(cm_array)] = NAN_VAL
    del cm_array

    # seed_index = peak_local_max(cm_array_gaussian_s1, threshold_abs=peak_threshold,
    #                             exclude_border=False, min_distance=0)

    seed_index = np.where(cm_array_gaussian_s1 > peak_threshold)
    cm_seed = cm_array_gaussian_s1[seed_index]
    zip_list = sorted(
        zip(cm_seed, np.transpose(seed_index)), key=lambda t: t[0], reverse=True
    )
    seed_index = [x[1] for x in zip_list]  # type: ignore
    # seed_index = np.flip(seed_index, axis=0)

    mask_s1 = np.zeros((n_rows + 2, n_cols + 2)).astype(np.uint8)
    mask_label_s1 = np.zeros((n_rows + 2, n_cols + 2))
    floodflags_base = 8
    floodflags_base |= cv2.FLOODFILL_MASK_ONLY
    floodflags_base |= cv2.FLOODFILL_FIXED_RANGE
    cm_stack = np.dstack(
        [cm_array_gaussian_s1, cm_array_gaussian_s1, cm_date_array]
    ).astype(np.float32)
    for i in range(len(seed_index)):
        # print(i)
        remainder = i % 255
        floodflags = floodflags_base | ((remainder + 1) << 8)
        seedcm = cm_array_gaussian_s1[tuple(seed_index[i])]
        # gradiant = np.max([seedcm*floodfill_ratio, (seedcm-peak_threshold)])
        gradiant = seedcm * floodfill_ratio
        num, im, mask_s1, rect = floodFill(
            cm_stack,
            mask_s1,
            tuple(reversed(seed_index[i])),
            0,
            loDiff=[gradiant, gradiant, date_interval],
            upDiff=[gradiant, gradiant, date_interval],
            flags=floodflags,
        )
        # the opencv mask only supports 8-bit, we hack it by updating the label value for every 255 object
        if remainder == 254:
            no = int(i / 255)
            mask_label_s1[(mask_label_s1 == 0) & (mask_s1 > 0)] = (
                mask_s1[(mask_label_s1 == 0) & (mask_s1 > 0)].astype(int) + no * 255
            )
        # flood_fill(cm_array_log, tuple(reversed(seed_index[i])), new_value=-np.ceil(floodfill_threshold)*2-i,
        #            tolerance=floodfill_threshold, connectivity=2, in_place=True)

    if len(seed_index) > 0:
        # processing the rest that hasn't be recorded into mask_label
        no = int(i / 255)
        mask_label_s1[(mask_label_s1 == 0) & (mask_s1 > 0)] = (
            mask_s1[(mask_label_s1 == 0) & (mask_s1 > 0)].astype(int) + no * 255
        )

    object_map_s1 = mask_label_s1[1 : n_rows + 1, 1 : n_cols + 1].astype(np.int32)

    # free memory
    del cm_stack
    del mask_label_s1
    del mask_s1
    #######################################################################################
    #                                 Scale 2: change patch                              #
    #######################################################################################

    # superpixel-level object status
    # s1_info = pd.DataFrame(regionprops_table(object_map_s1, cm_array_gaussian_s1, cache=False,
    #                                          properties=['label', 'mean_intensity']))

    # unique always returned sorted list
    unq_s1, ids_s1, count_s1 = np.unique(
        object_map_s1, return_inverse=True, return_counts=True
    )
    mean_list = (
        np.bincount(
            ids_s1.flatten().astype(int), weights=cm_array_gaussian_s1.reshape(ids_s1.flatten().shape)
        )
        / count_s1
    )
    # mean_list = np.sqrt(np.bincount(ids_s1.astype(int),
    # weights=np.square(cm_array_gaussian_s1.astype(np.int64)).reshape(ids_s1.shape)) / count_s1)
    mean_list[unq_s1 == 0] = NAN_VAL  # force mean of unchanged objects to be -9999
    s1_info = pd.DataFrame({"label": unq_s1, "mean_intensity": mean_list})
    object_map_s2 = object_map_s1.copy()
    object_map_s2[object_map_s2 > 0] = 1
    object_map_s2 = sklabel(object_map_s2, connectivity=2, background=0)
    # object_map_s2 = object_map_s1   # only superpixel
    # for only patch level
    # unq_s1, ids_s1, count_s1 = np.unique(object_map_s2, return_inverse=True, return_counts=True)
    # mean_list = np.bincount(ids_s1.astype(int), weights=cm_array_gaussian_s1.reshape(ids_s1.shape)) / count_s1
    # mean_list = np.sqrt(np.bincount(ids_s1.astype(int),
    # weights=np.square(cm_array_gaussian_s1.astype(np.int64)).reshape(ids_s1.shape)) / count_s1)
    # mean_list[unq_s1 == 0] = NAN_VAL  # force mean of unchanged objects to be -9999
    # s1_info = pd.DataFrame({'label': unq_s1, 'mean_intensity': mean_list})
    # object_map_s1 = object_map_s2.copy()
    return object_map_s1, cm_date_array, object_map_s2, s1_info


# def normalize_clip(data, min, max, na_val=None):
#     if max == min:
#         tmp = np.full_like(data, fill_value=0)
#     else:
#         tmp = (data - min) / (max - min)
#         np.clip(tmp, 0, 1, out=tmp)
#     if na_val is not None:
#         tmp[data == na_val] = na_val
#     return tmp


def segmentation_slic(
    cm_array: np.ndarray,
    cm_date_array: np.ndarray,
    cm_array_l1=None,
    cm_array_l1_date=None,
    low_bound=None,
    parameters=None,
):
    """
    hierachical segmentation based on slic
    Parameters
    ----------
    cm_array: 2-d numpy array, change magnitude array
    cm_date_array: 2-d numpy array, change date array
    cm_array_l1: 2-d numpy array
    cm_array_l1_date: 2-d numpy array
    low_bound: float
        the change magnitude ratio of the considered pixel over the seed pixel to be included into the cluster

    Returns
    -------
    [object_map_s1, cm_date_array, object_map_s2, s1_info]:
        object_map_s1: object map for superpixel,
        cm_date_array: change date maps which is updated by filling na values from the previous change date map,
                       it is generated as an input for saving out the new break date map (we need to know the date for
                       the new break to generate the new temporal segments and harmonic coefficients)
        object_map_s2: object map for object level
        s1_info: a zipped list of id and average change magnitude in the superpixel level
    """
    if parameters is None:
        parameters = defaults["OBCOLD"]

    if low_bound is None:
        low_bound = chi2.ppf(parameters["gate_probability"], 5)
    else:
        low_bound = low_bound
    [n_rows, n_cols] = cm_array.shape
    if cm_array_l1 is None:
        cm_array_l1 = np.full((n_rows, n_cols), NAN_VAL, dtype=np.int16)
    if cm_array_l1_date is None:
        cm_array_l1_date = np.full((n_rows, n_cols), NAN_VAL, dtype=np.int32)

    # assign valid CM values in the stack into current cm array where NA values are
    cm_array[cm_array == NAN_VAL] = cm_array_l1[cm_array == NAN_VAL]
    cm_array = cm_array.astype(float) / parameters["cm_scale"]

    cm_date_array[cm_date_array == NAN_VAL] = cm_array_l1_date[cm_date_array == NAN_VAL]

    #######################################################################################
    #                               Scale 1: change superpixel                            #
    #######################################################################################
    cm_array[cm_array < low_bound] = np.nan
    bandwidth = 1

    # using gaussian kernel ( with 1 sigma value) to smooth images in hpc_preparation for floodfill
    kernel = Gaussian2DKernel(x_stddev=bandwidth, y_stddev=bandwidth)
    cm_array_gaussian_s1 = convolve(
        cm_array, kernel, boundary="extend", preserve_nan=True
    )
    cm_array_gaussian_s1[np.isnan(cm_array)] = NAN_VAL

    mask = np.full_like(cm_array_gaussian_s1, fill_value=0)
    mask[cm_array_gaussian_s1 > low_bound] = 1

    # cm_date_selected = cm_date_array[cm_date_array > 0]
    # if len(cm_date_selected) > 0:
    #     cm_date_min = np.min(cm_date_selected)
    #     cm_date_max = np.max(cm_date_selected)
    # else:
    #     cm_date_min = NAN_VAL
    #     cm_date_max = NAN_VAL
    cm_stack = cm_array_gaussian_s1

    n_segments = int(np.ceil(len(mask[mask == 1]) / 25))
    ell = np.unique(sklabel(mask))
    # n_segments = len(ell) * 2
    # if n_segments == 0:
    #    object_map_s1 = np.full_like(mask, 0)
    if len(ell) == 1 and ell[0] == 0:
        object_map_s1 = np.full_like(mask, 0)
    else:
        object_map_s1 = slic(
            cm_stack,
            mask=mask,
            n_segments=n_segments,
            compactness=0.01,
            min_size_factor=n_segments / cm_stack.shape[0] / cm_stack.shape[1],
        )

    # superpixel-level object status
    # s1_info = pd.DataFrame(regionprops_table(object_map_s1, cm_array_gaussian_s1, cache=False,
    #                                          properties=['label', 'mean_intensity']))
    unq_s1, ids_s1, count_s1 = np.unique(
        object_map_s1, return_inverse=True, return_counts=True
    )
    mean_list = (
        np.bincount(
            ids_s1.astype(int), weights=cm_array_gaussian_s1.reshape(ids_s1.shape)
        )
        / count_s1
    )
    # mean_list = np.sqrt(np.bincount(ids_s1.astype(int), weights=np.square(cm_array_gaussian_s1.astype(np.int64)).reshape(ids_s1.shape)) / count_s1)
    mean_list[unq_s1 == 0] = NAN_VAL  # force mean of unchanged objects to be -9999
    s1_info = pd.DataFrame({"label": unq_s1, "mean_intensity": mean_list})
    #######################################################################################
    #                                 Scale 2: change patch                              #
    #######################################################################################
    object_map_s2 = object_map_s1.copy()
    object_map_s2[object_map_s2 > 0] = 1
    object_map_s2 = sklabel(object_map_s2, connectivity=2, background=0)
    # seed_index = peak_local_max(cm_array_gaussian_s1, threshold_abs=11.07,
    #                             exclude_border=False, min_distance=0)
    # lut_dict_s1 = dict(zip(unq_s1, mean_list))
    # cm_array_s2 = np.vectorize(lut_dict_s1.get)(object_map_s1).astype(np.float32)
    # bandwidth = 1
    # mask_s2 = np.zeros((n_rows + 2, n_cols + 2)).astype(np.uint8)
    # mask_label_s2 = np.zeros((n_rows + 2, n_cols + 2))
    # floodflags_base = 8
    # floodflags_base |= cv2.FLOODFILL_MASK_ONLY
    # for i in range(len(seed_index)):
    #     # print(i)
    #    remainder = i % 255
    #     floodflags = floodflags_base | ((remainder + 1) << 8)
    #    num, im, mask_s2, rect = floodFill(cm_array_gaussian_s2, mask_s2, tuple(reversed(seed_index[i])), 0,
    #                                       loDiff=[10],
    #                                       upDiff=[10],
    #                                       flags=floodflags)  # assign an extremely large value
    #   # to connect any polygon adjacent to each other
    #   # the opencv mask only supports 8-bit, so every 255 seed needs to update values in mask_label
    #    if remainder == 254:
    #        no = int(i / 255)
    #        mask_label_s2[(mask_label_s2 == 0) & (mask_s2 > 0)] = mask_s2[
    #                                                                  (mask_label_s2 == 0) & (mask_s2 > 0)].\
    #                                                                  astype(int) + no * 255

    # if len(seed_index) > 0:
    #   # processing the rest that hasn't be recorded into mask_label
    #    no = int(i / 255)
    #    mask_label_s2[(mask_label_s2 == 0) & (mask_s2 > 0)] = mask_s2[(mask_label_s2 == 0) & (mask_s2 > 0)].astype(
    #        int) + no * 255
    # object_map_s2 = mask_label_s2[1:n_rows + 1, 1:n_cols + 1]

    return object_map_s1, cm_date_array, object_map_s2, s1_info


def segmentation_watershed(
    cm_array,
    cm_date_array,
    cm_array_l1=None,
    cm_array_l1_date=None,
    low_bound=None,
    parameters=None,
):
    """
    hierachical segmentation based on watershed
    Parameters
    ----------
    cm_array: 2-d numpy array, change magnitude array
    cm_date_array: 2-d numpy array, change date array
    cm_array_l1: 2-d numpy array
    cm_array_l1_date: 2-d numpy array
    low_bound: float
        the change magnitude ratio of the considered pixel over the seed pixel to be included into the cluster
    Returns
    -------
    [object_map_s1, cm_date_array, object_map_s2, s1_info]:
        object_map_s1: object map for superpixel,
        cm_date_array: change date maps which is updated by filling na values from the previous change date map,
                       it is generated as an input for saving out the new break date map (we need to know the date for
                       the new break to generate the new temporal segments and harmonic coefficients)
        object_map_s2: object map for object level
        s1_info: a zipped list of id and average change magnitude in the superpixel level
    """
    if parameters is None:
        parameters = defaults["OBCOLD"]

    if low_bound is None:
        low_bound = chi2.ppf(parameters["gate_probability"], 5)
    else:
        low_bound = low_bound
    [n_rows, n_cols] = cm_array.shape
    if cm_array_l1 is None:
        cm_array_l1 = np.full((n_rows, n_cols), NAN_VAL, dtype=np.int16)
    if cm_array_l1_date is None:
        cm_array_l1_date = np.full((n_rows, n_cols), NAN_VAL, dtype=np.int32)

    # assign valid CM values in the stack into current cm array where NA values are
    cm_array[cm_array == NAN_VAL] = cm_array_l1[cm_array == NAN_VAL]
    cm_array = cm_array.astype(float) / parameters["cm_scale"]

    cm_date_array[cm_date_array == NAN_VAL] = cm_array_l1_date[cm_date_array == NAN_VAL]

    #######################################################################################
    #                               Scale 1: change superpixel                            #
    #######################################################################################
    cm_array[cm_array < low_bound] = np.nan
    bandwidth = 1

    # using gaussian kernel ( with 1 sigma value) to smooth images in hpc_preparation for floodfill
    kernel = Gaussian2DKernel(x_stddev=bandwidth, y_stddev=bandwidth)
    cm_array_gaussian_s1 = convolve(
        cm_array, kernel, boundary="extend", preserve_nan=True
    )
    cm_array_gaussian_s1[np.isnan(cm_array)] = NAN_VAL

    mask = np.full_like(cm_array_gaussian_s1, fill_value=0)
    mask[cm_array_gaussian_s1 > low_bound] = 1

    object_map_s1 = watershed(
        -cm_array_gaussian_s1, connectivity=2, compactness=0, mask=mask
    )
    # enforce lower probability pixel to be 0
    object_map_s1[cm_array_gaussian_s1 < low_bound] = 0

    # superpixel-level object status
    # s1_info = pd.DataFrame(regionprops_table(object_map_s1, cm_array_gaussian_s1, cache=False,
    #                                          properties=['label', 'mean_intensity']))
    unq_s1, ids_s1, count_s1 = np.unique(
        object_map_s1, return_inverse=True, return_counts=True
    )
    mean_list = (
        np.bincount(
            ids_s1.astype(int), weights=cm_array_gaussian_s1.reshape(ids_s1.shape)
        )
        / count_s1
    )
    mean_list[unq_s1 == 0] = NAN_VAL  # force mean of unchanged objects to be -9999
    s1_info = pd.DataFrame({"label": unq_s1, "mean_intensity": mean_list})
    #######################################################################################
    #                                 Scale 2: change patch                              #
    #######################################################################################
    object_map_s2 = object_map_s1.copy()
    object_map_s2[object_map_s2 > 0] = 1
    object_map_s2 = sklabel(object_map_s2, connectivity=2, background=0)

    return object_map_s1, cm_date_array, object_map_s2, s1_info


def object_analysis(
    object_map_s1,
    object_map_s2,
    s1_info,
    classification_map=None,
    uniform_threshold=None,
    uniform_sizeslope=None,
    parameters=None,
):
    """_summary_

    Parameters
    ----------
    object_map_s1 : np.ndarray
        object map for superpixel
    object_map_s2 : np.ndarray
        _object map for patch level
    s1_info : dict
        a zipped list, the first element is id and the second element is average
    classification_map : np.ndarray, optional
        2-d array, the classification map which is one year prior to object_map_s1, by default None
    uniform_threshold : double, optional
        A uniform change probability for all land categories, only used for parameter testing, by default None
    uniform_sizeslope : _type_, optional
        A uniform size factor for all land categories, only used for parameter testing, by default None
    parameters : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    # if classification_map is not None:
    #     assert classification_map.shape == object_map_s1.shape

    [n_rows, n_cols] = object_map_s1.shape
    change_map = np.zeros((n_rows, n_cols)).astype(np.uint8)

    unq_s2, ids_s2, count_s2 = np.unique(
        object_map_s2, return_inverse=True, return_counts=True
    )
    # force ids_s2 to be one-dimensional array, for compatibility with numpy2.x
    # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    ids_s2 = ids_s2.reshape(-1)

    lut_dict_s2 = dict(zip(unq_s2, count_s2))
    size_map = np.vectorize(lut_dict_s2.get)(
        object_map_s2
    )  # we use size of scale 2 to represent object szie

    # def pixelmode(regionmask):
    #     return stats.mode(regionmask)
    class_labels = s1_info["label"].to_list()
    if classification_map is None:
        class_list = [255] * len(s1_info)
    else:
        class_list = modeby(
            classification_map.astype(np.int16).reshape(ids_s2.shape),
            object_map_s1.reshape(ids_s2.shape),
        )

        # mode_list = mode_list[1:]
        # mode_list = regionprops_table(object_map_s1, classification_map, extra_properties=(pixelmode))
    size_list = modeby(
        size_map.reshape(ids_s2.shape), object_map_s1.reshape(ids_s2.shape)
    )  # the mode of s1 objects in sizemap

    # mode_list = regionprops_table(object_map_s1, size_map, extra_properties=(pixelmode, ))
    # mode_list, median_list = mode_median_by(classification_map.reshape(ids_s2.shape),
    #                                         cm_array_gaussian_s1.reshape(ids_s2.shape),
    #                                         object_map_s1.reshape(ids_s2.shape))

    # Use the tags/IDs to perform ID based summation of array2 elems and
    # thus divide by the ID counts to get ID based average values
    # stats_lut = pd.DataFrame(np.column_stack((unq_s1, mean_list,
    #                                           mode_list, size_list)),
    #                          columns=['id', 'cm_average', 'mode', 'npixels'])

    # need to remove the first element represent 0
    stats_lut = s1_info.assign(mode=class_list, npixels=size_list)
    change_group = []
    if len(class_labels) > 0:  # if non-background change objects > 0
        for index in class_labels:
            stats_lut_row = stats_lut.loc[stats_lut["label"] == index]
            if _is_change_object(
                stats_lut_row=stats_lut_row,
                keyword="mean_intensity",
                classification_map=classification_map,
                uniform_threshold=uniform_threshold,
                uniform_sizeslope=uniform_sizeslope,
                parameters=parameters,
            ):
                change_group.append(index)
        change_map[np.isin(object_map_s1, change_group)] = 1
    return change_map


class ObjectAnalystHPC:
    def __init__(
        self,
        dataset_info: DatasetInfo,
        stack_path: str,
        result_path: str,
        starting_date: int,
        band_num: int = 7,
        cmmap_path: Optional[str] = None,
        obia_path: Optional[str] = None,
        obcold_recg_path: Optional[str] = None,
        thematic_path: Optional[str] = None,
    ):
        """
        Args:
            dataset_info: dataset_info
            stack_path: the path of stack dataset
            result_path: the path of COLD rec_cg result
            starting_date: the starting date of the dataset to be processed
            cmmap_path: the path of change magnitude and dates files, the default is 'result_path/cm_maps'
            obia_path: the path of OBIA results, the default is 'result_path/OBIAresults'
            obcold_recg_path: the path of obcold change records, the default is 'result_path/obcold'
            thematic_path: : the path of classification maps, None represents no thematic maps were used for procedure
        """
        self.dataset_info = dataset_info
        self.thematic_path = thematic_path
        if cmmap_path is None:  # default path
            self.cmmap_path = join(result_path, "cm_maps")
        else:
            self.cmmap_path = cmmap_path
        if obia_path is None:  # default path
            self.obia_path = join(result_path, "OBIAresults")
        else:
            self.obia_path = obia_path
        if obcold_recg_path is None:  # default path
            self.obcold_recg_path = join(result_path, "obcold")
        else:
            self.obcold_recg_path = obcold_recg_path
        self.stack_path = stack_path
        self.starting_date = starting_date
        self.year_lowbound = pd.Timestamp.fromordinal(starting_date).year
        self.band_num = band_num

    @staticmethod
    def _check_inputs(
        dataset_info,
        stack_path,
        result_path,
        cmmap_path,
        obia_path,
        obcold_recg_path,
        thematic_path,
    ):
        if (isinstance(dataset_info.n_rows, int) is False) or (dataset_info.n_rows < 0):
            raise ValueError("n_rows must be positive integer")
        if (isinstance(dataset_info.n_cols, int) is False) or (dataset_info.n_cols < 0):
            raise ValueError("n_cols must be positive integer")
        if (isinstance(dataset_info.n_block_x, int) is False) or (
            dataset_info.n_block_x < 0
        ):
            raise ValueError("n_block_x must be positive integer")
        if (isinstance(dataset_info.n_block_y, int) is False) or (
            dataset_info.n_block_y < 0
        ):
            raise ValueError("n_block_y must be positive integer")
        if (isinstance(dataset_info.n_block_y, int) is False) or (
            dataset_info.n_block_y < 0
        ):
            raise ValueError("n_block_y must be positive integer")

        if os.path.isdir(stack_path) is False:
            raise FileExistsError("No such directory: {}".format(stack_path))

        if os.path.isdir(result_path) is False:
            raise FileExistsError("No such directory: {}".format(result_path))

    def hpc_preparation(self):
        """
        prepare folders for saving intermediate files
        """
        if exists(self.cmmap_path) is False:
            try:
                os.mkdir(self.cmmap_path)
            except IOError as e:
                raise e

        if exists(self.obia_path) is False:
            try:
                os.mkdir(self.obia_path)
            except IOError as e:
                raise e

        if exists(self.obcold_recg_path) is False:
            try:
                os.mkdir(self.obcold_recg_path)
            except IOError as e:
                raise e

    def get_lastyear_cmap_fromdate(self, date):
        if (
            pd.Timestamp.fromordinal(date).year - 1 < self.year_lowbound
        ):  # we used the the year before the date
            classified_year = self.year_lowbound
        else:
            classified_year = pd.Timestamp.fromordinal(date).year - 1
        try:
            classification_map = np.load(
                join(
                    self.thematic_path,
                    "yearlyclassification_{}.npy".format(classified_year),
                )
            )
        except IOError as e:
            raise e
        return classification_map

    def obia_execute(
        self,
        date: int,
        floodfill_ratio: Optional[float] = None,
        uniform_threshold: Optional[float] = None,
        uniform_sizeslope: Optional[float] = None,
        method: str = "floodfill",
        cm_output_interval: int = 60,
    ):
        """
        a function for executing OBIA pipeline
        Args:
            date: the ordinal date as an index of the temporal snapshot to be processed
            floodfill_ratio: the ratio of change magnitude relative to the seed pixel value as the fixed range for connecting
                             neighborhood in the floodfill algorithm
                             (https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaf1f55a048f8a45bc3383586e80b1f0d0)
            uniform_threshold: double, if not none, using a uniform change probability for all land categories, only used
                              for parameter testing
            uniform_sizeslope: double, if not none, using a uniform size factor for all land categories, only used
                              for parameter testing
            method: 'floodfill', 'slic', 'watershed'

        Returns:

        """
        if date - cm_output_interval < self.starting_date:
            change_map = np.full(
                (self.dataset_info.n_rows, self.dataset_info.n_cols), 0, dtype=np.byte
            )
            cm_date_array_updated = np.full(
                (self.dataset_info.n_rows, self.dataset_info.n_cols), 0, dtype=np.int32
            )
        else:
            date_array = np.load(
                join(self.cmmap_path, _cmdatename_fromdate(date) + ".npy")
            ).astype(np.int32)
            date_array_copy = date_array
            date_array = date_array + defaults["COMMON"]["JULIAN_LANDSAT4_LAUNCH"]
            date_array[date_array_copy == -9999] = -9999
            del date_array_copy
            date_array_last = np.load(
                join(
                    self.cmmap_path,
                    _cmdatename_fromdate(date - cm_output_interval) + ".npy",
                )
            ).astype(np.int32)
            date_array_copy = date_array_last
            date_array_last = (
                date_array_last + defaults["COMMON"]["JULIAN_LANDSAT4_LAUNCH"]
            )
            date_array_last[date_array_copy == -9999] = -9999
            del date_array_copy
            if method == "floodfill":
                [
                    object_map_s1,
                    cm_date_array_updated,
                    object_map_s2,
                    s1_info,
                ] = segmentation_floodfill(
                    np.load(join(self.cmmap_path, _cmname_fromdate(date) + ".npy")),
                    date_array,
                    np.load(
                        join(
                            self.cmmap_path,
                            _cmname_fromdate(date - cm_output_interval) + ".npy",
                        )
                    ),
                    date_array_last,
                )
            elif method == "slic":
                [object_map_s1, cm_date_array_updated, object_map_s2, s1_info] = (
                    segmentation_slic(
                        np.load(join(self.cmmap_path, _cmname_fromdate(date) + ".npy")),
                        date_array,
                        np.load(
                            join(
                                self.cmmap_path,
                                _cmname_fromdate(date - cm_output_interval) + ".npy",
                            )
                        ),
                        date_array_last,
                    )
                )
            elif method == "watershed":
                [
                    object_map_s1,
                    cm_date_array_updated,
                    object_map_s2,
                    s1_info,
                ] = segmentation_watershed(
                    np.load(join(self.cmmap_path, _cmname_fromdate(date) + ".npy")),
                    date_array,
                    np.load(
                        join(
                            self.cmmap_path,
                            _cmname_fromdate(date - cm_output_interval) + ".npy",
                        )
                    ),
                    date_array_last,
                )

            if self.thematic_path is not None:
                classification_map = self.get_lastyear_cmap_fromdate(date)
                change_map = object_analysis(
                    object_map_s1,
                    object_map_s2,
                    s1_info,
                    classification_map=classification_map,
                    uniform_threshold=uniform_threshold,
                    uniform_sizeslope=uniform_sizeslope,
                )
            else:
                change_map = object_analysis(
                    object_map_s1,
                    object_map_s2,
                    s1_info,
                    uniform_threshold=uniform_threshold,
                    uniform_sizeslope=uniform_sizeslope,
                )
        self.save_obiaresult(date, change_map, cm_date_array_updated)

    def save_obiaresult(self, date, change_map, cm_date_array_updated):
        """
        save OBIA array as a list of block-based subarrays which are either 0 or new break dates
        Args:
            date: int, the date as an index of the temporal snapshot to be processed
            change_map: array, the resultant change map to be saved out
            cm_date_array_updated: the change date map which was updated by filling NA values from the previous cm date maps
        Returns:

        """
        filenm = _obiaresname_fromdate(date)
        bytesize = 4

        # prevent -9999 * date
        cm_date_array_updated[cm_date_array_updated < 0] = 0
        # need yo load date snapshot
        # cm_date_snapshot = np.load(join(self.cmmap_path, _cmdatename_fromdate(date)+'.npy'))
        result_blocks = np.lib.stride_tricks.as_strided(
            np.multiply(change_map, cm_date_array_updated).astype(np.int32),
            shape=(
                self.dataset_info.n_block_y,
                self.dataset_info.n_block_x,
                self.dataset_info.block_height,
                self.dataset_info.block_width,
            ),
            strides=(
                self.dataset_info.n_cols * self.dataset_info.block_height * bytesize,
                self.dataset_info.block_width * bytesize,
                self.dataset_info.n_cols * bytesize,
                bytesize,
            ),
        )
        for i in range(self.dataset_info.n_block_y):
            for j in range(self.dataset_info.n_block_x):
                np.save(
                    join(self.obia_path, filenm + "_x{}_y{}.npy".format(j + 1, i + 1)),
                    result_blocks[i][j],
                )

        with open(join(self.obia_path, "tmp_OBIA_{}_finished.txt".format(date)), "w"):
            pass

    def is_finished_object_analysis(self, date_list):
        """
        Args:
            date_list: a list of dates for the whole period of the dataset

        Returns:
            True -> finished; False -> not finished
        """
        for date in date_list:
            if not exists(
                join(self.obia_path, "tmp_OBIA_{}_finished.txt".format(date))
            ):
                return False
        return True

    def get_allobiaresult_asarray(self, block_x, block_y):
        """
        Args:
            block_x: the x id of the block
            block_y: the y id of the block

        Returns:
            block-specific obia results which are formatted as 0 or break dates
        """
        obia_files = [
            f
            for f in os.listdir(self.obia_path)
            if f.startswith("obiaresult")
            and f.endswith("_x{}_y{}.npy".format(block_x, block_y))
        ]
        # sort image files by dates
        cm_dates = [
            int(
                f[
                    f.find("obiaresult_")
                    + len("obiaresult_") : f.find("obiaresult_")
                    + len("obiaresult_")
                    + 6
                ]
            )
            for f in obia_files
        ]
        files_date_zip = sorted(zip(cm_dates, obia_files))

        obia_tstack = [
            np.load(join(self.obia_path, f[1])).reshape(
                self.dataset_info.block_width * self.dataset_info.block_height
            )
            for count, f in enumerate(files_date_zip)
        ]
        return np.vstack(obia_tstack)

    def reconstruct_reccg(
        self,
        block_id: int,
        img_stack: Optional[np.ndarray] = None,
        img_dates_sorted: Optional[np.ndarray] = None,
        logger: Optional[Logger] = None,
        conse: int = 6,
    ) -> Union[List[Any], None]:
        """
        the third step of OBCOLD, it reconstructs the new temporal segment based on the new spatially adjusted break
        Args:
            block_id: the block id to be processed
            img_stack: time series block-based dataset
            img_dates_sorted: the new break dates from obiaresult_{}.npy
            logger: logger handle
        Returns:
            change records
        """
        if logger is None:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s |%(levelname)s| %(funcName)-15s| %(message)s",
                stream=sys.stdout,
            )
            logger = logging.getLogger(__name__)
        else:
            logger = logger

        block_y = get_block_y(block_id, self.dataset_info.n_block_x)
        block_x = get_block_x(block_id, self.dataset_info.n_block_x)
        if img_stack is None and img_dates_sorted is None:
            block_folder = join(
                self.stack_path, "block_x{}_y{}".format(block_x, block_y)
            )
            img_stack, img_dates_sorted = _read_blockdata(
                block_folder,
                self.dataset_info.n_cols * self.dataset_info.n_rows,
                self.band_num + 1,
            )
        obia_breaks = self.get_allobiaresult_asarray(block_x, block_y)
        result_collect = []
        if img_stack is None or img_dates_sorted is None:
            return None
        for pos in range(
            self.dataset_info.block_width * self.dataset_info.block_height
        ):
            original_row, original_col = get_rowcol_intile(
                pos,
                self.dataset_info.block_width,
                self.dataset_info.block_height,
                block_x,
                block_y,
            )

            try:
                obcold_result = obcold_reconstruct(
                    img_dates_sorted,
                    img_stack[pos, 0, :].astype(np.int64),
                    img_stack[pos, 1, :].astype(np.int64),
                    img_stack[pos, 2, :].astype(np.int64),
                    img_stack[pos, 3, :].astype(np.int64),
                    img_stack[pos, 4, :].astype(np.int64),
                    img_stack[pos, 5, :].astype(np.int64),
                    img_stack[pos, 6, :].astype(np.int64),
                    img_stack[pos, 7, :].astype(np.int64),
                    obia_breaks[:, pos].astype(np.int64),
                    conse=conse,
                    pos=self.dataset_info.n_cols * (original_row - 1) + original_col,
                )
            except RuntimeError:
                logger.error(
                    "COLD fails at original_row {}, original_col {}".format(
                        original_row, original_col
                    )
                )
            except Exception:  # pass exception
                pass
            else:
                result_collect.append(obcold_result)
        return result_collect

    def save_obcoldrecords(self, block_id, result_collect):
        block_x = get_block_x(block_id, self.dataset_info.n_block_x)
        block_y = get_block_y(block_id, self.dataset_info.n_block_x)
        np.save(
            join(
                self.obcold_recg_path,
                "record_change_x{}_y{}_obcold.npy".format(block_x, block_y),
            ),
            np.hstack(result_collect),
        )
