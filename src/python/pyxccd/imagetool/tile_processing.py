# Author: Su Ye
# This script is an example for running pyscold in UCONN job array environment
# Due to the independence features of job array, we use writing disk files for communication
# Two types of logging files are used: 1) print() to save block-based  processing info into individual slurm file;
# 2) tile_processing_report.log records config for each tile
# 2) is only called when rank == 1
# python tile_processing.py --stack_path=/data/results/204_22_stack --result_path=/data/results/204_22_ccdc --yaml_path=/home/colory666/pyxccd_imagetool/config.yaml --n_cores=30 "--source_dir=/data/landsat_c2/204_22"
import yaml
import os
from os.path import join
import pandas as pd
import datetime as dt
import numpy as np
import pyxccd
from datetime import datetime
from pytz import timezone
import click
import time
import pickle
from dateutil.parser import parse
from pyxccd import cold_detect, sccd_detect

from pyxccd.utils import (
    get_rowcol_intile,
    unindex_sccdpack,
    class_from_dict,
)
import functools
import multiprocessing
from pyxccd.common import DatasetInfo

TZ = timezone("Asia/Shanghai")


def tileprocessing_report(
    result_log_path,
    stack_path,
    version,
    algorithm,
    dataset_info,
    startpoint,
    cold_timepoint,
    tz,
    n_cores,
    probability_threshold,
    conse,
):
    """
    output tile-based processing report
    Parameters
    ----------
    result_log_path: string
        outputted log path
    stack_path: string
        stack path
    version: string
    algorithm: string
    dataset_info: dictionary structure
    startpoint: a time point, when the program starts
    tz: string, time zone
    n_cores: the core number used
    probability_threshold: change probability threshold
    conse: number of consecutive observations
    """
    endpoint = datetime.now(TZ)
    file = open(result_log_path, "w")
    file.write("pyxccd V{} \n".format(version))
    file.write("Author: Su Ye(remoteseningsuy@gmail.com)\n")
    file.write("Algorithm: {} \n".format(algorithm))
    file.write("Starting_time: {}\n".format(startpoint.strftime("%Y-%m-%d %H:%M:%S")))
    file.write("Change probability threshold: {}\n".format(probability_threshold))
    file.write("Conse: {}\n".format(conse))
    file.write("stack_path: {}\n".format(stack_path))
    file.write("The number of requested cores: {}\n".format(n_cores))
    file.write(
        "The program starts at {}\n".format(startpoint.strftime("%Y-%m-%d %H:%M:%S"))
    )
    file.write(
        "The processing ends at {}\n".format(
            cold_timepoint.strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    file.write(
        "The program ends at {}\n".format(endpoint.strftime("%Y-%m-%d %H:%M:%S"))
    )
    file.write(
        "The program lasts for {:.2f}mins\n".format(
            (endpoint - startpoint) / dt.timedelta(minutes=1)
        )
    )
    file.close()


def is_finished_cold_blockfinished(result_path, nblocks):
    """
    check if the algorithm finishes all blocks
    Parameters
    ----------
    result_path: the path that save results
    nblocks: the block number

    Returns: boolean
    -------
        True -> all block finished
    """
    for n in range(nblocks):
        if not os.path.exists(
            os.path.join(result_path, "COLD_block{}_finished.txt".format(n + 1))
        ):
            return False
    return True


def get_stack_date(
    dataset_info: DatasetInfo,
    block_x: int,
    block_y: int,
    stack_path: str,
    low_datebound: int = 0,
    high_datebound: int = 999999,
    nband: int = 8,
):
    """
    :param dataset_info:
    :param block_x: block id at x axis
    :param block_y: block id at y axis
    :param stack_path: stack path
    :param low_datebound: ordinal data of low bounds of selection date range
    :param high_datebound: ordinal data of upper bounds of selection date range
    :return:
        img_tstack, img_dates_sorted
        img_tstack - 3-d array (dataset_info.block_width * dataset_info.block_height, nband, nimage)
    """
    dataset_info.block_width = int(
        dataset_info.n_cols / dataset_info.n_block_x
    )  # width of a block
    dataset_info.block_height = int(
        dataset_info.n_rows / dataset_info.n_block_y
    )  # height of a block
    block_folder = join(stack_path, "block_x{}_y{}".format(block_x, block_y))
    img_files = [
        f for f in os.listdir(block_folder) if f.startswith("L") or f.startswith("S")
    ]
    if len(img_files) == 0:
        return None, None
    # sort image files by dates
    img_dates = [
        pd.Timestamp.toordinal(
            dt.datetime(int(folder_name[9:13]), 1, 1)
            + dt.timedelta(int(folder_name[13:16]) - 1)
        )
        for folder_name in img_files
    ]
    img_dates_selected = [
        img_dates[i]
        for i, date in enumerate(img_dates)
        if img_dates[i] >= low_datebound and img_dates[i] <= high_datebound
    ]
    img_files_selected = [
        img_files[i]
        for i, date in enumerate(img_dates)
        if img_dates[i] >= low_datebound and img_dates[i] <= high_datebound
    ]

    files_date_zip = sorted(zip(img_dates_selected, img_files_selected))
    img_files_sorted = [x[1] for x in files_date_zip]
    img_dates_sorted = np.asarray([x[0] for x in files_date_zip])
    img_tstack = np.dstack(
        [
            np.load(join(block_folder, f)).reshape(
                dataset_info.block_width * dataset_info.block_height, nband
            )
            for f in img_files_sorted
        ]
    )
    return img_tstack, img_dates_sorted


def block_tile_processing(
    dataset_info,
    stack_path,
    result_path,
    method,
    probability_threshold,
    conse,
    b_c2,
    low_datebound,
    upper_datebound,
    block_id,
):
    if block_id > dataset_info.n_block_x * dataset_info.n_block_y:
        return

    # note that block_x and block_y start from 1
    block_y = int((block_id - 1) / dataset_info.n_block_x) + 1
    block_x = int((block_id - 1) % dataset_info.n_block_x) + 1

    finished_file = (
        "COLD_block{}_finished.txt".format(block_id)
        if method == "COLD"
        else "SCCD_block{}_finished.txt".format(block_id)
    )
    if os.path.exists(join(result_path, finished_file)):
        print(
            "Per-pixel {} processing is finished for block_x{}_y{} ({})".format(
                method, block_x, block_y, datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        return

    img_tstack, img_dates_sorted = get_stack_date(
        dataset_info, block_x, block_y, stack_path, low_datebound, upper_datebound
    )

    if img_tstack is None:  # empty block
        print(
            "Empty block block_x{}_y{} ({})".format(
                block_x, block_y, datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        with open(join(result_path, finished_file), "w"):
            pass
        return
    # for sccd, as the output is heterogeneous, we continuously save the sccd pack for each pixel
    if method == "SCCDOFFLINE":
        result_file = join(
            result_path,
            "record_change_x{}_y{}_sccd.npy".format(block_x, block_y),
        )
        # start looping every pixel in the block
        try:
            with open(result_file, "wb+") as f:
                for pos in range(dataset_info.block_width * dataset_info.block_height):
                    original_row, original_col = get_rowcol_intile(
                        pos,
                        dataset_info.block_width,
                        dataset_info.block_height,
                        block_x,
                        block_y,
                    )
                    try:
                        sccd_result = sccd_detect(
                            img_dates_sorted,
                            img_tstack[pos, 0, :].astype(np.int64),
                            img_tstack[pos, 1, :].astype(np.int64),
                            img_tstack[pos, 2, :].astype(np.int64),
                            img_tstack[pos, 3, :].astype(np.int64),
                            img_tstack[pos, 4, :].astype(np.int64),
                            img_tstack[pos, 5, :].astype(np.int64),
                            img_tstack[pos, 6, :].astype(np.int64),
                            p_cg=probability_threshold,
                            conse=conse,
                            b_c2=b_c2,
                            pos=dataset_info.n_cols * (original_row - 1) + original_col,
                        )
                    except RuntimeError:
                        print(
                            "S-CCD fails at original_row {}, original_col {} ({})".format(
                                original_row,
                                original_col,
                                datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                            )
                        )
                    else:
                        pickle.dump(unindex_sccdpack(sccd_result), f)

            with open(join(result_path, finished_file), "w"):
                pass

            print(
                "Per-pixel SCCD processing is finished for block_x{}_y{} ({})".format(
                    block_x, block_y, datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
                )
            )

        except Exception as e:
            print(
                "SCCD processing failed for block_x{}_y{}: {} ({})".format(
                    block_x,
                    block_y,
                    str(e),
                    datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            if os.path.exists(result_file):
                try:
                    os.remove(result_file)
                except:
                    pass

    else:
        result_collect = []
        result_file = join(
            result_path,
            "record_change_x{}_y{}_cold.npy".format(block_x, block_y),
        )

        try:
            for pos in range(dataset_info.block_width * dataset_info.block_height):
                original_row, original_col = get_rowcol_intile(
                    pos,
                    dataset_info.block_width,
                    dataset_info.block_height,
                    block_x,
                    block_y,
                )
                try:
                    cold_result = cold_detect(
                        img_dates_sorted,
                        img_tstack[pos, 0, :].astype(np.int64),
                        img_tstack[pos, 1, :].astype(np.int64),
                        img_tstack[pos, 2, :].astype(np.int64),
                        img_tstack[pos, 3, :].astype(np.int64),
                        img_tstack[pos, 4, :].astype(np.int64),
                        img_tstack[pos, 5, :].astype(np.int64),
                        img_tstack[pos, 6, :].astype(np.int64),
                        img_tstack[pos, 7, :].astype(np.int64),
                        p_cg=probability_threshold,
                        conse=conse,
                        b_c2=b_c2,
                        pos=dataset_info.n_cols * (original_row - 1) + original_col,
                    )
                except RuntimeError:
                    print(
                        "COLD fails at original_row {}, original_col {} ({})".format(
                            original_row,
                            original_col,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    )
                except Exception as e:
                    print(
                        "COLD error at original_row {}, original_col {}: {} ({})".format(
                            original_row,
                            original_col,
                            str(e),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    )
                else:
                    result_collect.append(cold_result)

            # Save COLD results
            if len(result_collect) > 0:
                np.save(result_file, np.hstack(result_collect))

            with open(join(result_path, finished_file), "w"):
                pass

            print(
                "Per-pixel COLD processing is finished for block_x{}_y{} ({})".format(
                    block_x, block_y, datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
                )
            )

        except Exception as e:
            print(
                "COLD processing failed for block_x{}_y{}: {} ({})".format(
                    block_x,
                    block_y,
                    str(e),
                    datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            if os.path.exists(result_file):
                try:
                    os.remove(result_file)
                except:
                    pass


@click.command()
@click.option("--rank", type=int, default=1, help="the rank id")
@click.option("--n_cores", type=int, default=1, help="the total cores assigned")
@click.option("--stack_path", type=str, help="the path for stack data")
@click.option("--result_path", type=str, help="the path for storing results")
@click.option("--yaml_path", type=str, help="YAML path")
@click.option(
    "--method",
    type=click.Choice(["COLD", "SCCDOFFLINE"]),
    default="COLD",
    help="COLD or SCCD-OFFLINE",
)
@click.option(
    "--low_datebound",
    type=str,
    default=None,
    help="low date bound of image selection for processing. Example - 2019-01-01",
)
@click.option(
    "--upper_datebound",
    type=str,
    default=None,
    help="upper date bound of image selection for processing. Example - 2024-12-31",
)
def main(
    rank,
    n_cores,
    stack_path,
    result_path,
    yaml_path,
    method,
    low_datebound,
    upper_datebound,
):
    start_time = datetime.now(TZ)

    if low_datebound is None:
        low_datebound = 0
    else:
        low_datebound = parse(low_datebound).toordinal()

    if upper_datebound is None:
        upper_datebound = 999999
    else:
        upper_datebound = parse(upper_datebound).toordinal()

    # Reading config
    with open(yaml_path, "r") as yaml_obj:
        config_general = yaml.safe_load(yaml_obj)
    dataset_info = class_from_dict(DatasetInfo, config_general["DATASETINFO"])

    conse = int(config_general["ALGORITHMINFO"]["conse"])
    probability_threshold = config_general["ALGORITHMINFO"]["probability_threshold"]

    if (dataset_info.n_cols % dataset_info.block_width != 0) or (
        dataset_info.n_rows % dataset_info.block_height != 0
    ):
        print(
            "n_cols, n_rows must be divisible respectively by dataset_info.block_width, dataset_info.block_height! Please check your config yaml"
        )
        exit()

    # logging and folder preparation
    if rank == 1:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(
            "The per-pixel time series processing begins: {}".format(
                start_time.strftime("%Y-%m-%d %H:%M:%S")
            )
        )

        if not os.path.exists(stack_path):
            print(
                "Failed to locate stack folders. The program ends: {}".format(
                    datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            return

    #########################################################################
    #                        per-pixel processing procedure                 #
    #########################################################################
    block_list = list(range(1, dataset_info.n_block_x * dataset_info.n_block_y + 1, 1))
    pool = multiprocessing.Pool(n_cores)

    partial_func = functools.partial(
        block_tile_processing,
        dataset_info,
        stack_path,
        result_path,
        method,
        probability_threshold,
        conse,
        True,
        low_datebound,
        upper_datebound,
    )

    pool.map(partial_func, block_list)
    pool.close()
    pool.join()

    # wait for all cores to be finished
    while not is_finished_cold_blockfinished(
        result_path, dataset_info.n_block_x * dataset_info.n_block_y
    ):
        time.sleep(30)

    if rank == 1:
        cold_timepoint = datetime.now(TZ)
        tileprocessing_report(
            join(result_path, "tile_processing_report.log"),
            stack_path,
            pyxccd.__version__,
            method,
            dataset_info,
            start_time,
            cold_timepoint,
            TZ,
            n_cores,
            probability_threshold,
            conse,
        )
        print(
            "The whole procedure finished: {}".format(
                datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
            )
        )


if __name__ == "__main__":
    main()
