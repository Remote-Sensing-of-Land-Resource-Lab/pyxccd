# Author: Su Ye
# generating yearly, recent and first-disturbance maps from change records
# python export_change_map.py --source_dir=/data/landsat_c2/204_22 --result_path=/data/results/204_22_ccdc --yaml_path=/home/colory666/pyxccd_imagetool/config.yaml --n_cores=20 --out_path=/data/results/204_22_ccdc
import os
from os.path import join, isfile, join
import numpy as np
import pandas as pd
import tarfile

# from osgeo import gdal
import click

# from mpi4py import MPI
import multiprocessing
import functools

# from osgeo import gdal_array
import pickle
import rasterio
import yaml
import datetime as datetime
from pyxccd.utils import (
    class_from_dict,
    rio_loaddata,
    extract_features,
    getcategory_cold,
    getcategory_sccd,
)
from pyxccd.common import DatasetInfo, SccdOutput, sccd_dt, nrtqueue_dt, nrtmodel_dt


PACK_ITEM = 6
coef_names = ["a0", "c1", "a1", "b1", "a2", "b2", "a3", "b3", "cv", "rmse"]
band_names = [0, 1, 2, 3, 4, 5, 6]
SLOPE_SCALE = 10000

# copy from /pyxccd/src/python/pyxccd/pyclassifier.py because MPI has conflicts with the pyxccd package in UCONN HPC.
# Dirty approach!


def index_sccdpack(sccd_pack_single):
    """
    convert list of sccdpack to namedtuple to facilitate parse,
    :param sccd_pack_single: a nested list
    :return: a namedtuple SccdOutput
    """
    if len(sccd_pack_single) != PACK_ITEM:
        raise Exception(f"the element number of sccd_pack_single must be {PACK_ITEM}")

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


# def getcategory_obcold(cold_plot, i_curve, last_dist_type):
#     t_c = -250
#     if (
#         cold_plot[i_curve]["magnitude"][3] > t_c
#         and cold_plot[i_curve]["magnitude"][2] < -t_c
#         and cold_plot[i_curve]["magnitude"][4] < -t_c
#     ):
#         if (
#             cold_plot[i_curve + 1]["coefs"][3, 1]
#             > np.abs(cold_plot[i_curve]["coefs"][3, 1])
#             and cold_plot[i_curve + 1]["coefs"][2, 1]
#             < -np.abs(cold_plot[i_curve]["coefs"][2, 1])
#             and cold_plot[i_curve + 1]["coefs"][4, 1]
#             < -np.abs(cold_plot[i_curve]["coefs"][4, 1])
#         ):
#             return 3  # aforestation
#         else:
#             return 2  # regrowth
#     else:
#         if i_curve > 0:
#             if (
#                 cold_plot[i_curve]["t_break"] - cold_plot[i_curve - 1]["t_break"]
#                 > 365.25 * 5
#             ) or (last_dist_type != 1):
#                 return 1
#             flip_count = 0
#             for b in range(5):
#                 if (
#                     cold_plot[i_curve]["magnitude"][b + 1]
#                     * cold_plot[i_curve - 1]["magnitude"][b + 1]
#                     < 0
#                 ):
#                     flip_count = flip_count + 1
#             if flip_count >= 4:
#                 return 4
#             else:
#                 return 1
#         else:
#             return 1  # land disturbance


def _export_map_processing(
    dataset_info,
    method,
    year_uppbound,
    year_lowbound,
    coefs,
    coefs_bands,
    result_path,
    out_path,
    iblock,
):
    if method == "SCCDOFFLINE":
        dt = np.dtype(
            [
                ("t_start", np.int32),
                ("t_break", np.int32),
                ("num_obs", np.int32),
                # note that the slope coefficient was scaled up by 10000
                ("coefs", np.float32, (6, 6)),
                ("rmse", np.float32, 6),
                ("magnitude", np.float32, 6),
            ]
        )
    else:
        dt = np.dtype(
            [
                ("t_start", np.int32),
                ("t_end", np.int32),
                ("t_break", np.int32),
                ("pos", np.int32),
                ("num_obs", np.int32),
                ("category", np.short),
                ("change_prob", np.short),
                # note that the slope coefficient was scaled up by 10000
                ("coefs", np.float32, (7, 8)),
                ("rmse", np.float32, 7),
                ("magnitude", np.float32, 7),
            ]
        )
    if iblock >= dataset_info.nblocks:
        return

    current_block_y = int(iblock / dataset_info.n_block_x) + 1
    current_block_x = iblock % dataset_info.n_block_x + 1
    if method == "OBCOLD":
        filename = f"record_change_x{current_block_x}_y{current_block_y}_obcold.npy"
    elif method == "COLD":
        filename = f"record_change_x{current_block_x}_y{current_block_y}_cold.npy"
    else:
        filename = f"record_change_x{current_block_x}_y{current_block_y}_sccd.npy"

    results_block = [
        np.full(
            (dataset_info.block_height, dataset_info.block_width), -9999, dtype=np.int16
        )
        for t in range(year_uppbound - year_lowbound + 1)
    ]
    if coefs is not None:
        results_block_coefs = np.full(
            (
                dataset_info.block_height,
                dataset_info.block_width,
                len(coefs) * len(coefs_bands),
                year_uppbound - year_lowbound + 1,
            ),
            -9999,
            dtype=np.float32,
        )

    print(f"Processing the rec_cg file {join(result_path, filename)}")
    if not os.path.exists(join(result_path, filename)):
        print(f"the rec_cg file {join(result_path, filename)} is missing")
        for year in range(year_lowbound, year_uppbound + 1):
            outfile = join(out_path, f"tmp_map_block{iblock + 1}_{year}.npy")
            np.save(outfile, results_block[year - year_lowbound])
            # save into the coef files
            if coefs is not None:
                outfile = join(out_path, f"tmp_coefmap_block{iblock + 1}_{year}.npy")
                np.save(outfile, results_block_coefs[:, :, :, year - year_lowbound])
        return
    if method == "SCCDOFFLINE":
        file = open(join(result_path, filename), "rb")
        cold_block = []
        while True:
            try:
                cold_block.append(index_sccdpack(pickle.load(file)))
            except EOFError:
                break
        file.close()
    else:
        cold_block = np.array(np.load(join(result_path, filename)), dtype=dt)
        # cold_block = [np.array(element, dtype=dt) for element in cold_block]
        if len(cold_block) == 0:
            print(f"the rec_cg file {join(result_path, filename)} is missing")
            for year in range(year_lowbound, year_uppbound + 1):
                outfile = join(out_path, f"tmp_map_block{iblock + 1}_{year}.npy")
                np.save(outfile, results_block[year - year_lowbound])

    if method == "SCCDOFFLINE":
        for count, plot in enumerate(cold_block):
            for i_count, curve in enumerate(plot.rec_cg):
                if curve["t_break"] == 0 or count == (
                    len(cold_block) - 1
                ):  # last segment
                    continue

                i_col = (
                    int((plot.position - 1) % dataset_info.n_cols)
                    - (current_block_x - 1) * dataset_info.block_width
                )
                i_row = (
                    int((plot.position - 1) / dataset_info.n_cols)
                    - (current_block_y - 1) * dataset_info.block_height
                )
                if i_col < 0:
                    print(
                        f"Processing {filename} failed: i_row={i_row}; i_col={i_col} for {filename}"
                    )
                    # return
                current_dist_type = getcategory_sccd(plot.rec_cg, i_count)
                break_year = pd.Timestamp.fromordinal(curve["t_break"]).year
                if break_year < year_lowbound or break_year > year_uppbound:
                    continue
                results_block[break_year - year_lowbound][i_row][i_col] = (
                    current_dist_type * 1000
                    + curve["t_break"]
                    - (
                        pd.Timestamp.toordinal(
                            datetime.datetime(break_year, 1, 1, 0, 0)
                        )
                    )
                    + 1
                )
    else:
        cold_block.sort(order="pos")
        current_processing_pos = cold_block[0]["pos"]
        current_dist_type = 0
        year_list_to_predict = list(range(year_lowbound, year_uppbound + 1))
        ordinal_day_list = [
            pd.Timestamp.toordinal(datetime.datetime(year, 7, 1, 0, 0))
            for year in year_list_to_predict
        ]
        for count, curve in enumerate(cold_block):
            if curve["pos"] != current_processing_pos:
                current_processing_pos = curve["pos"]
                current_dist_type = 0

            if (
                curve["change_prob"] < 100
                or curve["t_break"] == 0
                or count == (len(cold_block) - 1)
            ):  # last segment
                continue

            i_col = (
                int((curve["pos"] - 1) % dataset_info.n_cols)
                - (current_block_x - 1) * dataset_info.block_width
            )
            i_row = (
                int((curve["pos"] - 1) / dataset_info.n_cols)
                - (current_block_y - 1) * dataset_info.block_height
            )
            if i_col < 0:
                print(
                    f"Processing {filename} failed: i_row={i_row}; i_col={i_col} for {join(result_path, filename)}"
                )
                return

            # if method == "OBCOLD":
            #     current_dist_type = getcategory_obcold(
            #         cold_block, count, current_dist_type
            #     )
            # else:
            current_dist_type = getcategory_cold(cold_block, count)
            break_year = pd.Timestamp.fromordinal(curve["t_break"]).year
            if break_year < year_lowbound or break_year > year_uppbound:
                continue
            results_block[break_year - year_lowbound][i_row][i_col] = (
                current_dist_type * 1000
                + curve["t_break"]
                - (pd.Timestamp.toordinal(datetime.datetime(break_year, 1, 1, 0, 0)))
                + 1
            )
            # e.g., 1315 means that disturbance happens at doy of 315

        if coefs is not None:
            cold_block_split = np.split(
                cold_block, np.argwhere(np.diff(cold_block["pos"]) != 0)[:, 0] + 1
            )
            for element in cold_block_split:
                # the relative column number in the block
                i_col = (
                    int((element[0]["pos"] - 1) % dataset_info.n_cols)
                    - (current_block_x - 1) * dataset_info.block_width
                )
                i_row = (
                    int((element[0]["pos"] - 1) / dataset_info.n_cols)
                    - (current_block_y - 1) * dataset_info.block_height
                )

                for band_idx, band in enumerate(coefs_bands):
                    feature_row = extract_features(
                        element, band, ordinal_day_list, -9999, feature_outputs=coefs
                    )
                    for index, coef in enumerate(coefs):
                        results_block_coefs[i_row][i_col][
                            index + band_idx * len(coefs)
                        ][:] = feature_row[index]

        # save the temp dataset out
        for year in range(year_lowbound, year_uppbound + 1):
            outfile = join(out_path, f"tmp_map_block{iblock + 1}_{year}.npy")
            np.save(outfile, results_block[year - year_lowbound])
            if coefs is not None:
                outfile = join(out_path, f"tmp_coefmap_block{iblock + 1}_{year}.npy")
                np.save(outfile, results_block_coefs[:, :, :, year - year_lowbound])


@click.command()
@click.option(
    "--source_dir",
    type=str,
    default=None,
    help="the folder directory of Landsat tar files " "downloaded from USGS website",
)
@click.option("--n_cores", type=int, default=1, help="the total cores assigned")
@click.option("--result_path", type=str, help="rec_cg folder")
@click.option("--out_path", type=str, help="output folder for saving image")
@click.option(
    "--method",
    type=click.Choice(["COLD", "OBCOLD", "SCCDOFFLINE"]),
    default="COLD",
    help="the algorithm used for processing",
)
@click.option("--yaml_path", type=str, help="path for yaml file")
@click.option(
    "--year_lowbound", type=int, default=1982, help="the starting year for exporting"
)
@click.option(
    "--year_uppbound", type=int, default=2020, help="the ending year for exporting"
)
@click.option("--coefs", type=str, default=None, help="if output coefs layers")
@click.option(
    "--coefs_bands",
    type=str,
    default="0, 1, 2, 3, 4, 5, 6",
    help="indicate the ba_nds for output coefs_bands,"
    "only works when coefs is True; note that the band "
    "order is b,g,r,n,s1,s2,t",
)
@click.option(
    "--collection",
    type=click.Choice(["ARD", "C2", "HLS", "HLS14", "ARD-C2"]),
    default="C2",
    help="image source category",
)
def main(
    source_dir,
    n_cores,
    result_path,  # record of change.npy
    out_path,
    method,
    year_lowbound,
    year_uppbound,
    yaml_path,
    coefs,
    coefs_bands,
    collection,
):
    # ref_path = '/Users/coloury/Dropbox/UCONN/spatial/test_results/h016v010/recentdist_mapCOLD.tif'
    # method = 'SCCDOFFLINE'
    # yaml_path = '/home/coloury/Dropbox/Documents/PyCharmProjects/HLS_NRT/config_hls.yaml'
    # result_path ='/home/coloury'
    # out_path = '/home/coloury'
    # year_lowbound = 1982
    # year_uppbound = 2020

    if method == "OBCOLD":
        result_path = join(result_path, "obcold")
        out_path = join(out_path, "obcold_maps")
    elif method == "COLD":
        out_path = join(out_path, "cold_maps")
    elif method == "SCCDOFFLINE":
        out_path = join(out_path, "sccd_maps")

    if coefs is not None:
        try:
            coefs = list(coefs.split(","))
            coefs = [str(coef) for coef in coefs]
        except ValueError:
            print(
                "Illegal coefs inputs: example, --coefs='a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse'"
            )

        try:
            coefs_bands = list(coefs_bands.split(","))
            coefs_bands = [int(coefs_band) for coefs_band in coefs_bands]
        except ValueError:
            print(
                "Illegal coefs_bands inputs: example, --coefs_bands='0, 1, 2, 3, 4, 5, 6'"
            )

    # outname'obcold':
    # outname = 'breakyear_cold_h11v9_{}_{}_{}'.format(lower_year, upper_year, method)
    if method == "SCCDOFFLINE":
        dt = np.dtype(
            [
                ("t_start", np.int32),
                ("t_break", np.int32),
                ("num_obs", np.int32),
                # note that the slope coefficient was scaled up by 10000
                ("coefs", np.float32, (6, 6)),
                ("rmse", np.float32, 6),
                ("magnitude", np.float32, 6),
            ]
        )
    else:
        dt = np.dtype(
            [
                ("t_start", np.int32),
                ("t_end", np.int32),
                ("t_break", np.int32),
                ("pos", np.int32),
                ("num_obs", np.int32),
                ("category", np.short),
                ("change_prob", np.short),
                # note that the slope coefficient was scaled up by 10000
                ("coefs", np.float32, (7, 8)),
                ("rmse", np.float32, 7),
                ("magnitude", np.float32, 7),
            ]
        )

    if coefs is not None:
        assert all(elem in coef_names for elem in coefs)
        assert all(elem in band_names for elem in coefs_bands)

    with open(yaml_path, "r") as yaml_obj:
        config = yaml.safe_load(yaml_obj)
    dataset_info = class_from_dict(DatasetInfo, config["DATASETINFO"])

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if collection == "C2":
        folder_list = [
            f[0 : len(f) - 4]
            for f in os.listdir(source_dir)
            if (isfile(join(source_dir, f)) and f.endswith(".tar"))
        ]
        if not os.path.exists(join(source_dir, "ref_folder")):
            with tarfile.open(join(source_dir, folder_list[0] + ".tar")) as tar_ref:
                tar_ref.extractall(join(source_dir, "ref_folder"))
        ref_path = join(source_dir, "ref_folder", f"{folder_list[0]}_SR_B1.TIF")
        ref_image = rio_loaddata(ref_path)
        with rasterio.open(ref_path) as src_dataset:
            profile = src_dataset.profile
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
        dataset_info.nblocks = dataset_info.n_block_x * dataset_info.n_block_y
        ref_image = None

    block_list = list(range(0, dataset_info.nblocks))
    pool = multiprocessing.Pool(n_cores)
    partial_func = functools.partial(
        _export_map_processing,
        dataset_info,
        method,
        year_uppbound,
        year_lowbound,
        coefs,
        coefs_bands,
        result_path,
        out_path,
    )  # TODO
    pool.map(partial_func, block_list)
    pool.close()
    pool.join()

    # for i in range(dataset_info.nblocks):
    #     _export_map_processing(ranks_percore, n_cores, dataset_info, method, year_uppbound, year_lowbound, coefs, coefs_bands, result_path, out_path, i)

    # assemble
    for year in range(year_lowbound, year_uppbound + 1):
        tmp_map_blocks = [
            np.load(join(out_path, f"tmp_map_block{x + 1}_{year}.npy"))
            for x in range(dataset_info.nblocks)
        ]

        results = np.hstack(tmp_map_blocks)
        results = np.vstack(np.hsplit(results, dataset_info.n_block_x))

        # for x in range(dataset_info.nblocks):
        #     os.remove(join(out_path, "tmp_map_block{}_{}.npy".format(x + 1, year)))
        profile.update(dtype=rasterio.int16)
        with rasterio.open(
            join(out_path, f"{year}_break_map_{method}.tif"), "w", **profile
        ) as dst:
            dst.write(results[0 : profile["height"], 0 : profile["width"]], 1)

    if coefs is not None:
        for year in range(year_lowbound, year_uppbound + 1):
            tmp_map_blocks = [
                np.load(join(out_path, f"tmp_coefmap_block{x + 1}_{year}.npy"))
                for x in range(dataset_info.nblocks)
            ]

            results = np.hstack(tmp_map_blocks)
            results = np.vstack(np.hsplit(results, dataset_info.n_block_x))
            ninput = 0
            for band_idx, band_name in enumerate(coefs_bands):
                for coef_index, coef in enumerate(coefs):
                    profile.update(dtype=rasterio.float32)
                    with rasterio.open(
                        join(
                            out_path,
                            f"{year}_coefs_{method}_{band_name}_{coef}.tif",
                        ),
                        "w",
                        **profile,
                    ) as dst:
                        dst.write(
                            results[
                                0 : profile["height"], 0 : profile["width"], ninput
                            ],
                            1,
                        )
                    ninput = ninput + 1

            # for x in range(dataset_info.nblocks):
            #     os.remove(join(out_path, f"tmp_coefmap_block{x + 1}_{year}.npy"))

    # output recent disturbance year
    recent_dist = np.full((profile["height"], profile["width"]), 0, dtype=np.int16)
    for year in range(year_lowbound, year_uppbound + 1):
        breakmap = rio_loaddata(join(out_path, f"{year}_break_map_{method}.tif"))
        recent_dist[(breakmap / 1000).astype(np.byte) == 1] = year

    profile.update(dtype=rasterio.int16)
    with rasterio.open(
        join(
            out_path,
            f"recent_disturbance_map_{method}.tif",
        ),
        "w",
        **profile,
    ) as dst:
        dst.write(recent_dist, 1)

    first_dist = np.full((profile["height"], profile["width"]), 0, dtype=np.int16)
    for year in range(year_uppbound, year_lowbound - 1, -1):
        breakmap = rio_loaddata(join(out_path, f"{year}_break_map_{method}.tif"))
        first_dist[(breakmap / 1000).astype(np.byte) == 1] = year
    with rasterio.open(
        join(
            out_path,
            f"first_disturbance_map_{method}.tif",
        ),
        "w",
        **profile,
    ) as dst:
        dst.write(first_dist, 1)


if __name__ == "__main__":
    main()
