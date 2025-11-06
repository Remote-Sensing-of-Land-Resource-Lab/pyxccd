# Author: Su Ye
# Modified for HLS data input
# generating yearly, recent and first-disturbance maps from change records
import os
from os.path import join
import numpy as np
import pandas as pd
import click
import multiprocessing
import functools
import pickle
import rasterio
import yaml
import datetime as datetime
from pyxccd.utils import (
    class_from_dict,
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


def find_hls_tif(hls_dir):
    """
    Find the first TIFF file in HLS directory to get spatial reference
    """
    for root, dirs, files in os.walk(hls_dir):
        for file in files:
            if file.endswith(".tif"):
                return join(root, file)
    raise FileNotFoundError(f"No TIFF file found in {hls_dir}")


def index_sccdpack(sccd_pack_single):
    """
    convert list of sccdpack to namedtuple to facilitate parse,
    :param sccd_pack_single: a nested list
    :return: a namedtuple SccdOutput
    """
    if len(sccd_pack_single) != PACK_ITEM:
        raise Exception(f"the element number of sccd_pack_single must be {PACK_ITEM}")

    sccd_pack_single = SccdOutput(*sccd_pack_single)

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
                ("coefs", np.float32, (7, 8)),
                ("rmse", np.float32, 7),
                ("magnitude", np.float32, 7),
            ]
        )

    if iblock >= dataset_info.nblocks:
        return

    current_block_y = int(iblock / dataset_info.n_block_x) + 1
    current_block_x = iblock % dataset_info.n_block_x + 1

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

    if method == "COLD":
        filename = f"record_change_x{current_block_x}_y{current_block_y}_cold.npy"
    else:
        filename = f"record_change_x{current_block_x}_y{current_block_y}_sccd.npy"

    # print(f"Processing the rec_cg file {join(result_path, filename)}")

    # Ensure that the output file is generated no matter what
    try:
        if not os.path.exists(join(result_path, filename)):
            print(
                f"[Warning] The input file {join(result_path, filename)} does not exist"
            )
            raise FileNotFoundError(
                f"Input file not found: {join(result_path, filename)}"
            )

        if method == "SCCDOFFLINE":
            cold_block = []
            try:
                with open(join(result_path, filename), "rb") as file:
                    while True:
                        try:
                            cold_block.append(index_sccdpack(pickle.load(file)))
                        except EOFError:
                            break
                        except Exception as e:
                            print(f"[Error] Failed to parse file {filename}: {str(e)}")
                            cold_block = []
                            break
            except Exception as e:
                print(f"[Error] Failed to read file {filename}: {str(e)}")
                cold_block = []
        else:
            try:
                cold_block = np.array(np.load(join(result_path, filename)), dtype=dt)
                if len(cold_block) == 0:
                    print(f"[Warning] File {filename} is empty")
            except Exception as e:
                print(f"[Error] Failed to load file {filename}: {str(e)}")
                cold_block = np.array([], dtype=dt)

        # Process data
        if method == "SCCDOFFLINE":
            for count, plot in enumerate(cold_block):
                for i_count, curve in enumerate(plot.rec_cg):
                    if curve["t_break"] == 0 or count == (len(cold_block) - 1):
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

                    current_dist_type = getcategory_sccd(plot.rec_cg, i_count)
                    break_year = pd.Timestamp.fromordinal(curve["t_break"]).year
                    if break_year < year_lowbound or break_year > year_uppbound:
                        continue
                    results_block[break_year - year_lowbound][i_row][i_col] = (
                        current_dist_type * 1000
                        + curve["t_break"]
                        - pd.Timestamp.toordinal(
                            datetime.datetime(break_year, 1, 1, 0, 0)
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
                ):
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

                current_dist_type = getcategory_cold(cold_block, count)
                break_year = pd.Timestamp.fromordinal(curve["t_break"]).year
                if break_year < year_lowbound or break_year > year_uppbound:
                    continue
                results_block[break_year - year_lowbound][i_row][i_col] = (
                    current_dist_type * 1000
                    + curve["t_break"]
                    - pd.Timestamp.toordinal(datetime.datetime(break_year, 1, 1, 0, 0))
                    + 1
                )

            if coefs is not None:
                cold_block_split = np.split(
                    cold_block, np.argwhere(np.diff(cold_block["pos"]) != 0)[:, 0] + 1
                )
                for element in cold_block_split:
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
                            element,
                            band,
                            ordinal_day_list,
                            -9999,
                            feature_outputs=coefs,
                        )
                        for index, coef in enumerate(coefs):
                            results_block_coefs[i_row][i_col][
                                index + band_idx * len(coefs)
                            ][:] = feature_row[index]

            for year in range(year_lowbound, year_uppbound + 1):
                outfile = join(out_path, f"tmp_map_block{iblock + 1}_{year}.npy")
                np.save(outfile, results_block[year - year_lowbound])
                if coefs is not None:
                    outfile = join(
                        out_path, f"tmp_coefmap_block{iblock + 1}_{year}.npy"
                    )
                    np.save(outfile, results_block_coefs[:, :, :, year - year_lowbound])
                    pass

    except Exception as e:
        print(
            f"[Critical Error] Exception occurred while processing block {iblock+1}: {str(e)}"
        )

    finally:
        # Ensure results are saved regardless of any issues
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
    help="The directory of Landsat tar files downloaded from USGS website",
)
@click.option("--n_cores", type=int, default=1, help="The total cores assigned")
@click.option(
    "--result_path",
    type=str,
    help="Path to change detection results (record_change_*.npy files)",
)
@click.option("--out_path", type=str, help="Output directory for saving maps")
@click.option(
    "--method",
    type=click.Choice(["COLD", "SCCDOFFLINE"]),
    default="COLD",
    help="Choose change detection algorithm used",
)
@click.option("--yaml_path", type=str, help="Path to YAML configuration file")
@click.option(
    "--year_lowbound", type=int, default=2019, help="Starting year for analysis"
)
@click.option(
    "--year_uppbound", type=int, default=2024, help="Ending year for analysis"
)
@click.option(
    "--coefs", type=str, default=None, help="Indicate whether to output coefs layers"
)
@click.option(
    "--coefs_bands",
    type=str,
    default="0,1,2,3,4,5,6",
    help="Indicate the bands for output coefs_bands,"
    "only works when coefs is True; note that the band "
    "order is b,g,r,n,s1,s2,t",
)
def main(
    source_dir,
    n_cores,
    result_path,
    out_path,
    method,
    year_lowbound,
    year_uppbound,
    yaml_path,
    coefs,
    coefs_bands,
):
    # Initialize output directories
    if method == "COLD":
        out_path = join(out_path, "cold_maps")
    elif method == "SCCDOFFLINE":
        out_path = join(out_path, "sccd_maps")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Parse coefficients and bands if provided
    if coefs is not None:
        try:
            coefs = [c.strip() for c in coefs.split(",")]
            assert all(elem in coef_names for elem in coefs)
        except:
            raise ValueError(
                "Illegal coefs inputs: example, --coefs='a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse'"
            )

        try:
            coefs_bands = [int(b.strip()) for b in coefs_bands.split(",")]
            assert all(elem in band_names for elem in coefs_bands)
        except:
            raise ValueError(
                "Illegal coefs_bands inputs: example, --coefs_bands='0, 1, 2, 3, 4, 5, 6'"
            )

    # Load configuration from YAML
    with open(yaml_path, "r") as yaml_obj:
        config = yaml.safe_load(yaml_obj)
    dataset_info = class_from_dict(DatasetInfo, config["DATASETINFO"])

    # Get spatial reference from HLS TIFF file
    ref_tif = find_hls_tif(source_dir)
    with rasterio.open(ref_tif) as src:
        profile = src.profile
        profile.update(dtype="int16", nodata=-9999, count=1)

        # Update dataset info based on actual HLS data dimensions
        dataset_info.n_rows = src.height
        dataset_info.n_cols = src.width

        # Adjust block parameters if needed
        if src.height % dataset_info.block_height > 0:
            dataset_info.n_block_y = (src.height // dataset_info.block_height) + 1
        if src.width % dataset_info.block_width > 0:
            dataset_info.n_block_x = (src.width // dataset_info.block_width) + 1

        dataset_info.nblocks = dataset_info.n_block_x * dataset_info.n_block_y

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        # Clean up any existing temporary files
        for f in os.listdir(out_path):
            if f.startswith("tmp_map_block") or f.startswith("tmp_coefmap_block"):
                try:
                    os.remove(join(out_path, f))
                except:
                    pass

    # Print debug information
    print(f"Input directory: {result_path}")
    print(f"Output directory: {out_path}")
    print(f"Number of blocks: {dataset_info.nblocks}")
    print(f"Year range: {year_lowbound}-{year_uppbound}")

    # Process blocks in parallel
    block_list = list(range(dataset_info.nblocks))
    with multiprocessing.Pool(n_cores) as pool:
        pool.map(
            functools.partial(
                _export_map_processing,
                dataset_info,
                method,
                year_uppbound,
                year_lowbound,
                coefs,
                coefs_bands,
                result_path,
                out_path,
            ),
            block_list,
        )

    for year in range(year_lowbound, year_uppbound + 1):
        missing = 0
        for x in range(dataset_info.nblocks):
            if not os.path.exists(join(out_path, f"tmp_map_block{x + 1}_{year}.npy")):
                missing += 1
                # Create empty block file
                empty_block = np.full(
                    (dataset_info.block_height, dataset_info.block_width),
                    -9999,
                    dtype=np.int16,
                )
                np.save(join(out_path, f"tmp_map_block{x + 1}_{year}.npy"), empty_block)

        if missing > 0:
            print(f"[Note] Created {missing} empty block files for year {year}")

    # Assemble final maps
    for year in range(year_lowbound, year_uppbound + 1):
        # Assemble annual change map
        tmp_map_blocks = [
            np.load(join(out_path, f"tmp_map_block{x + 1}_{year}.npy"))
            for x in range(dataset_info.nblocks)
        ]
        results = np.vstack(
            [
                np.hstack(
                    [
                        tmp_map_blocks[y * dataset_info.n_block_x + x]
                        for x in range(dataset_info.n_block_x)
                    ]
                )
                for y in range(dataset_info.n_block_y)
            ]
        )

        with rasterio.open(
            join(out_path, f"{year}_break_map_{method}.tif"), "w", **profile
        ) as dst:
            dst.write(results[: profile["height"], : profile["width"]], 1)

        # Clean up temporary files
        for x in range(dataset_info.nblocks):
            os.remove(join(out_path, f"tmp_map_block{x + 1}_{year}.npy"))

        # Process coefficient maps if requested
        if coefs is not None:
            tmp_coef_blocks = [
                np.load(join(out_path, f"tmp_coefmap_block{x + 1}_{year}.npy"))
                for x in range(dataset_info.nblocks)
            ]
            coef_results = np.vstack(
                [
                    np.hstack(
                        [
                            tmp_coef_blocks[y * dataset_info.n_block_x + x]
                            for x in range(dataset_info.n_block_x)
                        ]
                    )
                    for y in range(dataset_info.n_block_y)
                ]
            )

            ninput = 0
            for band_idx, band_name in enumerate(coefs_bands):
                for coef_index, coef in enumerate(coefs):
                    profile.update(dtype="float32")
                    with rasterio.open(
                        join(out_path, f"{year}_coefs_{method}_{band_name}_{coef}.tif"),
                        "w",
                        **profile,
                    ) as dst:
                        dst.write(
                            coef_results[
                                : profile["height"], : profile["width"], ninput
                            ],
                            1,
                        )
                    ninput += 1

            for x in range(dataset_info.nblocks):
                os.remove(join(out_path, f"tmp_coefmap_block{x + 1}_{year}.npy"))

    # Generate recent disturbance map
    recent_dist = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
    for year in range(year_uppbound, year_lowbound - 1, -1):
        with rasterio.open(join(out_path, f"{year}_break_map_{method}.tif")) as src:
            breakmap = src.read(1)
        recent_dist[(breakmap // 1000 == 1) & (recent_dist == 0)] = year

    profile.update(dtype="int16")
    with rasterio.open(
        join(out_path, f"recent_disturbance_map_{method}.tif"), "w", **profile
    ) as dst:
        dst.write(recent_dist, 1)

    # Generate first disturbance map
    first_dist = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
    for year in range(year_lowbound, year_uppbound + 1):
        with rasterio.open(join(out_path, f"{year}_break_map_{method}.tif")) as src:
            breakmap = src.read(1)
        first_dist[(breakmap // 1000 == 1) & (first_dist == 0)] = year

    with rasterio.open(
        join(out_path, f"first_disturbance_map_{method}.tif"), "w", **profile
    ) as dst:
        dst.write(first_dist, 1)


if __name__ == "__main__":
    main()
