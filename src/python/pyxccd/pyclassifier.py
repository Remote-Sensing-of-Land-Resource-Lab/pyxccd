import os
import datetime as dt
from os.path import join, exists
import joblib
import time
import logging
from logging import Logger
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# from osgeo import gdal_array
from typing import Optional
from pyxccd.app import defaults
from pyxccd.utils import (
    get_block_y,
    get_block_x,
    get_col_index,
    get_row_index,
    assemble_array,
    rio_loaddata,
    extract_features,
)
from pyxccd.common import DatasetInfo


def generate_sample_num(label: np.ndarray, sample_parameters: dict) -> np.ndarray:
    """generate sample number for each land cover category using the method from the paper 'Optimizing
    selection of training and auxiliary data for operational land cover classification for the LCMAP initiative'

    Parameters
    ----------
    label : np.ndarray
        a label map
    sample_parameters : dict
        a dictionary must include "total_landcover_category", "total_samples", "max_category_samples", and "min_category_samples"

    Returns
    -------
    np.ndarray
        1-d array that stores the sample number for each label
    """
    unique_category, unique_counts_category = np.unique(
        label[label <= sample_parameters["total_landcover_category"]],
        return_counts=True,
    )
    counts_category = np.array([0] * sample_parameters["total_landcover_category"])
    for x in range(len(unique_counts_category)):
        counts_category[int(unique_category[x] - 1)] = unique_counts_category[x]
    percate_samples = np.array(
        [
            round(x / sum(counts_category) * sample_parameters["total_samples"])
            for x in counts_category
        ]
    )
    percate_samples[percate_samples > sample_parameters["max_category_samples"]] = (
        sample_parameters["max_category_samples"]
    )
    percate_samples[percate_samples < sample_parameters["min_category_samples"]] = (
        sample_parameters["min_category_samples"]
    )
    # needs to check not exceed the total category pixels
    percate_samples = np.minimum(percate_samples, counts_category)
    return percate_samples


def get_features(path: str) -> np.ndarray:
    """get block-based features

    Parameters
    ----------
    path : str
        Path for feature output

    Returns
    -------
    np.ndarray
        2-d array for block-based features, (block_width*block_height, total feature)
    """

    return np.load(path)


class PyClassifier:
    def __init__(
        self,
        dataset_info: DatasetInfo,
        feature_outputs: list = ["a0", "a1", "b1"],
        logger: Optional[Logger] = None,
        band_num: int = 7,
    ):
        """_summary_

        Parameters
        ----------
        dataset_info : :py:type:`~pyxccd.common.DatasetInfo`
            time-series basic dataset info
        feature_outputs : list, optional
            Indicate which features to be outputted.  They must be within [a0, c1, a1, b1,a2, b2, a3,
            b3, rmse], by default ["a0", "a1", "b1"]
        logger : Optional[Logger], optional
            log handler, by default None
        band_num : int, optional
            the band number to be processed, by default 7
        """

        self.dataset_info = dataset_info
        # self.dataset_info.block_width = int(self.dataset_info.n_cols / self.dataset_info.n_block_x)
        # self.dataset_info.block_height = int(self.dataset_info.n_rows / self.dataset_info.n_block_y)
        # self.dataset_info.nblocks = self.dataset_info.n_block_x * self.dataset_info.n_block_y
        for feature in feature_outputs:
            assert feature in ["a0", "c1", "a1", "b1", "a2", "b2", "a3", "b3", "rmse"]
        self.n_features = band_num * len(feature_outputs)
        self.feature_outputs = feature_outputs
        if logger is None:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s |%(levelname)s| %(funcName)-15s| %(message)s",
                stream=sys.stdout,
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.band_num = band_num

    def predict_features(
        self, block_id: int, cold_block: np.ndarray, year_list_to_predict: list
    ) -> np.ndarray:
        """
        Parameters
        ----------
        block_id: int
            Block id, started from 1
        cold_block: np.ndarray
            Block-based :py:type:`~pyxccd.common.cold_rec_cg` produced by COLD algorithms
        year_list_to_predict: list
            A list of the years to extract features
            Note that the reason for not parsing cold_block to get year bounds is that the year range of blocks
            may vary from each other, so the year bounds are required to be defined from the tile level, not block level
            such as from 'starting_end_date.txt'

        Returns
        -------
        np.ndarray
            3d array, (len(year_list_to_predict), block_width*block_height, n_features)
        """
        block_features = np.full(
            (
                len(year_list_to_predict),
                self.dataset_info.block_width * self.dataset_info.block_height,
                self.n_features,
            ),
            defaults["COMMON"]["NAN_VAL"],
            dtype=np.float32,
        )
        ordinal_day_list = [
            dt.date(year, 7, 1).toordinal() for year in year_list_to_predict
        ]
        if len(cold_block) == 0:
            self.logger.warning(
                "the rec_cg file for block_id={} has no records".format(block_id)
            )
            return block_features

        cold_block_split = np.split(
            cold_block, np.argwhere(np.diff(cold_block["pos"]) != 0)[:, 0] + 1
        )
        for element in cold_block_split:
            # the relative column number in the block
            i_col = get_col_index(
                element[0]["pos"],
                self.dataset_info.n_cols,
                get_block_x(block_id, self.dataset_info.n_block_x),
                self.dataset_info.block_width,
            )
            i_row = get_row_index(
                element[0]["pos"],
                self.dataset_info.n_cols,
                get_block_y(block_id, self.dataset_info.n_block_x),
                self.dataset_info.block_height,
            )

            for band in range(self.band_num):
                feature_row = extract_features(
                    element,
                    band,
                    ordinal_day_list,
                    defaults["COMMON"]["NAN_VAL"],
                    feature_outputs=self.feature_outputs,
                )
                for index in range(int(self.n_features / self.band_num)):
                    block_features[
                        :,
                        i_row * self.dataset_info.block_width + i_col,
                        int(band * self.n_features / self.band_num) + index,
                    ] = feature_row[index]

        return block_features

    def train_rfmodel(self, full_feature_array: np.ndarray, label: np.ndarray):
        """training a random forest model based on feature layers and a label map

        Parameters
        ----------
        full_feature_array : np.ndarray
            3-d array for full feature layers, (nrows, ncols, feature_number)
        label : np.ndarray
            1-d array for a label map

        Returns
        -------
        class
            A sklearn random forest model
        """

        assert label.shape == (self.dataset_info.n_rows, self.dataset_info.n_cols)
        samplecount = generate_sample_num(label, defaults["CLASSIFIER"])
        index_list = []
        label_list = []
        for i in range(defaults["CLASSIFIER"]["total_landcover_category"]):
            index = np.argwhere(label == i + 1)
            np.random.seed(42)  # set random seed to reproduce the same result
            index_sample = index[
                np.random.choice(len(index), int(samplecount[i]), replace=False)
            ]
            index_list.append(index_sample)
            label_list.append(np.array([i + 1] * len(index_sample)))
        index_list = np.vstack(index_list)
        label_list = np.hstack(label_list)
        feature_extraction = np.array(
            [full_feature_array[tuple(x)] for x in index_list]
        ).astype(np.float32)
        rf_model = RandomForestClassifier(random_state=42, max_depth=20)
        rf_model.fit(feature_extraction, label_list)
        return rf_model

    def classify_block(self, rf_model, tmp_feature: np.ndarray) -> np.ndarray:
        """classify feature block for a single year

        Parameters
        ----------
        rf_model : dict
            sklearn random forest model
        tmp_feature : np.ndarray
            2-d array for features, (block_width*block_height, n_features)

        Returns
        -------
        np.ndarray
            2d array, a classification map, (block_height, block_width)
        """
        cmap = rf_model.predict(tmp_feature).reshape(
            self.dataset_info.block_height, self.dataset_info.block_width
        )
        mask = np.all(tmp_feature == defaults["COMMON"]["NAN_VAL"], axis=1).reshape(
            self.dataset_info.block_height, self.dataset_info.block_width
        )
        cmap[mask] = 255
        return cmap

    # def _assemble_covermap(self, blocklist_yearlyclass, year):
    #     full_yearlyclass_array = assemble_array(blocklist_yearlyclass, self.dataset_info.n_block_x)
    #     if (full_yearlyclass_array.shape[1] != self.dataset_info.n_cols) or (full_yearlyclass_array.shape[0] !=
    #                                                                       self.dataset_info.n_rows):
    #         logger.error('The assemble category map is incomplete for {}'.format(year))
    #         return full_yearlyclass_array[:, :, 0]  # we only return the first channel


class PyClassifierHPC(PyClassifier):
    """
    this class adds IO functions based on the HPC environment for the base class
    """

    def __init__(
        self,
        dataset_info: DatasetInfo,
        record_path: str,
        band_num: int = 7,
        year_list_to_predict=list(range(1982, 2022)),
        tmp_path: Optional[str] = None,
        output_path: Optional[str] = None,
        feature_outputs: list = ["a0", "a1", "b1"],
        seedmap_path: Optional[str] = None,
        rf_path: Optional[str] = None,
        logger: Optional[Logger] = None,
    ):
        """_summary_

        Parameters
        ----------
        dataset_info : DatasetInfo
            Time-series dataset information data class
        record_path : str
            Path that are source folder for the COLD results
        band_num : int, optional
            The band number, by default 7
        year_list_to_predict : _type_, optional
            A list of years for classification, by default list(range(1982, 2022))
        tmp_path : Optional[str], optional
            Path for saving temporal files, by default None. if None, will set /record_path/feature_maps
        output_path : Optional[str], optional
            Path to save classification outputS, by default None
        feature_outputs : list, optional
            A list of outputted feature name, it must be within [a0, c1, a1, b1,a2, b2, a3, b3,
            rmse]. by default ["a0", "a1", "b1"]
        seedmap_path : Optional[str], optional
            Path for the seed map that provides labels to produce rf model, by default None
        rf_path : Optional[str], optional
            Path for existing random forest forest, by default None
        logger : Optional[Logger], optional
            Logger handler, by default None

        Raises
        ------
        e
            Raise the error when the parameter inputs for initializing classifier are not correct
        """
        try:
            self._check_inputs_thematic(
                dataset_info, record_path, tmp_path, seedmap_path, rf_path
            )
        except (ValueError, FileExistsError) as e:
            raise e

        self.dataset_info = dataset_info
        self.record_path = record_path
        for feature in feature_outputs:
            assert feature in ["a0", "c1", "a1", "b1", "a2", "b2", "a3", "b3", "rmse"]
        self.feature_outputs = feature_outputs

        if tmp_path is None:
            self.tmp_path = join(record_path, "feature_maps")  # default path
        else:
            self.tmp_path = tmp_path

        if output_path is None:
            self.output_path = join(record_path, "feature_maps")  # default path
        else:
            self.output_path = self.tmp_path

        self.n_features = band_num * len(feature_outputs)

        self.year_list_to_predict = year_list_to_predict
        self.seedmap_path = seedmap_path
        if rf_path is None:
            self.rf_path = join(self.output_path, "rf.model")  # default path
        else:
            self.rf_path = rf_path

        if logger is None:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s |%(levelname)s| %(funcName)-15s| %(message)s",
                stream=sys.stdout,
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.band_num = band_num

    @staticmethod
    def _check_inputs_thematic(
        dataset_info, record_path, tmp_path, seedmap_path, rf_path
    ):
        """check the inputs for initializing a classifier

        Parameters
        ----------
        dataset_info : DatasetInfo
            Basic information of time-series dataset
        record_path : str
            Path that saved COLD results
        tmp_path : str
            Path for saving temporal files, by default None. if None, will set /record_path/feature_maps
        seedmap_path : str
            Path for the seed map that provides labels to produce rf model, by default None
        rf_path : str
            Path for existing random forest forest, by default None
        """
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

        if os.path.isdir(record_path) is False:
            raise FileExistsError("No such directory: {}".format(record_path))

        if seedmap_path is not None:
            if os.path.isfile(seedmap_path) is False:
                raise FileExistsError("No such file: {}".format(seedmap_path))

        if rf_path is not None:
            if os.path.isfile(rf_path) is False:
                raise FileExistsError("No such file: {}".format(rf_path))

    def _save_features(self, block_id, block_features):
        """
        Parameters
        ----------
        block_id: integer
        block_features
        Returns
        -------

        """
        for id, year in enumerate(self.year_list_to_predict):
            np.save(
                os.path.join(self.tmp_path, "tmp_feature_year{}_block{}.npy").format(
                    year, block_id
                ),
                block_features[id, :, :],
            )

    def is_finished_step1_predict_features(self):
        for iblock in range(self.dataset_info.nblocks):
            if not exists(
                join(
                    self.tmp_path,
                    "tmp_step1_predict_{}_finished.txt".format(iblock + 1),
                )
            ):
                return False
        return True

    @staticmethod
    def _save_rf_model(rf_model, rf_path):
        joblib.dump(rf_model, rf_path, compress=3)

    def _is_finished_step2_train_rfmodel(self):
        return exists(self.rf_path)

    def _get_rf_model(self):
        return joblib.load(self.rf_path)

    def _save_yearlyclassification_maps(self, block_id, year, cmap):
        outfile = join(
            self.tmp_path,
            "tmp_yearlyclassification{}_block{}.npy".format(year, block_id),
        )
        np.save(outfile, cmap)

    def _is_finished_step3_classification(self):
        """
        :return: True or false
        """
        for iblock in range(self.dataset_info.nblocks):
            if not exists(
                join(
                    self.tmp_path,
                    "tmp_step3_classification_{}_finished.txt".format(iblock + 1),
                )
            ):
                return False
        return True

    def _save_covermaps(self, full_yearlyclass_array, year):
        np.save(
            join(self.output_path, "yearlyclassification_{}.npy".format(year)),
            full_yearlyclass_array,
        )

    def _clean(self):
        tmp_yearlyclass_filenames = [
            file for file in os.listdir(self.tmp_path) if file.startswith("tmp_")
        ]
        for file in tmp_yearlyclass_filenames:
            os.remove(join(self.tmp_path, file))

    def _get_fullclassification_forcertainyear(self, year):
        tmp_yearlyclass_filenames = [
            file
            for file in os.listdir(self.tmp_path)
            if file.startswith("tmp_yearlyclassification{}".format(year))
        ]

        # sort to guarantee order follows low to high rows
        tmp_yearlyclass_filenames.sort(
            key=lambda t: int(t[t.find("block") + 5 : t.find(".npy")])
        )
        return [
            np.load(join(self.tmp_path, file)).reshape(
                self.dataset_info.block_height, self.dataset_info.block_width, 1
            )
            for file in tmp_yearlyclass_filenames
        ]

    def get_fullfeature_forcertainyear(self, year: int) -> list:
        """get

        Parameters
        ----------
        year : int
            Year to get a feature

        Returns
        -------
        list
            A list of all blocks as 3-d array (block_height, block_width, n_features)
        """
        tmp_feature_filenames = [
            file
            for file in os.listdir(self.tmp_path)
            if file.startswith("tmp_feature_year{}".format(year))
        ]
        if len(tmp_feature_filenames) < self.dataset_info.nblocks:
            self.logger.warning(
                "tmp features are incomplete! should have {}; but actually have {} feature images".format(
                    self.dataset_info.nblocks, len(tmp_feature_filenames)
                )
            )

        tmp_feature_filenames.sort(
            key=lambda t: int(t[t.find("block") + 5 : t.find(".npy")])
        )  # sorted by row number

        return [
            np.load(join(self.tmp_path, file)).reshape(
                self.dataset_info.block_height,
                self.dataset_info.block_width,
                self.n_features,
            )
            for file in tmp_feature_filenames
        ]

        # full_feature_array = assemble_array(, self.dataset_info.n_block_x)
        # if (full_feature_array.shape[1] != self.dataset_info.n_cols) or (full_feature_array.shape[0] !=
        #                                                               self.dataset_info.n_rows):
        #     logger.error('The feature image is incomplete for {}'.format(year))
        # return full_feature_array

    def hpc_preparation(self):
        if exists(self.tmp_path) is False:
            try:
                os.mkdir(self.tmp_path)
            except IOError as e:
                raise e

        if exists(self.output_path) is False:
            try:
                os.mkdir(self.output_path)
            except IOError as e:
                raise e

    def step1_feature_generation(self, block_id: int):
        """generate feature based on COLD results

        Parameters
        ----------
        block_id : int
            the id of block
        """
        if os.path.exists(
            join(self.record_path, "record_change_x{}_y{}_cold.npy").format(
                get_block_x(block_id, self.dataset_info.n_block_x),
                get_block_y(block_id, self.dataset_info.n_block_x),
            )
        ):
            cold_block = np.load(
                join(self.record_path, "record_change_x{}_y{}_cold.npy").format(
                    get_block_x(block_id, self.dataset_info.n_block_x),
                    get_block_y(block_id, self.dataset_info.n_block_x),
                )
            )
            block_features = self.predict_features(
                block_id, cold_block, self.year_list_to_predict
            )
        else:
            block_features = np.full(
                (
                    len(self.year_list_to_predict),
                    self.dataset_info.block_width * self.dataset_info.block_height,
                    self.n_features,
                ),
                defaults["COMMON"]["NAN_VAL"],
                dtype=np.float32,
            )
        self._save_features(block_id, block_features)
        with open(
            join(self.tmp_path, "tmp_step1_predict_{}_finished.txt".format(block_id)),
            "w",
        ):
            pass

    def step2_train_rf(
        self, ref_year: Optional[int] = None, rf_path: Optional[str] = None
    ):
        """training a random forest model, and save it into rf_path

        Parameters
        ----------
        ref_year : int, optional
            The year to provide features which will be further connected to seed map, by default None
        rf_path : str, optional
            Path to save random forest model , by default None. If none, will saved to rf_path

        Raises
        ------
        ValueError
            couldn't locate seedmap that provide label maps
        """
        while not self.is_finished_step1_predict_features():
            time.sleep(5)

        if ref_year is None:
            ref_year = defaults["CLASSIFIER"]["training_year"]

        # if ref_year not in self.year_list_to_predict:
        #     raise Exception("Ref_year {} is not in year_list_to_predict {}. "
        #                     "PLease included it and re-run step1_feature_generation".format(ref_year,
        #                                                                                     self.year_list_to_predict))

        full_feature_array = assemble_array(
            self.get_fullfeature_forcertainyear(ref_year), self.dataset_info.n_block_x
        )
        if self.seedmap_path is None:
            raise ValueError("seedmap_path is not assigned")

        label_array = rio_loaddata(os.fspath(self.seedmap_path))
        rf_model = self.train_rfmodel(full_feature_array, label_array)
        if rf_path is None:
            self._save_rf_model(rf_model, self.rf_path)
        else:
            self._save_rf_model(rf_model, rf_path)

    def step3_classification(self, block_id: int):
        """classify a block

        Parameters
        ----------
        block_id : int_
            the block id

        Raises
        ------
        IOError
            _description_
        """
        while not self._is_finished_step2_train_rfmodel():
            time.sleep(5)
        time.sleep(
            5
        )  # wait for 5 more seconds to guarantee all data of rf.model was down to the disk
        try:
            rf_model = self._get_rf_model()
        except IOError as e:
            raise IOError(
                "Please double check your rf model file directory or generate random forest model first"
            ) from e

        for year in self.year_list_to_predict:
            tmp_feature_block = get_features(
                join(
                    self.tmp_path,
                    "tmp_feature_year{}_block{}.npy".format(year, block_id),
                )
            )
            cmap = self.classify_block(rf_model, tmp_feature_block)
            self._save_yearlyclassification_maps(block_id, year, cmap)
        with open(
            join(
                self.tmp_path,
                "tmp_step3_classification_{}_finished.txt".format(block_id),
            ),
            "w",
        ):
            pass

    def step3_classification_sccd(self, block_id: int):
        """classifying based on sccd output

        Parameters
        ----------
        block_id : int
            block id

        Raises
        ------
        IOError
            couldn't locate rf model file
        """
        while not self._is_finished_step2_train_rfmodel():
            time.sleep(5)
        try:
            rf_model = self._get_rf_model()
        except IOError as e:
            raise IOError(
                "Please double check your rf model file directory or generate random forest model first"
            ) from e

        # for year in self.year_list_to_predict:
        #     tmp_feature_block = get_features(join(self.tmp_path, 'tmp_feature_year{}_block{}.npy'.format(year,
        #                                                                                                 block_id)))
        #     cmap = self.classify_block(rf_model, tmp_feature_block)
        #     self._save_yearlyclassification_maps(block_id, year, cmap)

        tmp_feature_block = get_features(
            join(self.tmp_path, "tmp_feature_now_block{}.npy".format(block_id))
        )
        if exists(
            join(
                self.tmp_path,
                "tmp_step3_classification_{}_finished.txt".format(block_id),
            )
        ):
            return

        cmap = self.classify_block(rf_model, tmp_feature_block)
        self._save_yearlyclassification_maps(block_id, "now", cmap)
        with open(
            join(
                self.tmp_path,
                "tmp_step3_classification_{}_finished.txt".format(block_id),
            ),
            "w",
        ):
            pass

    def step4_assemble(self, clean=True):
        """assesmble all block-based classification results into one map

        Parameters
        ----------
        clean : bool, optional
            _description_, by default True
        """
        while not self._is_finished_step3_classification():
            time.sleep(5)
        for year in self.year_list_to_predict:
            full_yearlyclass_array = assemble_array(
                self._get_fullclassification_forcertainyear(year),
                self.dataset_info.n_block_x,
            )[:, :, 0]
            self._save_covermaps(full_yearlyclass_array, year)
        if clean:
            self._clean()  # _clean all temp files

    def step4_assemble_sccd(self, clean=True):
        """assessmble all block-based classification results based upon sccd into one map

        Parameters
        ----------
        clean : bool, optional
            _description_, by default True
        """
        while not self._is_finished_step3_classification():
            time.sleep(5)

        full_yearlyclass_array = assemble_array(
            self._get_fullclassification_forcertainyear("now"),
            self.dataset_info.n_block_x,
        )[:, :, 0]
        self._save_covermaps(full_yearlyclass_array.astype(np.uint8), "now")

        # for year in self.year_list_to_predict:
        #     full_yearlyclass_array = assemble_array(self._get_fullclassification_forcertainyear(year),
        #                                             self.dataset_info.n_block_x)[:, :, 0]
        #     self._save_covermaps(full_yearlyclass_array, year)
        if clean:
            self._clean()  # _clean all temp files

    def is_finished_step4_assemble(self) -> bool:
        """check if step is finished

        Returns
        -------
        bool
        """
        for year in self.year_list_to_predict:
            if not os.path.exists(
                join(self.output_path, "yearlyclassification_{}.npy".format(year))
            ):
                return False
        else:
            return True
