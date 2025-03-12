import yaml
import os
import shutil
import numpy as np
import joblib
import pathlib

from pyxccd.pyclassifier import PyClassifierHPC
from pyxccd.utils import assemble_array, class_from_dict, rio_loaddata
from pyxccd.common import DatasetInfo


# Use this file to determine where the resources are.  If for some reason we
# are do not have __file__ available (e.g.  copy/pasting in IPython) then
# assume we are in the repo reoot.
# TODO: should likely use pkg_resources instead
try:
    TEST_RESOURCE_DPATH = (pathlib.Path(__file__).parent / 'resources').resolve()
except NameError:
    TEST_RESOURCE_DPATH = pathlib.Path('tests/resources').resolve()


with open(TEST_RESOURCE_DPATH / 'test_config_pyclassifier.yaml', 'r') as yaml_obj:
    test_config = yaml.safe_load(yaml_obj)
data_info = class_from_dict(DatasetInfo, test_config['DATASETINFO'])

ref_path = TEST_RESOURCE_DPATH / 'LCMAP_CU_2001_V01_LCPRI_027009_100_100.tif'
year_lowbound = 2015
year_uppbound = 2020
testing_folder = TEST_RESOURCE_DPATH / 'tmp'


def test_step1_4():
    pyclassifier = PyClassifierHPC(
        data_info, record_path=TEST_RESOURCE_DPATH, tmp_path=testing_folder,
        output_path=testing_folder,
        year_list_to_predict=list(range(year_lowbound, year_uppbound + 1)),
        seedmap_path=ref_path)

    # delete testing folder if it has
    if os.path.exists(pyclassifier.tmp_path):
        shutil.rmtree(pyclassifier.tmp_path)

    pyclassifier.hpc_preparation()

    for iblock in range(data_info.nblocks):
        pyclassifier.step1_feature_generation(iblock + 1)
        for year in range(year_lowbound, year_uppbound + 1):
            assert (pathlib.Path(pyclassifier.tmp_path) /
                    'tmp_feature_year{}_block{}.npy'.format(year, iblock + 1)).exists()

    pyclassifier.step2_train_rf(2017)
    assert (pathlib.Path(pyclassifier.output_path) / 'rf.model').exists()

    for iblock in range(data_info.nblocks):
        pyclassifier.step3_classification(iblock + 1)
        for year in range(year_lowbound, year_uppbound + 1):
            fname = 'tmp_yearlyclassification{}_block{}.npy'.format(year, iblock + 1)
            assert pathlib.Path(pyclassifier.tmp_path, fname).exists()
    pyclassifier.step4_assemble()
    for year in range(year_lowbound, year_uppbound + 1):
        fname = 'yearlyclassification_{}.npy'.format(year)
        assert pathlib.Path(pyclassifier.output_path, fname).exists()

    # import matplotlib.pyplot as plt
    # plt.imshow(np.load(TEST_RESOURCE_DPATH / 'tmp/yearlyclassification_2017.npy'))
    shutil.rmtree(pyclassifier.tmp_path)


def test_predict_features():
    pyclassifier = PyClassifierHPC(
        data_info, record_path=TEST_RESOURCE_DPATH,
        year_list_to_predict=list(range(year_lowbound, year_uppbound + 1)),
        seedmap_path=ref_path)
    cold_block = np.load(TEST_RESOURCE_DPATH / 'record_change_x1_y1_cold.npy')
    block_features = pyclassifier.predict_features(
        1, cold_block, list(range(year_lowbound, year_uppbound + 1)))
    assert np.shape(block_features) == (6, data_info.block_height * data_info.block_width,
                                        21)   # 6 years, 2500 pixels (50, 50), 21 features


def test_train_rf_model():
    pyclassifier = PyClassifierHPC(
        data_info, record_path=TEST_RESOURCE_DPATH, tmp_path=TEST_RESOURCE_DPATH / 'feature_maps',
        year_list_to_predict=list(range(year_lowbound, year_lowbound + 1)),
        seedmap_path=ref_path)

    full_feature_2017 = assemble_array(pyclassifier.get_fullfeature_forcertainyear(2017),
                                       data_info.n_block_x)
    rf_model = pyclassifier.train_rfmodel(
        full_feature_2017, rio_loaddata(os.fspath(ref_path)))
    # pyclassifier.save_rf_model(rf_model)
    assert rf_model is not None


def test_classification_block():
    import pytest
    pytest.skip("Disable this test due to conflicts of Sklearn versions for now")
    pyclassifier = PyClassifierHPC(
        data_info, record_path=TEST_RESOURCE_DPATH, tmp_path=TEST_RESOURCE_DPATH / 'feature_maps',
        rf_path=TEST_RESOURCE_DPATH / 'feature_maps/rf.model',
        year_list_to_predict=list(range(year_lowbound, year_lowbound + 1)))
    rf_model = joblib.load(TEST_RESOURCE_DPATH / 'feature_maps/rf.model')
    tmp_feature_block = np.load(
        TEST_RESOURCE_DPATH / 'feature_maps/tmp_feature_year2017_block1.npy')
    cmap = pyclassifier.classification_block(rf_model, tmp_feature_block)
    print(cmap.shape)
    assert cmap.shape == (50, 50)
