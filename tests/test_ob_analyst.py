import numpy as np
from pyxccd.ob_analyst import segmentation_floodfill
from pyxccd.ob_analyst import object_analysis
from pyxccd.ob_analyst import ObjectAnalystHPC
from pyxccd.utils import class_from_dict
from pyxccd.common import DatasetInfo
import yaml
import shutil
import pathlib

# Use this file to determine where the resources are.  If for some reason we
# are do not have __file__ available (e.g.  copy/pasting in IPython) then
# assume we are in the repo reoot.
# TODO: should likely use pkg_resources instead
try:
    TEST_RESOURCE_DPATH = (pathlib.Path(__file__).parent / 'resources').resolve()
except NameError:
    TEST_RESOURCE_DPATH = pathlib.Path('tests/resources').resolve()

with open(TEST_RESOURCE_DPATH / 'test_config_obanalyst.yaml', 'r') as yaml_obj:
    test_config = yaml.safe_load(yaml_obj)
data_info = class_from_dict(DatasetInfo, test_config['DATASETINFO'])

date = 730329

cm_array = np.load(TEST_RESOURCE_DPATH / 'cm_maps/CM_maps_730329_2000210.npy')
cm_array_l1 = np.load(TEST_RESOURCE_DPATH / 'cm_maps/CM_maps_730297_2000178.npy')
cm_date_array = np.load(TEST_RESOURCE_DPATH / 'cm_maps/CM_date_maps_730329_2000210.npy')
cm_array_l1_date = np.load(TEST_RESOURCE_DPATH / 'cm_maps/CM_date_maps_730297_2000178.npy')


def test_workflow():
    ob_analyst = ObjectAnalystHPC(data_info, starting_date=date,
                                  stack_path=TEST_RESOURCE_DPATH,
                                  result_path=TEST_RESOURCE_DPATH,
                                  thematic_path=TEST_RESOURCE_DPATH / 'feature_maps')
    ob_analyst.hpc_preparation()
    ob_analyst.obia_execute(date, method='slic')
    assert ob_analyst.is_finished_object_analysis(date_list=[date])
    shutil.rmtree(ob_analyst.obia_path)
    shutil.rmtree(ob_analyst.obcold_recg_path)


def test_segmentation():
    [object_map_s1, cm_date_array_updated, object_map_s2, s1_info] = segmentation_floodfill(
        cm_array, cm_date_array, cm_array_l1, cm_array_l1_date)
    assert len(np.unique(object_map_s1)) > 1
    assert len(np.unique(object_map_s2)) > 1


def test_object_analysis():
    [object_map_s1, cm_date_array_updated, object_map_s2, s1_info] = segmentation_floodfill(
        cm_array, cm_date_array, cm_array_l1, cm_array_l1_date)
    classification_map = np.load(TEST_RESOURCE_DPATH / 'feature_maps/yearlyclassification_1999.npy')
    uniform_threshold = None
    uniform_sizeslope = None
    parameters = None
    change_map = object_analysis(
        object_map_s1, object_map_s2, s1_info, classification_map,
        uniform_threshold, uniform_sizeslope, parameters
    )
    # import matplotlib.pyplot as plt
    # plt.imshow(change_map)
    # test_config = yaml.safe_load(yaml_obj)
    assert len(np.unique(change_map)) > 1

    # import matplotlib
    # from random import random
    # import matplotlib.pyplot as plt
    # colors = [(1,1,1)] + [(random(),random(),random()) for i in range(255)]
    # new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
    # plt.imshow(object_map_s1, cmap=new_map)


def test_get_lastyear_cmap_fromdate():
    ob_analyst = ObjectAnalystHPC(data_info, starting_date=date - 366,
                                  stack_path=TEST_RESOURCE_DPATH,
                                  result_path=TEST_RESOURCE_DPATH,
                                  thematic_path=TEST_RESOURCE_DPATH / 'feature_maps')
    cmap = ob_analyst.get_lastyear_cmap_fromdate(date)
    assert cmap is not None
