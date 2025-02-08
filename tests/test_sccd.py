from pyxccd import sccd_detect, sccd_update
from pyxccd.utils import read_data
import pickle
import pathlib
from collections import namedtuple
SccdOutput = namedtuple("SccdOutput", "position rec_cg min_rmse nrt_mode nrt_model nrt_queue")

# Use this file to determine where the resources are.  If for some reason we
# are do not have __file__ available (e.g.  copy/pasting in IPython) then
# assume we are in the repo reoot.
# TODO: should likely use pkg_resources instead
try:
    TEST_RESOURCE_DPATH = (pathlib.Path(__file__).parent / 'resources').resolve()
except NameError:
    TEST_RESOURCE_DPATH = pathlib.Path('tests/resources').resolve()


def test_sccd_detect():
    import pytest
    pytest.skip("Disable SCCD for now")
    # running COLD for a Landsat time series provided by a csv
    in_path = TEST_RESOURCE_DPATH / 'spectral_336_3980_obs.csv'
    data = read_data(in_path)
    dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, sensor = data.copy()  # exclude header
    sccd_result = sccd_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)
    assert sccd_result.rec_cg['t_break'] == 735016
    assert sccd_result.rec_cg['num_obs'] == 141
    assert sccd_result.nrt_model['num_obs'] == 63
    assert sccd_result.nrt_mode == 1


def test_sccd_update():
    import pytest
    pytest.skip("Disable SCCD for now")
    import numpy as np
    pickle_file = open(TEST_RESOURCE_DPATH / 'spectral_15043028_sccdplot.pickle', 'rb')

    sccd_plot = SccdOutput(*pickle.load(pickle_file))

    # Test that the named tuple structure is the same
    from pyxccd._colds_cython import SccdOutput as SccdOutputCy
    assert SccdOutputCy._fields == SccdOutput._fields
    assert sccd_plot._fields == SccdOutput._fields

    in_path = TEST_RESOURCE_DPATH / 'spectral_15043028_ext.csv'
    data = read_data(in_path)
    dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, sensor = data.copy()  # exclude header

    for idx, (key, item) in enumerate(zip(sccd_plot._fields, sccd_plot)):
        print('----------')
        print(f'idx={idx}, key={key}')
        if isinstance(item, np.ndarray):
            item.shape
            print(f'item.shape={item.shape}')
            print(f'item.dtype={item.dtype}')
        else:
            print(f'item={item}')

    # Check that the packed numpy dtype hasn't unexpectedly changed between
    # testdata and the current library version.
    from pyxccd._colds_cython import nrtmodel_dt
    sccd_pack = sccd_plot
    assert nrtmodel_dt.itemsize == sccd_pack.nrt_model.dtype.itemsize, (
        'The test data does not correspond to the current code')

    sccd_plot_new = sccd_update(sccd_pack, dates, blues, greens, reds,
                                nirs, swir1s, swir2s, thermals, qas)
    assert sccd_plot_new.nrt_model['num_obs'] == 64
    assert sccd_plot_new.nrt_model['obs_date_since1982'][0][3] == 9059
