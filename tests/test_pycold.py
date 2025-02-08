# this script show how to run COLD from a csv file that store time series information
import numpy as np
import pathlib
from pyxccd import cold_detect, obcold_reconstruct
from pyxccd.utils import read_data

Landsat_bandname = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Thermal']
t_c = -200  # the threshold used for get_breakcategory
w = np.pi * 2 / 365.25

# slope_in_rec_cg = original_slope * slope_scale, we used this way to keep precision for model coefficients in
# float datatype
slope_scale = 10000


# Use this file to determine where the resources are.  If for some reason we
# are do not have __file__ available (e.g.  copy/pasting in IPython) then
# assume we are in the repo reoot.
# TODO: should likely use pkg_resources instead
try:
    TEST_RESOURCE_DPATH = (pathlib.Path(__file__).parent / 'resources').resolve()
except NameError:
    TEST_RESOURCE_DPATH = pathlib.Path('tests/resources').resolve()


def get_breakcategory(ccd_plot, i_curve):
    """
    get break category:
    :param ccd_plot: rec_cg
    :param i_curve: the number of the curve to be analysised
    :return: 1 - disturbance break; 2 - natural recovery; 3 - aforestation
    see section 3.3.7 in Zhu, Z., Zhang, J., Yang, Z., Aljaddani, A. H., Cohen, W. B., Qiu, S., & Zhou, C. (2020).
    Continuous monitoring of land disturbance based on Landsat time series. Remote Sensing of Environment, 238, 111116.
    """
    if ccd_plot[i_curve]['magnitude'][3] > t_c and ccd_plot[i_curve]['magnitude'][2] < -t_c and \
            ccd_plot[i_curve]['magnitude'][4] < -t_c:
        if ccd_plot[i_curve + 1]['coefs'][3, 1] > np.abs(ccd_plot[i_curve]['coefs'][3, 1]) and \
                ccd_plot[i_curve + 1]['coefs'][2, 1] < -np.abs(ccd_plot[i_curve]['coefs'][2, 1]) and \
                ccd_plot[i_curve + 1]['coefs'][4, 1] < -np.abs(ccd_plot[i_curve]['coefs'][4, 1]):
            return 3
        else:
            return 2
    else:
        return 1


def test_cold_detect():
    # running COLD for a Landsat time series provided by a csv
    in_path = TEST_RESOURCE_DPATH / 'spectral_336_3980_obs.csv'
    data = read_data(in_path)
    dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, sensor = data.copy()  # exclude header
    cold_result = cold_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)
    assert cold_result[0]['t_break'] == 735040
    assert cold_result[0]['num_obs'] == 142
    assert cold_result[1]['t_end'] == 737368


def test_obcold_reconstruct():
    # running reconstructing for a Landsat time series provided by a csv
    in_path = TEST_RESOURCE_DPATH / 'spectral_336_3980_obs.csv'
    break_path = TEST_RESOURCE_DPATH / 'spectral_336_3980_breaks.csv'
    data = read_data(in_path)
    breaks = read_data(break_path)
    dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, sensor = data.copy()  # exclude header
    cold_result = obcold_reconstruct(dates, blues, greens, reds,
                                     nirs, swir1s, swir2s, thermals, qas, breaks)
    assert cold_result[0]['t_break'] == 735040
    # assert cold_result[0]['num_obs'] == 139
    # assert cold_result[1]['t_end'] == 737352
