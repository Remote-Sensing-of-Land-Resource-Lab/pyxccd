"""Public feature names and aliases for the original training and database files."""

# Insertion order is the production random forest's input order.
FEATURE_ALIASES = {
    'elevation_m': 'dem',
    'slope_deg': 'slope',
    'seasonal_doy_input': 'sin_doy',
    'seasonal_aux_input': 'cos_doy',
    'tc_brightness_baseline': 'brightness_c0',
    'tc_greenness_baseline': 'greeness_c0',
    'tc_wetness_baseline': 'wetness_c0',
    'elapsed_days_6obs': 'day_num_c6',
    'change_angle_6obs_x100': 'angle_c6',
    'normalized_change_magnitude_6obs_x100': 'norm_cm_c6',
    'tc_brightness_slope_6obs_x100': 'brightness_sl_c6',
    'tc_brightness_intercept_6obs': 'brightness_in_c6',
    'tc_brightness_residual_skewness_6obs_x100': 'brightness_sk_c6',
    'tc_greenness_slope_6obs_x100': 'greeness_sl_c6',
    'tc_greenness_intercept_6obs': 'greeness_in_c6',
    'tc_greenness_residual_skewness_6obs_x100': 'greeness_sk_c6',
    'tc_wetness_slope_6obs_x100': 'wetness_sl_c6',
    'tc_wetness_intercept_6obs': 'wetness_in_c6',
    'tc_wetness_residual_skewness_6obs_x100': 'wetness_sk_c6',
}
FEATURES = list(FEATURE_ALIASES)
LEGACY_FEATURES = list(FEATURE_ALIASES.values())
COLUMN_ALIASES = {**FEATURE_ALIASES, 'detection_date': 'time', 'detection_doy': 'doy'}
CHANGE_MAGNITUDE = 'normalized_change_magnitude_6obs_x100'
DATABASE_ATTRIBUTES = ['row', 'col', 'detection_date', 'detection_doy']


def resolve_columns(available, required, source):
    """Map public column names to the names to read from a file."""
    available = set(available)
    columns = {
        name: name if name in available else COLUMN_ALIASES.get(name, name)
        for name in required
    }
    missing = [name for name, actual in columns.items() if actual not in available]
    if missing:
        raise ValueError(f'{source} is missing required columns: {missing}')
    return columns
