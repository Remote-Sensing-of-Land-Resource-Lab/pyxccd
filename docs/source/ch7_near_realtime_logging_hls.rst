第7课：近实时监测
===================================

**作者: Su Ye (remotesensingsuy@gmail.com)**

**时间序列数据集: Harmonized Landsat-Sentinel (HLS) 数据集**

**应用: 中国四川的伐木活动**

**随机连续变化检测（Stochastic Continuous Change Detection, S-CCD）** 结合了卡尔曼滤波器，这消除了每当新数据输入时需要重新拟合整个时间序列的需求 [1]。相反，模型系数（趋势和季节参数）以短记忆的方式增量更新，因此算法不会保留整个历史记录。一旦观测值被同化（或丢弃），原始数据就不再存储。这种设计使得算法在数据连续到达的近实时应用中具有可扩展性。

*[1] Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A
near-real-time approach for monitoring forest disturbance using Landsat
time series: Stochastic continuous change detection. Remote Sensing of
Environment, 252, 112167.*

--------------

回顾性数据处理
-----------------------------

让我们使用四川的一个 HLS 示例来演示 NRT 监测伐木活动。

为了启用 NRT 监测，我们需要运行 ``sccd_detect`` 或 ``sccd_detect_flex`` 来处理直到当前日期的历史时间序列数据集。假设当前日期是“2024-04-04”，我们首先需要处理从 2016 年到今天的历史 HLS 数据集：

.. code:: ipython3

    import numpy as np
    import pathlib
    import pandas as pd
    from typing import List, Tuple, Dict, Union, Optional
    from matplotlib.axes import Axes
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pyxccd import sccd_detect
    from pyxccd.common import SccdOutput
    from pyxccd.utils import getcategory_sccd, defaults
    
    def display_sccd_result(
        data: np.ndarray,
        band_names: List[str],
        band_index: int,
        sccd_result: SccdOutput,
        axe: Axes,
        title: str = 'S-CCD',
        plot_kwargs: Optional[Dict] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Compare COLD and SCCD change detection algorithms by plotting their results side by side.
        
        This function takes time series remote sensing data, applies both COLD and SCCD algorithms,
        and visualizes the curve fitting and break detection results for comparison. 
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array with shape (n_observations, n_bands + 2) where:
            - First column: ordinal dates (days since January 1, AD 1)
            - Next n_bands columns: spectral band values
            - Last column: QA flags (0-clear, 1-water, 2-shadow, 3-snow, 4-cloud)
            
        band_names : List[str]
            List of band names corresponding to the spectral bands in the data (e.g., ['red', 'nir'])
            
        band_index : int
            1-based index of the band to plot (e.g., 0 for first band, 1 for second band)
            
        sccd_result: SccdOutput
            Output of sccd_detect
        
        axe: Axes
            An Axes object represents a single plot within that Figure
        
        title: Str
            The figure title. The default is "S-CCD"
            
        plot_kwargs : Dict, optional
            Additional keyword arguments to pass to the display function. Possible keys:
            - 'marker_size': size of observation markers (default: 5)
            - 'marker_alpha': transparency of markers (default: 0.7)
            - 'line_color': color of model fit lines (default: 'orange')
            - 'font_size': base font size (default: 14)
            
        Returns:
        --------
        Tuple[plt.Figure, List[plt.Axes]]
            A tuple containing the matplotlib Figure object and a list of Axes objects
            (top axis is COLD results, bottom axis is SCCD results)
        
        """
        w = np.pi * 2 / 365.25
    
        # Set default plot parameters
        default_plot_kwargs: Dict[str, Union[int, float, str]] = {
            'marker_size': 5,
            'marker_alpha': 0.7,
            'line_color': 'orange',
            'font_size': 14
        }
        if plot_kwargs is not None:
            default_plot_kwargs.update(plot_kwargs)
    
        # Extract values with proper type casting
        font_size = default_plot_kwargs.get('font_size', 14)
        try:
            title_font_size = int(font_size) + 2
        except (TypeError, ValueError):
            title_font_size = 16 
    
    
        # Clean and prepare data
        data = data[np.all(np.isfinite(data), axis=1)]
        data_df = pd.DataFrame(data, columns=['dates'] + band_names + ['qa'])
    
    
        # Plot COLD results
        w = np.pi * 2 / 365.25
        slope_scale = 10000
    
        # Prepare clean data for COLD plot
        data_clean = data_df[(data_df['qa'] == 0) | (data_df['qa'] == 1)].copy()
        data_clean =  data_clean[(data_clean >= 0).all(axis=1) & (data_clean.drop(columns="dates") <= 10000).all(axis=1)]
        calendar_dates = [pd.Timestamp.fromordinal(int(row)) for row in data_clean["dates"]]
        data_clean.loc[:, 'dates_formal'] = calendar_dates
        
        # Calculate y-axis limits
        band_name = band_names[band_index]
        band_values = data_clean[data_clean['qa'] == 0 | (data_clean['qa'] == 1)][band_name]
        # band_values  = band_values[band_values <10000]
        q01, q99 = np.quantile(band_values, [0.01, 0.99])
        extra = (q99 - q01) * 0.4
        ylim_low = q01 - extra
        ylim_high = q99 + extra
    
        # Plot SCCD observations
        axe.plot(
            'dates_formal', band_name, 'go',
            markersize=default_plot_kwargs['marker_size'],
            alpha=default_plot_kwargs['marker_alpha'],
            data=data_clean
        )
    
        # Plot SCCD segments
        for segment in sccd_result.rec_cg:
            j = np.arange(segment['t_start'], segment['t_break'] + 1, 1)
            if len(segment['coefs'][band_index]) == 8:
                plot_df = pd.DataFrame(
                    {
                    'dates': j,
                    'trend': j * segment['coefs'][band_index][1] / slope_scale + segment['coefs'][band_index][0],
                    'annual': np.cos(w * j) * segment['coefs'][band_index][2] + np.sin(w * j) * segment['coefs'][band_index][3],
                    'semiannual': np.cos(2 * w * j) * segment['coefs'][band_index][4] + np.sin(2 * w * j) * segment['coefs'][band_index][5],
                    'trimodal': np.cos(3 * w * j) * segment['coefs'][band_index][6] + np.sin(3 * w * j) * segment['coefs'][band_index][7]
                })
            else:
                plot_df = pd.DataFrame(
                    {
                    'dates': j,
                    'trend': j * segment['coefs'][band_index][1] / slope_scale + segment['coefs'][band_index][0],
                    'annual': np.cos(w * j) * segment['coefs'][band_index][2] + np.sin(w * j) * segment['coefs'][band_index][3],
                    'semiannual': np.cos(2 * w * j) * segment['coefs'][band_index][4] + np.sin(2 * w * j) * segment['coefs'][band_index][5],
                    'trimodal': j * 0
                })
            plot_df['predicted'] = (
                plot_df['trend'] + 
                plot_df['annual'] + 
                plot_df['semiannual']+
                plot_df['trimodal']
            )
    
            # Convert dates and plot model fit
            calendar_dates = [pd.Timestamp.fromordinal(int(row)) for row in plot_df["dates"]]
            plot_df.loc[:, 'dates_formal'] = calendar_dates
            g = sns.lineplot(
                x="dates_formal", y="predicted",
                data=plot_df,
                label="Model fit",
                ax=axe,
                color=default_plot_kwargs['line_color']
            )
            if g.legend_ is not None: 
                g.legend_.remove()
    
        # Plot near-real-time projection for SCCD if available
        if hasattr(sccd_result, 'nrt_mode') and (sccd_result.nrt_mode %10 == 1 or sccd_result.nrt_mode == 3 or sccd_result.nrt_mode %10 == 5):
            recent_obs = sccd_result.nrt_model['obs_date_since1982'][sccd_result.nrt_model['obs_date_since1982']>0]
            j = np.arange(
                sccd_result.nrt_model['t_start_since1982'].item() + defaults['COMMON']['JULIAN_LANDSAT4_LAUNCH'], 
                recent_obs[-1].item()+ defaults['COMMON']['JULIAN_LANDSAT4_LAUNCH']+1, 
                1
            )
    
            if len(sccd_result.nrt_model['nrt_coefs'][band_index]) == 8:
                plot_df = pd.DataFrame(
                    {
                    'dates': j,
                    'trend': j * sccd_result.nrt_model['nrt_coefs'][band_index][1] / slope_scale + sccd_result.nrt_model['nrt_coefs'][band_index][0],
                    'annual': np.cos(w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][2] + np.sin(w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][3],
                    'semiannual': np.cos(2 * w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][4] + np.sin(2 * w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][5],
                    'trimodal': np.cos(3 * w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][6] + np.sin(3 * w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][7]
                })
            else:
                plot_df = pd.DataFrame(
                    {
                    'dates': j,
                    'trend': j * sccd_result.nrt_model['nrt_coefs'][band_index][1] / slope_scale + sccd_result.nrt_model['nrt_coefs'][band_index][0],
                    'annual': np.cos(w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][2] + np.sin(w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][3],
                    'semiannual': np.cos(2 * w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][4] + np.sin(2 * w * j) * sccd_result.nrt_model['nrt_coefs'][band_index][5],
                    'trimodal': j * 0
                })
                
            plot_df['predicted'] = plot_df['trend'] + plot_df['annual'] + plot_df['semiannual']+ plot_df['trimodal']
            calendar_dates = [pd.Timestamp.fromordinal(int(row)) for row in plot_df["dates"]]
            plot_df.loc[:, 'dates_formal'] = calendar_dates
            g = sns.lineplot(
                x="dates_formal", y="predicted",
                data=plot_df,
                label="Model fit",
                ax=axe,
                color=default_plot_kwargs['line_color']
            )
            if g.legend_ is not None: 
                g.legend_.remove()
    
        for i in range(len(sccd_result.rec_cg)):
            if getcategory_sccd(sccd_result.rec_cg, i) == 1:
                axe.axvline(pd.Timestamp.fromordinal(sccd_result.rec_cg[i]['t_break']), color='k')
            else:
                axe.axvline(pd.Timestamp.fromordinal(sccd_result.rec_cg[i]['t_break']), color='r')
        
        axe.set_ylabel(f"{band_name} * 10000", fontsize=default_plot_kwargs['font_size'])
    
        # Handle tick params with type safety
        tick_font_size = default_plot_kwargs['font_size']
        if isinstance(tick_font_size, (int, float)):
            axe.tick_params(axis='x', labelsize=int(tick_font_size)-1)
        else:
            axe.tick_params(axis='x', labelsize=13)  # fallback
    
        axe.set(ylim=(ylim_low, ylim_high))
        axe.set_xlabel("", fontsize=6)
    
        # Format spines
        for spine in axe.spines.values():
            spine.set_edgecolor('black')
        title_font_size = int(font_size) + 2 if isinstance(font_size, (int, float)) else 16
        axe.set_title(title, fontweight="bold", size=title_font_size, pad=2)
    
    
    TUTORIAL_DATASET = (pathlib.Path.cwd() / 'datasets').resolve() # modify it as you need
    assert TUTORIAL_DATASET.exists()
    in_path = TUTORIAL_DATASET/ '7_logging_hls_w0.csv'
    
    # read example csv for HLS time series
    data = pd.read_csv(in_path)
    
    # split the array by the column
    dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, sensor = data.to_numpy().copy().T
    
    # retrospective processing
    sccd_result = sccd_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, qas)
    
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    
    # Let's plot NIR and SWIR2 time series, which are the best two disturbance indicator bands
    display_sccd_result(data=np.stack((dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=3, sccd_result=sccd_result, axe=axes[0], title="Retrospective S-CCD")
    
    display_sccd_result(data=np.stack((dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=5, sccd_result=sccd_result, axe=axes[1], title="Retrospective S-CCD")



.. image:: 7_near_realtime_logging_hls_files/7_near_realtime_logging_hls_1_0.png


我们可以看到在 2018 年发生了一次历史干扰，很可能是由于胁迫干扰。但由于本课是为 NRT 监测设计的，我们只关注时间序列尾部正在发生的近期干扰。

如第 1 课简要介绍，``sccd_result`` 是一个结构化对象，包含六个元素。

+-----------------------+-----------------------+-----------------------+
| Element               | Datatype              | Description           |
+=======================+=======================+=======================+
| position              | int                   | Position of current   |
|                       |                       | pixel, commonly coded |
|                       |                       | as 10000*row+col      |
+-----------------------+-----------------------+-----------------------+
| rec_cg                | ndarray               | Temporal segment      |
|                       |                       | obtained by           |
|                       |                       | retrospective break   |
|                       |                       | detection             |
+-----------------------+-----------------------+-----------------------+
| nrt_mode              | int                   | Current mode: the 1st |
|                       |                       | digit indicate        |
|                       |                       | predictability and    |
|                       |                       | the 2nd is for        |
|                       |                       | ``nrt_model``         |
|                       |                       | availability          |
+-----------------------+-----------------------+-----------------------+
| nrt_model             | ndarray               | Near real-time model  |
|                       |                       | for the last segment, |
|                       |                       | which will be         |
|                       |                       | recursively updated   |
+-----------------------+-----------------------+-----------------------+
| nrt_queue             | ndarray               | Near real-time        |
|                       |                       | observations stored   |
|                       |                       | in a queue when       |
|                       |                       | ``nrt_model`` is not  |
|                       |                       | initialized           |
+-----------------------+-----------------------+-----------------------+
| min_rmse              | ndarray               | Minimum rmse in CCDC  |
|                       |                       | to avoid              |
|                       |                       | overdetection from    |
|                       |                       | black body            |
+-----------------------+-----------------------+-----------------------+

让我们在当前阶段打印相关的 ``nrt_model`` 信息：

.. code:: ipython3

    sccd_result




.. raw:: html

    <style>
    /* 覆盖样式 */
    .output-block .highlight {
        background: transparent !important;
        margin-bottom: 0 !important;
    }
    .output-block .highlight pre {
        background-color: #f0f4ff !important;
        padding: 0.8em !important;
        margin: 0 !important;          
        border-radius: 0 !important;
    }
    /* 添加底部间距 */
    .output-block {
        margin-bottom: 1.5em !important;  
    }
    </style>

.. code:: text
    :class: output-block

    SccdOutput(position=1, rec_cg=array([(735625, 736954, 60, [[ 1.48542393e+04, -1.98632324e+02, -4.02296638e+01,  2.51553841e+01, -1.45135765e+01,  9.93953896e+00], [ 8.38789941e+03, -1.07453278e+02, -8.17535400e+01,  4.86201286e+01,  1.21395941e+01, -1.15295398e+00], [ 2.46381088e+02,  9.38574553e-01, -2.61588840e+01,  7.75747833e+01, -2.77891006e+01,  1.91563301e+01], [-4.81553398e+04,  6.92187561e+02, -4.14877747e+02, -2.82666321e+01,  1.99714539e+02, -4.93824234e+01], [ 2.35799585e+03, -1.30108614e+01, -1.74128586e+02,  1.30244141e+02,  2.11633873e+01,  1.98471394e+01], [ 6.30437939e+03, -7.71522598e+01, -5.81244240e+01,  9.30946274e+01, -1.19096756e+01,  1.11940117e+01]], [ 35.270123,  43.48861 ,  45.4437  , 163.38066 ,  83.16016 ,  46.84747 ], [  15.8222275,   26.901932 ,   28.303795 , -266.93457  ,  203.08496  ,  130.46231  ])],
          dtype={'names': ['t_start', 't_break', 'num_obs', 'coefs', 'rmse', 'magnitude'], 'formats': ['<i4', '<i4', '<i4', ('<f4', (6, 6)), ('<f4', (6,)), ('<f4', (6,))], 'offsets': [0, 4, 8, 12, 156, 180], 'itemsize': 204, 'aligned': True}), min_rmse=array([ 36,  35,  32, 135,  68,  35], dtype=int16), nrt_mode=1, nrt_model=np.void((13212, 194, [[ 208,  224,  294,  229,  248,  280,    0,    0], [ 429,  454,  497,  452,  489,  505,    0,    0], [ 404,  425,  454,  418,  423,  472,    0,    0], [2312, 2346, 2454, 2387, 2669, 2581,    0,    0], [1459, 1462, 1523, 1503, 1580, 1634,    0,    0], [ 688,  712,  715,  702,  743,  776,    0,    0]], [15212, 15217, 15219, 15222, 15227, 15232,     0,     0], [[102.346245, 0.04497578, -30.347172, 45.64486, -8.82943, 2.1275249, 0.04497578, 0.00013879126, -0.012609219, 0.018866414, -0.0034646979, 0.0018913492, -30.347172, -0.012609219, 119.790405, -5.1260967, -16.251162, 28.159357, 45.64486, 0.018866414, -5.1260967, 140.28548, -30.084187, -27.806128, -8.82943, -0.0034646979, -16.251162, -30.084187, 102.9957, 9.017389, 2.1275249, 0.0018913492, 28.159357, -27.806128, 9.017389, 120.91161], [107.10884, 0.048987884, -31.711811, 47.068657, -8.922271, 1.9123313, 0.048987884, 0.0001462898, -0.013710619, 0.020167004, -0.0035791849, 0.00196249, -31.711811, -0.013710619, 124.05986, -5.503266, -16.729149, 28.45994, 47.068657, 0.020167004, -5.503266, 144.86067, -30.926754, -28.806025, -8.922271, -0.0035791849, -16.729149, -30.926754, 106.99994, 9.354953, 1.9123313, 0.00196249, 28.45994, -28.806025, 9.354953, 124.93636], [119.9631, 0.05424634, -35.35254, 50.44321, -9.054462, 1.158471, 0.05424634, 0.00014838214, -0.014942498, 0.020941857, -0.0033436487, 0.0020050062, -35.35254, -0.014942498, 137.51291, -6.3354993, -18.420902, 29.369724, 50.44321, 0.020941857, -6.3354993, 159.0635, -33.54387, -31.973179, -9.054462, -0.0033436487, -18.420902, -33.54387, 119.844986, 10.4234495, 1.158471, 0.0020050062, 29.369724, -31.973179, 10.4234495, 137.8432], [270.5095, 0.12888381, -75.77823, 84.981155, -12.605128, -4.579196, 0.12888381, 0.00020805227, -0.02948218, 0.027392946, 0.00049356616, 0.007500413, -75.77823, -0.02948218, 265.0389, -15.28049, -37.84924, 37.26224, 84.981155, 0.027392946, -15.28049, 300.24518, -61.682697, -62.50038, -12.605128, 0.00049356616, -37.84924, -61.682697, 243.35031, 22.362244, -4.579196, 0.007500413, 37.26224, -62.50038, 22.362244, 262.02927], [201.3874, 0.0949685, -57.748226, 69.92133, -10.7411785, -2.253249, 0.0949685, 0.00018210562, -0.023727853, 0.025870612, -0.0016557443, 0.004314661, -57.748226, -0.023727853, 210.62979, -11.125076, -29.022526, 33.581055, 69.92133, 0.025870612, -11.125076, 238.27393, -48.60641, -48.75678, -10.7411785, -0.0016557443, -29.022526, -48.60641, 190.04839, 16.61674, -2.253249, 0.004314661, 33.581055, -48.75678, 16.61674, 208.46994], [130.30319, 0.060314957, -38.273083, 53.171722, -9.230985, 0.67056316, 0.060314957, 0.00015543179, -0.016456433, 0.0222466, -0.0032705672, 0.002172302, -38.273083, -0.016456433, 147.19635, -7.0093193, -19.689304, 29.95765, 53.171722, 0.0222466, -7.0093193, 169.38004, -35.444485, -34.178364, -9.230985, -0.0032705672, -19.689304, -35.444485, 129.04063, 11.181543, 0.67056316, 0.002172302, 29.95765, -34.178364, 11.181543, 147.07889]], [[4478.114, -57.458527, -51.625023, 53.003376, -3.087816, -1.4335045], [4847.253, -59.154026, -98.957275, 57.93207, 12.371733, 3.2117136], [7299.53, -94.44655, -39.032204, 108.598015, -24.481653, 3.4193916], [-107754.68, 1496.6321, -470.6851, -155.14989, 133.37932, 76.867035], [31400.36, -405.64944, -218.80751, 180.55144, -4.065642, -7.789848], [25347.256, -334.49185, -84.89389, 142.63063, -19.270958, -13.397055]], [1998.7078, 2154.9038, 2691.485, 10128.377, 6481.8506, 3107.4956], [ 226549,  217558,  310620, 5765251, 1759864,  593911], -9999, -9999, 0), dtype={'names': ['t_start_since1982', 'num_obs', 'obs', 'obs_date_since1982', 'covariance', 'nrt_coefs', 'H', 'rmse_sum', 'norm_cm', 'cm_angle', 'anomaly_conse'], 'formats': ['<i2', '<i2', ('<i2', (6, 8)), ('<i2', (8,)), ('<f4', (6, 36)), ('<f4', (6, 6)), ('<f4', (6,)), ('<u4', (6,)), '<i2', '<i2', 'u1'], 'offsets': [0, 2, 4, 100, 116, 980, 1124, 1148, 1172, 1174, 1176], 'itemsize': 1180, 'aligned': True}), nrt_queue=array([], dtype=float64))



.. code:: ipython3

    # check if the current mode is still 1
    print(f"The current nrt mode is {sccd_result.nrt_mode}")
    
    # check the number of current anomlies
    print(f"The current number of consecutive anomlies is {sccd_result.nrt_model['anomaly_conse']}")
    recent_obs_date = sccd_result.nrt_model['obs_date_since1982'][sccd_result.nrt_model['obs_date_since1982']>0][-1]
    
    # check the last observation to be processed. Note that the observation date was formated the ordinal dates since the date of LANDSAT4_LAUNCH (723742) to save the date into int16. The user need to convert it to the formal date through the below code
    print(f"The date of the last observations being processed is {pd.Timestamp.fromordinal(recent_obs_date.item() +defaults['COMMON']['JULIAN_LANDSAT4_LAUNCH'])}")
    
    print(f"The observation number in the current segment is {sccd_result.nrt_model['num_obs']}")


.. code:: text
    :class: output-block

    The current nrt mode is 1
    The current number of consecutive anomlies is 0
    The date of the last observations being processed is 2024-03-29 00:00:00
    The observation number in the current segment is 194
    

``nrt_mode = 1`` 表示该像元**当前处于 NRT 监测阶段且具有可预测性**。``nrt_mode`` 有两位数字。

第一位数字：

0 - 具有可预测性

1 - 无可预测性

第二位数字：

0 - 无效模式，尚未初始化

1 - 监测模式

2 - 队列模式。一旦检测到断点，模式从监测模式过渡到队列模式

3 - 雪的监测模式

4 - 雪的队列模式

5 - 从监测模式到队列模式的过渡模式（同时保留 nrt_model 和 nrt_queue），在首次检测到断点后保持 15 天

一旦检测到 ``anomaly``，但其变化幅度只是中小程度，则不会触发 ``break``。在这种情况下，S-CCD 可能难以有效检测后续变化，即使 ``sccd_result`` 保留了 ``nrt_model``。这种限制的出现是因为 ``anomaly`` 向 ``nrt_model`` 注入了额外的波动，降低了其稳定性。

为了解决这个问题，S-CCD 会对每批新的观测值进行可预测性测试。该测试评估三个连续的观测值，检查它们的残差（与预测反射率的差异）是否低于阈值。``nrt_mode`` 的第一位数字保持为 1，直到通过可预测性测试（然后第一位数字将变为 0）。

第二位数字编码了 ``nrt_model`` 的可用性。我们分别使用 ``1`` 和 ``3`` 来表示正常情况下和雪况下 ``nrt_model`` 的可用性，使用 ``2`` 和 ``4`` 来表示缺乏 ``nrt_model``，因此收集观测值直到达到 CCDC 初始化条件（即队列模式）。

双周递归更新
--------------------------

第 1-2 周：发现异常
~~~~~~~~~~~~~~~~~~~~~~~~~

现在，让我们以两周为步长对 ``sccd_result`` 进行增量更新。读取自“2024-04-04”以来的前两周观测值，并使用 ``sccd_update`` 更新 ``sccd_result``：

.. code:: ipython3

    from pyxccd import sccd_update
    
    TUTORIAL_DATASET = (pathlib.Path.cwd() / 'datasets').resolve() # modify it as you need
    assert TUTORIAL_DATASET.exists()
    in_path = TUTORIAL_DATASET/ '7_logging_hls_w12.csv'
    
    # read example csv for HLS time series
    data_1 = pd.read_csv(in_path)
    
    # split the array by the column
    dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, thermals1, qas1, sensor1 = data_1.to_numpy().copy().T
    sccd_result = sccd_update(sccd_result, dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, qas1)
    
    dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m, sensor_m = np.concatenate((data, data_1)).copy().T
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    
    # Let's plot NIR and SWIR2 time series, which are the best two disturbance indicator bands
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=3, sccd_result=sccd_result, axe=axes[0], title="NRT S-CCD (Week 1-2)")
    
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=5, sccd_result=sccd_result, axe=axes[1], title="NRT S-CCD (Week 1-2)")
    
    print(f"The current nrt mode is {sccd_result.nrt_mode}")
    print(f"The current number of consecutive anomlies is {sccd_result.nrt_model['anomaly_conse']}")
    print(f"The observation number in the current segment is {sccd_result.nrt_model['num_obs']}")


.. code:: text
    :class: output-block

    The current nrt mode is 1
    The current number of consecutive anomlies is 3
    The observation number in the current segment is 197
    


.. image:: 7_near_realtime_logging_hls_files/7_near_realtime_logging_hls_6_1.png


从打印的信息中，你可以看到连续异常的数量从 0 跳到了 3，表明 S-CCD 在最近两周内检测到了三个异常。这些异常对应于上述 NIR 和 SWIR2 时间序列尾部的三个明显离群点。这些异常的发生是因为，在伐木之后，地表的所有树木都被移除，导致包括 NIR 波段在内的几乎所有 Landsat 波段反射率增加（注意：伐木并不一定导致 NIR 反射率下降）。

同时，当前时间段的观测计数从 194 增加到 197，证实了在此期间已有三个新的观测值被处理并纳入该时间段。这三个新观测值都被识别为光谱异常。

异常检测的灵敏度可以通过 ``sccd_update`` 中的参数 ``anomaly_pcg`` 进行调整。其默认值为 0.9，对应于使用卡方分布在 90% 概率水平下的临界值。降低此阈值会使检测器更灵敏（捕捉到更弱的异常），但也会增加误报的风险。

第 3-4 周：检测到断点！
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    
    in_path = TUTORIAL_DATASET/ '7_logging_hls_w34.csv'
    
    # read example csv for HLS time series
    data_2 = pd.read_csv(in_path)
    
    # split the array by the column
    dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, thermals1, qas1, sensor1 = data_2.to_numpy().copy().T
    sccd_result = sccd_update(sccd_result, dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, qas1)
    
    dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m, sensor_m = np.concatenate((data, data_1, data_2)).copy().T
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    
    # Let's plot NIR and SWIR2 time series, which are the best two disturbance indicator bands
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=3, sccd_result=sccd_result, axe=axes[0], title="NRT S-CCD (Week 3-4)")
    
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=5, sccd_result=sccd_result, axe=axes[1], title="NRT S-CCD (Week 3-4)")
    
    print(f"The current nrt mode is {sccd_result.nrt_mode}")
    print(f"The current number of consecutive anamlies is {sccd_result.nrt_model['anomaly_conse']}")
    print(f"The observation number in the current segment is {sccd_result.nrt_model['num_obs']}")


.. code:: text
    :class: output-block

    The current nrt mode is 5
    The current number of consecutive anamlies is 6
    The observation number in the current segment is 200
    


.. image:: 7_near_realtime_logging_hls_files/7_near_realtime_logging_hls_8_1.png


异常数量继续从 3 增加到 6，触发了一次断点检测。nrt 模式从 ``1`` 变为 ``5``。这里的 ``5`` 表示从“监测”模式过渡到“队列”模式，在此模式下，我们将同时保留 ``nrt_model`` 和 ``nrt_queue`` 15 天。因此，用户仍然可以从 ``nrt_model`` 进行断点分析。你可以检查（断点之后的）队列观测值：

.. code:: ipython3

    print(f"The observation queue is {sccd_result.nrt_queue}")


.. code:: text
    :class: output-block

    The observation queue is [([ 594,  832,  939, 3128, 3114, 1850], 15242)
     ([ 629,  831,  924, 3027, 2836, 1637], 15243)
     ([ 582,  790,  903, 2665, 2805, 1682], 15247)
     ([ 538,  810, 1054, 2996, 3046, 1787], 15257)
     ([ 747, 1026, 1270, 3644, 3740, 2197], 15259)
     ([ 695,  948, 1261, 3026, 3333, 2099], 15262)]
    

让我们快速检查断点类型（1-干扰；2-恢复）

.. code:: ipython3

    from pyxccd.utils import getcategory_sccd
    
    print(f"The break category (1-disturbance; 2-recovery) is {getcategory_sccd(sccd_result.rec_cg, 1)}")
    print(f"The recent disturbance date is {pd.Timestamp.fromordinal(sccd_result.rec_cg[-1]['t_break'])}")


.. code:: text
    :class: output-block

    The break category (1-disturbance; 2-recovery) is 1
    The recent disturbance date is 2024-04-08 00:00:00
    

最后，我们绘制时间序列的 Landsat 影像和 4 月 15 日的 Planet 影像来确认干扰的发生：

|image1|

|image2|

.. |image1| image:: image1.png
.. |image2| image:: image2.png

第 5-6 周：无清晰观测值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    in_path = TUTORIAL_DATASET/ '7_logging_hls_w56.csv'
    
    # read example csv for HLS time series
    data_3 = pd.read_csv(in_path)
    
    # split the array by the column
    dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, thermals1, qas1, sensor1 = data_3.to_numpy().copy().T
    sccd_result = sccd_update(sccd_result, dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, qas1)
    
    dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m, sensor_m = np.concatenate((data, data_1, data_2, data_3)).copy().T
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    
    # Let's plot NIR and SWIR2 time series, which are the best two disturbance indicator bands
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=3, sccd_result=sccd_result, axe=axes[0], title="NRT S-CCD (Week 5-6)")
    
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=5, sccd_result=sccd_result, axe=axes[1], title="NRT S-CCD (Week 5-6)")
    
    print(f"The current nrt mode is {sccd_result.nrt_mode}")
    print(f"The current number of consecutive anamlies is {sccd_result.nrt_model['anomaly_conse']}")
    print(f"The number of observation in the queue: {len(sccd_result.nrt_queue)}")
    print(f"The observation queue is {sccd_result.nrt_queue}")


.. code:: text
    :class: output-block

    The current nrt mode is 5
    The current number of consecutive anamlies is 6
    The number of observation in the queue: 6
    The observation queue is [([ 594,  832,  939, 3128, 3114, 1850], 15242)
     ([ 629,  831,  924, 3027, 2836, 1637], 15243)
     ([ 582,  790,  903, 2665, 2805, 1682], 15247)
     ([ 538,  810, 1054, 2996, 3046, 1787], 15257)
     ([ 747, 1026, 1270, 3644, 3740, 2197], 15259)
     ([ 695,  948, 1261, 3026, 3333, 2099], 15262)]
    


.. image:: 7_near_realtime_logging_hls_files/7_near_realtime_logging_hls_15_1.png


对于第 5–6 周，没有可用的清晰观测值，因此 ``nrt_model`` 和 ``nrt_queue`` 都没有更新。在 NRT 应用中，当未获取到有效观测值时，偶尔会出现这种情况。在这种情况下，监测暂时停止；然而，最近的断点信息仍然可以从 ``sccd_result`` 中检索。

第 7-8 周：断点后连续收集观测值（6->8）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    
    in_path = TUTORIAL_DATASET/ '7_logging_hls_w78.csv'
    
    # read example csv for HLS time series
    data_4 = pd.read_csv(in_path)
    
    # split the array by the column
    dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, thermals1, qas1, sensor1 = data_4.to_numpy().copy().T
    sccd_result = sccd_update(sccd_result, dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, qas1)
    
    dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m, sensor_m = np.concatenate((data, data_1, data_2, data_3, data_4)).copy().T
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    
    # Let's plot NIR and SWIR2 time series, which are the best two disturbance indicator bands
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=3, sccd_result=sccd_result, axe=axes[0], title="NRT S-CCD (Week 7-8)")
    
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=5, sccd_result=sccd_result, axe=axes[1], title="NRT S-CCD (Week 7-8)")
    
    print(f"The current nrt mode is {sccd_result.nrt_mode}")
    print(f"The current number of consecutive anamlies is {sccd_result.nrt_model['anomaly_conse']}")
    print(f"The number of observation in the queue: {len(sccd_result.nrt_queue)}")


.. code:: text
    :class: output-block

    The current nrt mode is 5
    The current number of consecutive anamlies is 6
    The number of observation in the queue: 8
    


.. image:: 7_near_realtime_logging_hls_files/7_near_realtime_logging_hls_18_1.png


第 9-12 周：不规则间隔监测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``S-CCD`` 允许你在 NRT 场景中输入任意长度的新观测值。例如，如果监测暂停了两周，你可以通过一次运行四周的观测值来恢复监测。

在本例中，请注意 ``nrt_mode`` 从 ``5`` 变为 ``12``。值 ``12`` 表示队列模式（具有不可预测性）。从此时起，``S-CCD`` 开始收集新的观测值，直到再次满足 CCDC 初始化条件。在此阶段，``nrt_queue`` 存储传入的观测值，而 nrt_model 被设置为 None。

过渡模式（``nrt_mode = 5``）仅在切换到队列模式之前保留 15 天。

.. code:: ipython3

    in_path = TUTORIAL_DATASET/ '7_logging_hls_w910.csv'
    
    # read example csv for HLS time series
    data_5 = pd.read_csv(in_path)
    
    # split the array by the column
    dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, thermals1, qas1, sensor1 = data_5.to_numpy().copy().T
    sccd_result = sccd_update(sccd_result, dates1, blues1, greens1, reds1, nirs1, swir1s1, swir2s1, qas1)
    
    dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m, sensor_m = np.concatenate((data, data_1, data_2, data_3, data_4, data_5)).copy().T
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    
    # Let's plot NIR and SWIR2 time series, which are the best two disturbance indicator bands
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=3, sccd_result=sccd_result, axe=axes[0], title="NRT S-CCD (Week 9-12)")
    
    display_sccd_result(data=np.stack((dates_m, blues_m, greens_m, reds_m, nirs_m, swir1s_m, swir2s_m, thermals_m, qas_m), axis=1), band_names=['blues', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermals'], band_index=5, sccd_result=sccd_result, axe=axes[1], title="NRT S-CCD (Week 9-12)")
    
    
    print(f"The current nrt mode is {sccd_result.nrt_mode}")
    print(f"nrt_model is {sccd_result.nrt_model}")
    print(f"The number of observation in the queue: {len(sccd_result.nrt_queue)}")
    


.. code:: text
    :class: output-block

    The current nrt mode is 12
    nrt_model is []
    The number of observation in the queue: 9
    


.. image:: 7_near_realtime_logging_hls_files/7_near_realtime_logging_hls_20_1.png


总结
-------

从上述案例中，我演示了 S-CCD 如何以递归方式检测断点。在实践中，用户可以将 ``sccd_result`` 本地保存，而不是存储完整的时间序列影像，这可以将数据存储需求减少约 90%，从而实现大范围处理。

值得注意的是，尽管第 3–4 周的断点被六个连续异常所确认，但 ``S-CCD`` 早在第 1–2 周就已经开始输出异常。更先进的方法是应用机器学习技术从 nrt_model 中提取特征，这样可以在少于六个连续异常的情况下实现更早的干扰检测。详情请参阅 [2] 中提出的机器学习框架。

*[2] Ye, S., Zhu, Z., & Suh, J. W. (2024). Leveraging past information
and machine learning to accelerate land disturbance monitoring. Remote
Sensing of Environment, 305, 114071.*