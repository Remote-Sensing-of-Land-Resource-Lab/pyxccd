第3课：灵活模式
=======================

**作者: Su Ye (remotesensingsuy@gmail.com)**

**时间序列数据集: Sentinel-2**

**应用: 中国河南的作物物候动态**

标准的 CCDC 方法只支持七个 Landsat 波段作为输入。``pyxccd`` 为 COLD（``cold_detect_flex``）和 S-CCD（``sccd_detect_flex``）提供了一个“灵活模式”，允许输入任意组合的波段、指数或多传感器时间序列。

--------------

来自非 Landsat 传感器的输入
--------------------------------------

以监测作物动态为例，我们使用 Sentinel-2 作为输入。我们将从为 ``cold_detect_flex`` 和 ``sccd_detect_flex`` 输入所有 Sentinel-2 波段开始：

.. code:: ipython3

    import numpy as np
    import os
    import pathlib
    import pandas as pd
    from dateutil import parser
    # Imports from this package
    from typing import List, Tuple, Dict, Union, Optional
    from matplotlib.axes import Axes
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pyxccd import sccd_detect_flex, cold_detect_flex
    from pyxccd.common import SccdOutput, cold_rec_cg
    from pyxccd.utils import getcategory_sccd, defaults, getcategory_cold
    
    def display_cold_result(
        data: np.ndarray,
        band_names: List[str],
        band_index: int,
        cold_result: cold_rec_cg,
        axe: Axes,
        title: str = 'COLD',
        plot_kwargs: Optional[Dict] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Compare COLD and SCCD change detection algorithms by plotting their results side by side.
        
        This function takes time series remote sensing data, applies both COLD algorithms,
        and visualizes the curve fitting and break detection results. 
        
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
        
        axe: Axes
            An Axes object represents a single plot within that Figure
        
        title: Str
            The figure title. The default is "COLD"
            
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
        band_values = data_clean[data_clean['qa'] == 0][band_name]
        q01, q99 = np.quantile(band_values, [0.01, 0.99])
        extra = (q99 - q01) * 0.4
        ylim_low = q01 - extra
        ylim_high = q99 + extra
    
        # Plot COLD observations
        axe.plot(
            'dates_formal', band_name, 'go',
            markersize=default_plot_kwargs['marker_size'],
            alpha=default_plot_kwargs['marker_alpha'],
            data=data_clean
        )
    
        # Plot COLD segments
        for segment in cold_result:
            j = np.arange(segment['t_start'], segment['t_end'] + 1, 1)
            plot_df = pd.DataFrame({
                'dates': j,
                'trend': j * segment['coefs'][band_index][1] / slope_scale + segment['coefs'][band_index][0],
                'annual': np.cos(w * j) * segment['coefs'][band_index][2] + np.sin(w * j) * segment['coefs'][band_index][3],
                'semiannual': np.cos(2 * w * j) * segment['coefs'][band_index][4] + np.sin(2 * w * j) * segment['coefs'][band_index][5],
                'trimodel': np.cos(3 * w * j) * segment['coefs'][band_index][6] + np.sin(3 * w * j) * segment['coefs'][band_index ][7]
            })
            plot_df['predicted'] = (
                plot_df['trend'] + 
                plot_df['annual'] + 
                plot_df['semiannual'] + 
                plot_df['trimodel']
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
    
        # Plot breaks
        for i in range(len(cold_result)):
            if  cold_result[i]['change_prob'] == 100:
                if getcategory_cold(cold_result, i) == 1:
                    axe.axvline(pd.Timestamp.fromordinal(cold_result[i]['t_break']), color='k')
                else:
                    axe.axvline(pd.Timestamp.fromordinal(cold_result[i]['t_break']), color='r')
        
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
    
        # Plot breaks
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
    
    in_path = TUTORIAL_DATASET/ '3_crop_sentinel2.csv' # read the MPB-affected plot in CO
    
    # read example csv for HLS time series
    data = pd.read_csv(in_path)
    
    # split the array by the column
    Tile, B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, QA60 = data.to_numpy().copy().T
    
    dates = np.array([parser.parse(tilename[0:8]).toordinal() for tilename in Tile])
    qas = QA60.copy()
    
    # Bit 10: Opaque clouds; Bit 11: Cirrus clouds
    qas[qas>0] = 4
    
    # for the flexible mode, we need to stack the chosen inputted bands into one array.
    # Let's choose the bands with the resolution smaller than 60 meters. 
    # B4 is green band, B12 is SWIR1, so they were chosen for tmask  (tmask_b1_index=3, tmask_b2_index=9).
    sccd_result = sccd_detect_flex(dates.astype(np.int32), np.stack((B2, B3, B4, B5, B6, B7, B8, B8A,  B11, B12), axis=1).astype(np.int32), qas.astype(np.int32), lam=20,  tmask_b1_index=3, tmask_b2_index=9)
    cold_result = cold_detect_flex(dates.astype(np.int32), np.stack((B2, B3, B4, B5, B6, B7, B8, B8A,  B11, B12), axis=1).astype(np.int32), qas.astype(np.int32), lam=20,  tmask_b1_index=3, tmask_b2_index=9)
    
                
    # Set up plotting style
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    display_cold_result(data=np.stack((dates, B2, B3, B4, B5, B6, B7, B8, B8A,  B11, B12, qas), axis=1).astype(np.int64), band_names=['blues', 'green', 'red', 'edge1', 'edge2','edge3', 'nir', 'edge4', 'swir1', 'swir2'], band_index=6, cold_result=cold_result, axe=axes[0], title="COLD (10 bands)")
    
    display_sccd_result(data=np.stack((dates, B2, B3, B4, B5, B6, B7, B8, B8A,  B11, B12, qas), axis=1).astype(np.int64), band_names=['blues', 'green', 'red', 'edge1', 'edge2','edge3', 'nir', 'edge4', 'swir1', 'swir2'], band_index=6, sccd_result=sccd_result, axe=axes[1], title="S-CCD (10 bands)")



.. image:: 3_flexible_inputs_crop_sentinel2_files/3_flexible_inputs_crop_sentinel2_1_0.png


第二个断点之后，S-CCD 尚未建立监测模型（即 ``nrt_mode`` 为 2）。因此，在这种情况下，2024 年之后没有显示拟合曲线。

与标准的 cold_detect 和 sccd_detect 相比，灵活版本（``cold_detect_flex`` 和 ``sccd_detect_flex``）要求用户明确指定 lambda 和 Tmask 波段索引：

(1) *Lambda*：用户必须提供一个 ``lambda`` 值，因为不同传感器可能具有不同的反射率值范围。如果输入数据缩放到 [0, 10000]，我们建议使用默认的 CCDC 设置（``lambda=20``）。通常，lambda 应根据实际输入范围相对于默认 Landsat 范围（10,000）进行缩放。例如，如果输入范围为 [0, 20000]，则新的 lambda 将为 20 * 20000 / 10000 = 40。

(2) *Tmask 波段索引*：在标准函数中，Tmask 硬编码为使用绿波段和短波红外1波段（即对于 Landsat，``tmask_b1_index = 2``，``tmask_b2_index = 5``）。这些波段用于从时间序列中过滤掉受云或噪声影响的观测值。在灵活模式下，波段索引作为用户定义参数公开，因为 Sentinel-2 等传感器或替代预处理数据集可能使用不同的波段顺序。

对于此案例，COLD 和 S-CCD 都检测到两个与作物动态相关的断点。2021 年的第一个断点与 **农田在 2021 年秋季休耕（特征是近红外值低）** 有关。2024 年的第二个断点可能与 **作物类型或品种变化有关，因为第二个生长季的近红外值明显高于正常值**。

三年际 S-CCD
--------------

在上面的例子中，与 COLD 相比，S-CCD 显示出较弱的拟合度，由于均方根误差（RMSE）增加，未能检测到 COLD 检测到的第二个断点。这里的主要原因是该农田 **每年有两个生长周期**，而标准的 S-CCD 仅用两个分量（年际和半年际）模拟季节性，优先考虑计算效率和最小存储。相比之下，COLD 包含三个分量（年际、半年际和四月期周期），这使得在有两个生长季节的情况下能够进行更准确的拟合。为了解决这个问题，S-CCD 的灵活模式提供了一个 ``trimodal`` 选项，**允许用户包含四月期周期，并为农田像元实现更好的曲线拟合**。

.. code:: ipython3

    # enable trimodal
    sccd_result = sccd_detect_flex(dates.astype(np.int32), np.stack((B2, B3, B4, B5, B6, B7, B8, B8A,  B11, B12), axis=1).astype(np.int32), qas.astype(np.int32), lam=20, tmask_b1_index=3, tmask_b2_index=9, trimodal=True)
    
                
    # Set up plotting style
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    display_cold_result(data=np.stack((dates, B2, B3, B4, B5, B6, B7, B8, B8A,  B11, B12, qas), axis=1).astype(np.int64), band_names=['blues', 'green', 'red', 'edge1', 'edge2','edge3', 'nir', 'edge4', 'swir1', 'swir2'], band_index=6, cold_result=cold_result, axe=axes[0], title="COLD (10 bands)")
    
    display_sccd_result(data=np.stack((dates, B2, B3, B4, B5, B6, B7, B8, B8A,  B11, B12, qas), axis=1).astype(np.int64), band_names=['blues', 'green', 'red', 'edge1', 'edge2','edge3', 'nir', 'edge4', 'swir1', 'swir2'], band_index=6, sccd_result=sccd_result, axe=axes[1], title="Trimodal S-CCD (10 bands)")



.. image:: 3_flexible_inputs_crop_sentinel2_files/3_flexible_inputs_crop_sentinel2_3_0.png


如上所示，S-CCD 和 COLD 生成的拟合曲线几乎相同，并且两种方法产生了相同的断点检测结果。

结合植被指数
-------------------------------

有时，仅依靠所有原始的 Sentinel-2 光谱波段作为输入并不能产生令人满意的断点检测性能。在下一节中，我们将探讨结合光谱波段和植被指数是否能改善结果。

输入植被指数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sentinel-2 MSI 覆盖 13 个光谱波段，表示如下：

======================== ================== ==============
Sentinel-2 波段          中心波长 (µm)       分辨率 (米)
======================== ================== ==============
Band 1 - Coastal aerosol 0.443              60
Band 2 - Blue            0.490              10
Band 3 - Green           0.560              10
Band 4 - Red             0.665              10
Band 5 - Red Edge        0.705              20
Band 6 - Red Edge        0.740              20
Band 7 - Red Edge        0.783              20
Band 8 - NIR             0.842              10
Band 8A - Red Edge       0.865              20
Band 9 - Water vapour    0.945              60
Band 10 - SWIR-Cirrus    1.375              60
Band 11 - SWIR           1.610              20
Band 12 - SWIR           2.190              20
======================== ================== ==============

对于农业监测，增强型植被指数（EVI）被广泛用于捕捉作物生长动态和植被生理状态。在本实验中，我们选择了五个原始的 Sentinel-2 光谱波段——绿、红、近红外、短波红外1 和短波红外2——与标准 CCDC 配置一致，并增加了一个 EVI 波段。然后，将这六个变量用作 ``cold_detect_flex`` 和 ``sccd_detect_flex`` 的输入特征：

.. code:: ipython3

    # scale EVI to [0, 10000]
    evi = 25000 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 10000)
    
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    
    # plot time series and detection results
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4)
    
    
    cold_result = cold_detect_flex(dates.astype(np.int32), np.stack((B3, B4, B8, B11, B12, evi), axis=1).astype(np.int32), qas.astype(np.int32), lam=20,  tmask_b1_index=1, tmask_b2_index=4)
    
    sccd_result = sccd_detect_flex(dates.astype(np.int32), np.stack((B3, B4, B8, B11, B12, evi), axis=1).astype(np.int32), qas.astype(np.int32), trimodal=True,  lam=20, tmask_b1_index=1, tmask_b2_index=4)
    
    display_cold_result(data=np.stack((dates, B3, B4, B8, B11, B12, evi, qas), axis=1).astype(np.int64), band_names=['green', 'red', 'nir',  'swir1', 'swir2', 'EVI'], band_index=5, cold_result=cold_result, axe=axes[0], title="COLD (5 bands+EVI)")
    
    display_sccd_result(data=np.stack((dates, B3, B4, B8, B11, B12, evi, qas), axis=1).astype(np.int64), band_names=['green', 'red', 'nir',  'swir1', 'swir2', 'EVI'], band_index=5, sccd_result=sccd_result, axe=axes[1],  title="Trimodel S-CCD (5 bands+EVI)")
    



.. image:: 3_flexible_inputs_crop_sentinel2_files/3_flexible_inputs_crop_sentinel2_5_0.png


总结
-------

虽然添加 EVI 并未改变此案例的断点检测结果，但我们通常建议将 EVI 或 NDVI 作为额外的输入纳入 COLD 或 S-CCD，以加强农田监测。除了提高断点检测的稳健性外，来自 EVI 或 NDVI 的拟合谐波模型（即八个谐波系数）对于表征种植强度、监测生长阶段以及作为产量估算的代理指标非常有价值。第 9 课（物候）将展示 EVI 谐波模型在作物物候监测中的应用。