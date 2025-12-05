第5课：状态分析
========================

**作者: Su Ye (remotesensingsuy@gmail.com)**

**时间序列数据集: MODIS & GPCP**

**应用: 绿化和降水季节性**

在状态空间理论中，状态是一个不可观测的（潜在的）变量，它随时间演变并支配系统的动态。我们收集的观测数据（例如，NDVI、反射率、降水）被假定为由这些隐藏状态生成，并附加噪声。形式上，基本的状态空间模型由两个方程定义：一个状态（转移）方程，描述潜在状态的时间演变；以及一个观测方程，将潜在状态与观测数据联系起来。两个方程都包含随机误差项，反映了演变过程和测量过程中的不确定性。这种随机基础是该框架被称为随机连续变化检测（S-CCD）的原因。

.. math:: a_{t}=Ta_{t-1} + Q

.. math:: y_{t}=Za_{t} + H

- :math:`a_{t}`: 时间 (t) 的隐藏状态
- :math:`T`: 转移矩阵（定义状态如何演变）
- :math:`Z`: 决定观测中包含项的系统矩阵
- :math:`y_{t}`: 观测值
- :math:`Q`: 过程噪声
- :math:`H`: 观测噪声

在 S-CCD [1] 中，状态表示为一个包含三个时间分量的向量：趋势、年际和半年际。当在灵活模式中指定 ``trimodal``\ =True 时，会额外包含一个对应四个月周期（``trimodal``）的分量。S-CCD 支持为这些分量中的每一个输出状态。与具有固定系数的谐波回归线不同，状态包含了观测和过程噪声，因此其时间变异性能够更好地捕捉局部波动。

本课展示了 ``states`` 在分析细微长期变化方面的一个新颖应用，这些变化往往被传统的断点检测方法所忽视。

*[1] Ye, S., Rogan, J., Zhu, Z., & Eastman, J. R. (2021). A
near-real-time approach for monitoring forest disturbance using Landsat
time series: Stochastic continuous change detection. Remote Sensing of
Environment, 252, 112167.*

--------------

绿化趋势分析
------------------------

.. code:: ipython3

    import pandas as pd
    import numpy as np
    import pathlib
    from dateutil import parser
    from pyxccd import sccd_detect_flex
    
    TUTORIAL_DATASET = (pathlib.Path.cwd() / 'datasets').resolve() # modify it as you need
    assert TUTORIAL_DATASET.exists()
    
    in_path = TUTORIAL_DATASET/ '5_greenning_modis.csv' # read single-pixel MODIS time series
    
    
    # read example csv for HLS time series
    data = pd.read_csv(in_path)
    
    # as the original data doesn't have qa, we append qa as all zeros value (meaning they are all clear)
    data['qa'] = np.zeros(data.shape[0])
    
    # force column name to 'date' to let display_sccd_states work
    data = data.rename(columns={'date': 'dates'})
    
    # convert them to ordinal dates
    ordinal_dates = [pd.Timestamp.toordinal(parser.parse(row)) for row in data["dates"]]
    data.loc[:, "dates"] = ordinal_dates
    
    dates, ndvi, qas = data.to_numpy().copy().T
    print(sccd_detect_flex(dates, ndvi, qas, lam=20, state_intervaldays=1))


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


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[1], line 27
         24 data.loc[:, "dates"] = ordinal_dates
         26 dates, ndvi, qas = data.to_numpy().copy().T
    ---> 27 print(sccd_detect_flex(dates, ndvi, qas, lam=20, state_intervaldays=1))
    

    File c:\Users\dell\env_collect\py311_geo\Lib\site-packages\pyxccd\ccd.py:929, in sccd_detect_flex(dates, ts_stack, qas, lam, p_cg, conse, pos, b_c2, output_anomaly, anomaly_pcg, anomaly_conse, state_intervaldays, tmask_b1_index, tmask_b2_index, fitting_coefs, trimodal)
        915 _validate_params(
        916     func_name="sccd_detect_flex",
        917     p_cg=p_cg,
       (...)
        926     trimodal=trimodal,
        927 )
        928 # make sure it is c contiguous array and 64 bit
    --> 929 dates, ts_stack, qas = _validate_data_flex(dates, ts_stack, qas)
        930 valid_num_scenes = ts_stack.shape[0]
        931 nbands = ts_stack.shape[1] if ts_stack.ndim > 1 else 1
    

    File c:\Users\dell\env_collect\py311_geo\Lib\site-packages\pyxccd\ccd.py:138, in _validate_data_flex(dates, ts_data, qas)
        126 """
        127 validate and forcibly change the data format
        128 Parameters
       (...)
        135 ----------
        136 """
        137 check_consistent_length(dates, ts_data, qas)
    --> 138 check_1d(dates, "dates")
        139 check_1d(qas, "qas")
        141 dates = dates.astype(dtype=numpy.int64, order="C")
    

    File c:\Users\dell\env_collect\py311_geo\Lib\site-packages\pyxccd\_param_validation.py:663, in check_1d(array, var_name)
        654     raise ValueError(
        655         "Expected 1D array for input {}, but got {}D".format(var_name, array.ndim)
        656     )
        658 if (
        659     (array.dtype != "int64")
        660     and (array.dtype != "int32")
        661     and (array.dtype != "int16")
        662 ):
    --> 663     raise ValueError(
        664         "Expected int16, int32, int64 for the input, but got {}".format(array.dtype)
        665     )
    

    ValueError: Expected int16, int32, int64 for the input, but got object


哎呀，我们遇到了错误“Expected int16, int32, int64 for the input, but got object”。这是因为 NDVI 的数据类型是双精度浮点数 [-1, 1]。我们需要将其转换为整数。**Pyxccd 只支持整数输入！**

缩放确实会影响 CCDC 的结果，因为 Lasso 参数对输入值的大小敏感，这直接影响拟合精度和正则化强度之间的平衡。为了确保最佳性能和可比性，我们建议缩放浮点反射率值，使其落在或接近标准 Landsat 反射率范围 [0,10000] 内。这种调整使输入数据与 CCDC 通常校准的尺度相协调，并有助于防止模型拟合中的偏差。在这种情况下，将反射率值乘以 10,000 提供了一个合适的变换：

.. code:: ipython3

    import pathlib
    from datetime import date
    from typing import List, Tuple, Dict, Union, Optional
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from pyxccd.common import SccdOutput
    from pyxccd.utils import getcategory_sccd, defaults
    
    def display_sccd_states_flex(
        data_df: pd.DataFrame,
        states:pd.DataFrame,
        axes: Axes,
        variable_name: str,
        title:str,
        band_name:str = "b0",
        plot_kwargs: Optional[Dict] = None
    ):
        default_plot_kwargs: Dict[str, Union[int, float, str]] = {
            'marker_size': 5,
            'marker_alpha': 0.7,
            'line_color': 'orange',
            'font_size': 14
        }
        if plot_kwargs is not None:
            default_plot_kwargs.update(plot_kwargs)
    
        # convert ordinal dates to calendar
        formal_dates = [pd.Timestamp.fromordinal(int(row)) for row in states["dates"]]
        states.loc[:, "dates_formal"] = formal_dates  
    
        extra = (np.max(states[f"{band_name}_trend"]) - np.min(states[f"{band_name}_trend"])) / 4
        axes[0].set(ylim=(np.min(states[f"{band_name}_trend"]) - extra, np.max(states[f"{band_name}_trend"]) + extra))
        sns.lineplot(x="dates_formal", y=f"{band_name}_trend", data=states, ax=axes[0], color="orange")
        axes[0].set(ylabel=f"Trend")
    
        extra = (np.max(states[f"{band_name}_annual"]) - np.min(states[f"{band_name}_annual"])) / 4
        axes[1].set(ylim=(np.min(states[f"{band_name}_annual"]) - extra, np.max(states[f"{band_name}_annual"]) + extra))
        sns.lineplot(x="dates_formal", y=f"{band_name}_annual", data=states, ax=axes[1], color="orange")
        axes[1].set(ylabel=f"Annual cycle")
    
        extra = (np.max(states[f"{band_name}_semiannual"]) - np.min(states[f"{band_name}_semiannual"])) / 4
        axes[2].set(ylim=(np.min(states[f"{band_name}_semiannual"]) - extra, np.max(states[f"{band_name}_semiannual"]) + extra))
        sns.lineplot(x="dates_formal", y=f"{band_name}_semiannual", data=states, ax=axes[2], color="orange")
        axes[2].set(ylabel=f"Semi-annual cycle")
    
    
        data_clean = data_df[(data_df["qa"] == 0) | (data_df['qa'] == 1)].copy() # CCDC also processes water pixels
        formal_dates = [pd.Timestamp.fromordinal(int(row)) for row in data_clean["dates"]]
        data_clean.loc[:, "dates_formal"] = formal_dates  # convert ordinal dates to calendar
        axes[3].plot(
            'dates_formal', variable_name, 'go',
            markersize=default_plot_kwargs['marker_size'],
            alpha=default_plot_kwargs['marker_alpha'],
            data=data_clean
        )
    
        if f"{band_name}_trimodal" in list(states.columns):
            states["General"] = states[f"{band_name}_annual"] + states[f"{band_name}_trend"] + states[f"{band_name}_semiannual"]+ states[f"{band_name}_trimodal"]
        else:
            states["General"] = states[f"{band_name}_annual"] + states[f"{band_name}_trend"] + states[f"{band_name}_semiannual"]
        g = sns.lineplot(
            x="dates_formal", y="General", data=states, label="fit", ax=axes[3], color="orange"
        )
                
        axes[3].set_ylabel(f"{variable_name}", fontsize=default_plot_kwargs['font_size'])
        axes[3].set_title(title, fontweight="bold", size=16 , pad=2)
        
        band_values = data_df[data_df['qa'] == 0][variable_name]
        q01, q99 = np.quantile(band_values, [0.01, 0.99])
        extra = (q99 - q01) * 0.4
        ylim_low = q01 - extra
        ylim_high = q99 + extra
        axes[3].set(ylim=(ylim_low, ylim_high))
    
    
    
    data_copy = data.copy()
    
    # Multiplying NDVI by 10000
    data_copy.NDVI = data_copy.NDVI.multiply(10000)
    dates, ndvi,  qas = data_copy.to_numpy().astype(np.int64).copy().T
    
    # Turn on the state output by setting state_intervaldays as non-zero value
    sccd_results, states = sccd_detect_flex(dates, ndvi, qas, lam=20, state_intervaldays=1)
    
    # Set up plotting style
    sns.set(style="darkgrid")
    sns.set_context("notebook")
    
    # Create figure and axes
    fig, axes = plt.subplots(4, 1, figsize=[12, 10], sharex=True)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)
    
    display_sccd_states_flex(data_df=data_copy,  states=states, axes=axes, variable_name="NDVI", title="S-CCD")
    



.. image:: 5_state_analysis_greenning&precipitation_coarse_files/5_state_analysis_greenning&precipitation_coarse_3_0.png


多项研究表明，我们的地球经历了植被绿化速度的减缓 [2, 3]。我们也从 ``Trend`` 分量中直观地确认了相同的减缓趋势。

*[2] Feng, X., Fu, B., Zhang, Y., Pan, N., Zeng, Z., Tian, H., … &
Penuelas, J. (2021). Recent leveling off of vegetation greenness and
primary production reveals the increasing soil water limitations on the
greening Earth. Science Bulletin, 66(14), 1462-1471.*

*[3] Ren, Y., Wang, H., Yang, K., Li, W., Hu, Z., Ma, Y., & Qiao, S.
(2024). Vegetation productivity slowdown on the Tibetan Plateau around
the late 1990s. Geophysical Research Letters, 51(4), e2023GL103865.*

寻找转折点
~~~~~~~~~~~~~~~~~~~~~~~~~

我将使用 ``kneed`` 包来自动定位这个 MODIS 像元绿化趋势开始减缓的时间点。``kneed`` 包基于曲线的最大曲率找到曲线的“拐点”，即曲线从陡峭斜率变为平坦斜率的地方。

.. code:: ipython3

    from kneed import KneeLocator
    
    # find the minimum NDVI trend value
    istart = np.where(states['b0_trend'].to_numpy()==np.min(states['b0_trend'].to_numpy()))[0][0]
    
    # Detecting the knee between the minimum NDVI to the maximum
    knee = KneeLocator(states['dates'].to_numpy()[istart:], states['b0_trend'].to_numpy()[istart:],  direction="increasing")
    xknee = states['dates'].to_numpy()[istart:][np.argmax(knee.y_difference)]
    yknee = states['b0_trend'].to_numpy()[istart:][np.argmax(knee.y_difference)]
    
    sns.set(style="darkgrid")
    sns.set_context("notebook")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    
    formal_dates = [pd.Timestamp.fromordinal(int(row)) for row in states["dates"]]
    states.loc[:, "dates_formal"] = formal_dates  
    sns.lineplot(data=states, ax=ax, x='dates_formal', y='b0_trend', legend=False, color='orange', palette='colorblind').set_title("The turning point of greening")
    
    ax.scatter(pd.Timestamp.fromordinal(int(xknee)), yknee , s=80, marker='o', facecolors='none', edgecolors='#1f77b4')
    
    print(f"The knee date (i.e., when NDVI saturates) is {pd.Timestamp.fromordinal(int(xknee))}")
    print(f"The knee NDVI  is {yknee/10000}")


.. code:: text
    :class: output-block

    The knee date (i.e., when NDVI saturates) is 2005-03-06 00:00:00
    The knee NDVI  is 0.8471548930431367
    


.. image:: 5_state_analysis_greenning&precipitation_coarse_files/5_state_analysis_greenning&precipitation_coarse_5_1.png


总结 1
~~~~~~~~~

S-CCD 提供了一个创新的框架，用于**将基于卫星的时间序列分解为多个可解释的分量**，同时明确考虑了时间断点。通过将信号分离为趋势和季节性分量，S-CCD 能够更清晰地理解各种干扰和恢复机制下的生态系统动态。提取的趋势分量尤其具有信息性，因为它可以揭示植被或生物物理条件的细微或渐进变化，这些变化常常被强烈的季节性波动或短期噪声所掩盖。这种长期趋势变化通常源于外部驱动因素，如气候变率、人类活动或自然干扰。通过有效抑制季节性和不规则变化，S-CCD 增强了这些潜在变化的可检测性，为生态系统随时间演变的轨迹提供了更稳健和可解释的视图。

降水季节性
-------------------------

年降水周期的季节性幅度是指季节性对比的强度。据报道，随着全球变暖，季节性幅度正在放大 [4]，表明最湿期和最干期之间的差异增大。但这种趋势可能是区域性的，需要仔细研究。

*[4] Wang, X., Luo, M., Song, F., Wu, S., Chen, Y. D., & Zhang, W.
(2024). Precipitation seasonality amplifies as earth warms. Geophysical
Research Letters, 51(10), e2024GL109132.*

下面是一个示例，我们将评估北极地区一个像元的这种趋势。让我们首先快速绘制时间序列：

.. code:: ipython3

    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    
    # Create figure and axes
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)
    
    TUTORIAL_DATASET = (pathlib.Path.cwd() / 'datasets').resolve() # modify it as you need
    assert TUTORIAL_DATASET.exists()
    in_path = TUTORIAL_DATASET/ '5_precip_gpcp.csv' # read the MPB-affected plot in CO
    
    # read example csv for HLS time series
    data = pd.read_csv(in_path)
    
    # as the original data doesn't have qa, we append qa as all zeros value (meaning they are all clear)
    data['qa'] = np.zeros(data.shape[0])
    
    formal_dates = [parser.parse(row) for row in data["time"]]
    data.loc[:, "dates_formal"] = formal_dates  
    
    # convert formal to ordinaldates
    ordinal_dates = [pd.Timestamp.toordinal(parser.parse(row)) for row in data["time"]]
    data.loc[:, "time"] = ordinal_dates
    
    fig, axes = plt.subplots(figsize=[12, 4])
    axes.plot(
        'dates_formal', "precip", 'go',
        data=data
    )
    plt.ylabel("Precip")
    plt.xlabel("Date")
    




.. code:: text
    :class: output-block

    Text(0.5, 0, 'Date')




.. code:: text
    :class: output-block

    <Figure size 640x480 with 0 Axes>



.. image:: 5_state_analysis_greenning&precipitation_coarse_files/5_state_analysis_greenning&precipitation_coarse_7_2.png


从上面的示例中，我们确实看到自2019年左右以来降水季节性的幅度增加了。在北极大部分高纬度地区，降水在夏季达到峰值（主要是降雨，由于开阔水域和更活跃的水汽输送）。变暖的海洋和减少的海冰增加了大气中的水汽通量，增强了夏季降水（尤其是降雨），从而增加了季节性的幅度。让我们使用 S-CCD 状态来研究这种增加的幅度。

.. code:: ipython3

    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    
    # Create figure and axes
    fig, axes = plt.subplots(4, 1, figsize=[12, 10], sharex=True)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)
    
    # read example csv for HLS time series
    TUTORIAL_DATASET = (pathlib.Path.cwd() / 'datasets').resolve() # modify it as you need
    assert TUTORIAL_DATASET.exists()
    in_path = TUTORIAL_DATASET/ '5_precip_gpcp.csv' # read the MPB-affected plot in CO
    data = pd.read_csv(in_path)
    
    # as the original data doesn't have qa, we append qa as all zeros value (meaning they are all clear)
    data['qa'] = np.zeros(data.shape[0])
    
    formal_dates = [parser.parse(row) for row in data["time"]]
    data.loc[:, "dates_formal"] = formal_dates  
    
    # convert formal to ordinaldates
    ordinal_dates = [pd.Timestamp.toordinal(parser.parse(row)) for row in data["time"]]
    data.loc[:, "time"] = ordinal_dates
    
    # scale precipitation by 1000
    data.loc[:, "precip"] = data.loc[:, "precip"].apply(lambda x: x*1000)
    
    # force column name to 'date' to let display_sccd_states work
    data = data.rename(columns={'time': 'dates'})
    
    dates, prep, qas = data.drop('dates_formal', axis=1).astype(np.int32).to_numpy().copy().T
    sccd_result, states = sccd_detect_flex(dates, prep, qas, lam=20, state_intervaldays=1)
    display_sccd_states_flex(data_df=data, states=states, axes=axes, variable_name="precip", title="precipitation * 1000")



.. image:: 5_state_analysis_greenning&precipitation_coarse_files/5_state_analysis_greenning&precipitation_coarse_9_0.png


不幸的是，我们只观察到在时间序列末尾半年际周期幅度有微弱的增加。进一步的研究可以揭示，这种模式是由于对极端夏季降水事件的拟合不足导致的，这导致未能捕捉到增强的季节性信号。为了改进 S-CCD 对该数据集的拟合，我们将 Lambda 降低到 0：

.. code:: ipython3

    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    
    # Create figure and axes
    fig, axes = plt.subplots(4, 1, figsize=[12, 10], sharex=True)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)
    
    # decrease lambda to 0
    sccd_result, states = sccd_detect_flex(dates, prep, qas, lam=0, state_intervaldays=1)
    display_sccd_states_flex(data_df=data, states=states, axes=axes, variable_name="precip", title="precipitation * 1000 (Lam=0)")



.. image:: 5_state_analysis_greenning&precipitation_coarse_files/5_state_analysis_greenning&precipitation_coarse_11_0.png


现在，在达到更好的模型拟合后，你可以看到增加的季节性幅度已分别在年际（第二个子图）和半年际（第三个子图）周期分量中被捕捉到，这是通过提高曲线对观测数据的拟合度实现的。

我们可以将年际和半年际分量相加，形成一个新分量来表示一般的季节性周期，以便更好地研究：

.. code:: ipython3

    def display_sccd_states_flex_adjusted(
        data_df: pd.DataFrame,
        states:pd.DataFrame,
        axes: Axes,
        variable_name: str,
        title:str,
        band_name:str = "b0",
        plot_kwargs: Optional[Dict] = None
    ):
        default_plot_kwargs: Dict[str, Union[int, float, str]] = {
            'marker_size': 5,
            'marker_alpha': 0.7,
            'line_color': 'orange',
            'font_size': 14
        }
        if plot_kwargs is not None:
            default_plot_kwargs.update(plot_kwargs)
    
        # convert ordinal dates to calendar
        formal_dates = [pd.Timestamp.fromordinal(int(row)) for row in states["dates"]]
        states.loc[:, "dates_formal"] = formal_dates  
    
        extra = (np.max(states[f"{band_name}_trend"]) - np.min(states[f"{band_name}_trend"])) / 4
        axes[0].set(ylim=(np.min(states[f"{band_name}_trend"]) - extra, np.max(states[f"{band_name}_trend"]) + extra))
        sns.lineplot(x="dates_formal", y=f"{band_name}_trend", data=states, ax=axes[0], color="orange")
        axes[0].set(ylabel=f"Trend")
    
        extra = (np.max(states[f"{band_name}_seasonality"]) - np.min(states[f"{band_name}_seasonality"])) / 4
        axes[1].set(ylim=(np.min(states[f"{band_name}_seasonality"]) - extra, np.max(states[f"{band_name}_seasonality"]) + extra))
        sns.lineplot(x="dates_formal", y=f"{band_name}_seasonality", data=states, ax=axes[1], color="orange")
        axes[1].set(ylabel=f"Seasonal cycle")
    
    
    
        data_clean = data_df[(data_df["qa"] == 0) | (data_df['qa'] == 1)].copy() # CCDC also processes water pixels
        formal_dates = [pd.Timestamp.fromordinal(int(row)) for row in data_clean["dates"]]
        data_clean.loc[:, "dates_formal"] = formal_dates  # convert ordinal dates to calendar
        axes[2].plot(
            'dates_formal', variable_name, 'go',
            markersize=default_plot_kwargs['marker_size'],
            alpha=default_plot_kwargs['marker_alpha'],
            data=data_clean
        )
    
        if f"{band_name}_trimodal" in list(states.columns):
            states["General"] = states[f"{band_name}_annual"] + states[f"{band_name}_trend"] + states[f"{band_name}_semiannual"]+ states[f"{band_name}_trimodal"]
        else:
            states["General"] = states[f"{band_name}_annual"] + states[f"{band_name}_trend"] + states[f"{band_name}_semiannual"]
        g = sns.lineplot(
            x="dates_formal", y="General", data=states, label="fit", ax=axes[2], color="orange"
        )
                
        axes[2].set_ylabel(f"{variable_name}", fontsize=default_plot_kwargs['font_size'])
        axes[2].set_title(title, fontweight="bold", size=16 , pad=2)
        
        band_values = data_df[data_df['qa'] == 0][variable_name]
        q01, q99 = np.quantile(band_values, [0.01, 0.99])
        extra = (q99 - q01) * 0.4
        ylim_low = q01 - extra
        ylim_high = q99 + extra
        axes[2].set(ylim=(ylim_low, ylim_high))
    
    
    fig, axes = plt.subplots(3, 1, figsize=[12, 8], sharex=True)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)
    sccd_result, states = sccd_detect_flex(dates, prep, qas, lam=0, state_intervaldays=1)
    
    # sum up two components
    states['b0_seasonality'] = states['b0_annual'] + states['b0_semiannual']
    
    display_sccd_states_flex_adjusted(data_df=data, states=states, axes=axes, variable_name="precip", title="precipitation * 1000 (Lam=0)")



.. image:: 5_state_analysis_greenning&precipitation_coarse_files/5_state_analysis_greenning&precipitation_coarse_13_0.png


总结 2
~~~~~~~~~

季节性强度的进一步量化可以通过计算上述步骤得出的“季节性周期”变量的最大值和最小值之间的差异来实现。与直接计算原始观测值的最大值和最小值之间的差异相比，这种方法提供了更稳定的年内变率估计。其关键优势在于 S-CCD 在拟合过程中包含了卡尔曼滤波器，它有效地平滑了时间序列，减少了噪声、异常值和不规则采样的影响。因此，估计的幅度能更好地反映潜在的季节性信号，而不是被异常的测量值所扭曲。

S-CCD 模型拟合
---------------

传统上，曲线拟合不考虑时间断点，这可能会降低模型性能。S-CCD 通过提供三种考虑断点的模型拟合方法来解决这一限制：

(1) 直接求和所有状态分量；
(2) 使用时间段内的所有观测值应用 LASSO 回归（``fitting_coefs=True``）；
(3) 采用卡尔曼滤波器在最后一个模型中估计的时间特定谐波模型系数（``fitting_coefs=False``）。

通常，求和所有状态（方法 1）是捕捉时间序列局部波动最有效的方法，因此能产生最低的均方根误差（RMSE）。应用 LASSO 回归（方法 2）对应于传统 CCDC 算法中使用的系数生成策略，并提供了最好的泛化能力，因为它只输出八个系数，可以作为机器学习输入的形状参数。使用时间特定的谐波模型系数（方法 3）仅反映最近观测值的模型行为，可能导致时间序列早期部分的拟合不足。这种方法特别适用于近实时监测应用，其首要重点是检测和表征最近的干扰。

以下是比较使用降水数据集的不同拟合方法的示例：

.. code:: ipython3

    from pyxccd.common import SccdOutput, cold_rec_cg, anomaly
    from matplotlib.lines import Line2D
    
    def display_sccd_result_single(
        data: np.ndarray,
        band_names: List[str],
        band_index: int,
        sccd_result: SccdOutput,
        axe: Axes,
        title: str = 'S-CCD',
        states:Optional[pd.DataFrame] = None,
        trimodal: bool = False,
        anomaly:Optional[anomaly] = None,
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
            
        trimodal: bool
            indicate whether using trimodal
        
        anomaly: anomaly, optional
            The anomaly detection outputs
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
        if trimodal:
            n_coefs = 8
        else:
            n_coefs = 6
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
        if states is not None:
            if trimodal is True:
                states['predicted'] = states['b0_trend']+states['b0_annual']+states['b0_semiannual']+states['b0_trimodal']
            else:
                states['predicted'] = states['b0_trend']+states['b0_annual']+states['b0_semiannual']
            calendar_dates = [pd.Timestamp.fromordinal(int(row)) for row in states["dates"]]
            states.loc[:, 'dates_formal'] = calendar_dates
            g = sns.lineplot(
                x="dates_formal", y="predicted",
                data=states,
                label="Model fit",
                ax=axe,
                color=default_plot_kwargs['line_color']
            )
            if g.legend_ is not None: 
                g.legend_.remove()
        else:
            for segment in sccd_result.rec_cg:
                j = np.arange(segment['t_start'], segment['t_break'] + 1, 1)
                if trimodal == True:
                    plot_df = pd.DataFrame(
                        {
                        'dates': j,
                        'trend': j * segment['coefs'][band_index][1] / slope_scale + segment['coefs'][band_index][0],
                        'annual': np.cos(w * j) * segment['coefs'][band_index][2] + np.sin(w * j) * segment['coefs'][band_index][3],
                        'semiannual': np.cos(2 * w * j) * segment['coefs'][band_index][4] + np.sin(2 * w * j) * segment['coefs'][band_index][5],
                        'trimodal': j * 0
                    })
    
                else:
                    plot_df = pd.DataFrame(
                        {
                        'dates': j,
                        'trend': j * segment['coefs'][band_index][1] / slope_scale + segment['coefs'][band_index][0],
                        'annual': np.cos(w * j) * segment['coefs'][band_index][2] + np.sin(w * j) * segment['coefs'][band_index][3],
                        'semiannual': np.cos(2 * w * j) * segment['coefs'][band_index][4] + np.sin(2 * w * j) * segment['coefs'][band_index][5],
                        'trimodal': np.cos(3 * w * j) * segment['coefs'][band_index][6] + np.sin(3 * w * j) * segment['coefs'][band_index][7]
                    })
                    
    
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
    
                if trimodal == True:
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
                
        # add manual legends
        if anomaly is not None:
            legend_elements = [Line2D([0], [0], label='Disturbance break', color='k'),
                                Line2D([0], [0], label='recovery break', color='r'),
                                Line2D([0], [0], marker='o', color="#EAEAF2",
                                markerfacecolor="#EAEAF2",markeredgecolor="black",
                                label='Disturbance anomalies', lw=0, markersize=8),
                                Line2D([0], [0], marker='o', color="#EAEAF2",
                                markerfacecolor="#EAEAF2",markeredgecolor="red",
                                label='Recovery anomalies', lw=0, markersize=8)]
        else:
            legend_elements = [Line2D([0], [0], label='Disturbance break', color='k'),
                        Line2D([0], [0], label='recovery break', color='r')]
        axe.legend(handles=legend_elements, loc='upper left', prop={'size': 9})
        
        # plot breaks
        for i in range(len(sccd_result.rec_cg)):
            # we used the sign of change magnitude to decide the category of the breaks
            if sccd_result.rec_cg[i]['magnitude'] < 0:
                axe.axvline(pd.Timestamp.fromordinal(sccd_result.rec_cg[i]['t_break']), color='k')
            else:
                axe.axvline(pd.Timestamp.fromordinal(sccd_result.rec_cg[i]['t_break']), color='r')
        
       
        
        # plot anomalies if available
        if anomaly is not None:
            for i in range(len(anomaly.rec_cg_anomaly)):
                pred_ref = np.asarray(
                        [
                            predict_ref(
                                anomaly.rec_cg_anomaly[i]["coefs"][0],
                                anomaly.rec_cg_anomaly[i]["obs_date_since1982"][i_conse]
                                + defaults['COMMON']['JULIAN_LANDSAT4_LAUNCH'], num_coefficients=n_coefs
                            ) for i_conse in range(3)
                        ]
                )
    
                cm = anomaly.rec_cg_anomaly[i]["obs"][0, 0: 3] - pred_ref
                
                # gpp increase is black line
                if np.median(cm) > 0:
                    yc = data[data[:,0] == anomaly.rec_cg_anomaly[i]['t_break']][0][1]
                    axe.plot(pd.Timestamp.fromordinal(anomaly.rec_cg_anomaly[i]['t_break']), yc,'ro',fillstyle='none',markersize=8)         
                # gpp decrease is red line
                else:
                    yc = data[data[:,0] == anomaly.rec_cg_anomaly[i]['t_break']][0][1]
                    axe.plot(pd.Timestamp.fromordinal(anomaly.rec_cg_anomaly[i]['t_break']), yc,'ko',fillstyle='none',markersize=8) 
            
        axe.set_ylabel(f"{band_name}", fontsize=default_plot_kwargs['font_size'])
    
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
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    plt.subplots_adjust(hspace=0.4)
    
    # fitting using the state output
    sccd_result1, states = sccd_detect_flex(dates, prep, qas, lam=0, state_intervaldays=1)
    display_sccd_result_single(data=data[['dates', 'precip', 'qa']].to_numpy().astype(np.int64), band_names=['precip'], band_index=0, sccd_result=sccd_result1, axe=axes[0], trimodal=False, states=states, title=f"Fitting using states")
    
    # fitting using lasso regression
    sccd_result2= sccd_detect_flex(dates, prep, qas, lam=0, fitting_coefs=True)
    display_sccd_result_single(data=data[['dates', 'precip', 'qa']].to_numpy().astype(np.int64), band_names=['precip'], band_index=0, sccd_result=sccd_result2, axe=axes[1], trimodal=False, title=f"Fitting using Lasso regression")
    
    # fitting using the last model cofficients from Kalman filter
    sccd_result3= sccd_detect_flex(dates, prep, qas, lam=0, fitting_coefs=False)
    display_sccd_result_single(data=data[['dates', 'precip', 'qa']].to_numpy().astype(np.int64), band_names=['precip'], band_index=0, sccd_result=sccd_result2, axe=axes[2], trimodal=False, title=f"Fitting using the last Kalman filter model")
    



.. image:: 5_state_analysis_greenning&precipitation_coarse_files/5_state_analysis_greenning&precipitation_coarse_16_0.png