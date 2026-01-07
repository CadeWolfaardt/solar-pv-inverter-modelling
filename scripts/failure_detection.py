# stdlib
import os
import time
import itertools
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any, cast, Tuple, Set, Union
# thridpartylib
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astral import LocationInfo
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split # pyright: ignore
# projectlib
from pv_inverter_modeling.utils.typing import Address, Field
from pv_inverter_modeling.data.loaders import Open, load_lazyframe, load_pandas
from pv_inverter_modeling.config.env import (
    RAW_DATA_ROOT, 
    DATA_ROOT, 
    PDM_SAMPLE_DEVICE,
    SITE_NAME,
    SITE_TZ,
    LAT,
    LON,
    COUNTRY,
    DEVICE_RATING
)
from pv_inverter_modeling.data.schemas import Column, Metric, KEYS
from pv_inverter_modeling.preprocessing.astronomy import build_sun_table
from pv_inverter_modeling.visualization.timeseries import Plot, use_dark_theme

def stationary_cols(
        df: pd.DataFrame, 
        cols: Optional[List[str]] = None, 
        include_nulls: bool = True
    ) -> List[str]:
    """
    Identify constant (stationary) columns in a DataFrame.

    This function inspects the number of unique values per column and
    identifies columns that contain either a single unique value or
    (optionally) only null values. Summary information is printed to
    stdout for exploratory inspection.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to analyze.
    cols : list[str], optional
        Subset of columns to consider. If None, all columns in the
        DataFrame are analyzed.
    include_nulls : bool, default True
        If True, columns containing only null values are treated as
        stationary.

    Returns
    -------
    list[str]
        List of column names identified as stationary.
    """
    # Display shape of data
    print(f"Total rows:                   ", df.shape[0] )
    # If no columns under specific consideration, consider all
    if cols is None:
        cols = list(df.columns)
    else:
        cols = list(cols)
    # Obtain number of unique bservations per considered column
    n_unique: Dict[str, Any] = (
        df[cols]
        .nunique(dropna=True)
        .to_dict() # pyright: ignore[reportUnknownMemberType]
    )
    # Define columns as stationary if 1 or 0 unique observations
    drop_cols: List[str] = []
    print(f"Total columns:                ", len(cols))
    for col_name, num in n_unique.items():
        if num == 1 or (num == 0 and include_nulls):
            val = 'Nan' if num == 0 else '1 unique'
            print(f"{col_name:30s}: constant ({val})")
            drop_cols.append(col_name)
        else:
            print( f"{col_name:30s}: {num} uniques")

    return drop_cols

def plot_daily_mean_power(
        df: pd.DataFrame, 
        metric: Field = Metric.AC_POWER, 
        device: Optional[str] = None, 
        title: Optional[str] = None
    ) -> None:
    """
    Plot daily mean power for a selected metric.

    This function filters the input DataFrame to a single metric (and
    optionally a single device), computes the daily mean value, and 
    renders a time-series plot of the resulting daily averages.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format DataFrame containing timestamped device 
        measurements. Must include timestamp, device, metric, and value 
        columns.
    metric : Field, default ``Metric.AC_POWER``
        Metric for which the daily mean power is computed.
    device : str, optional
        If provided, restricts the plot to a single device.
    title : str, optional
        Custom plot title. If not provided, a default title is generated
        based on the selected metric and device.
    """
    df_ = df.copy()
    ts = df_[Column.TIMESTAMP] = pd.to_datetime(
        df_[Column.TIMESTAMP], 
        errors="coerce"
    )
    # Keep the metric rows
    df_ = df_[(df_[Column.METRIC] == metric) & df_[Column.VALUE].notna()]

    # optional: single device
    if device is not None:
        df_ = df_[df_[Column.DEVICE] == device]
    daily = (
        df_.groupby( # pyright: ignore[reportUnknownMemberType]
            ts.dt.date
        )[Column.VALUE]
        .mean()
        .rename(Metric.MEAN_POWER)
        .to_frame()
    )
    # Add timestamp and device for sorting
    idx = cast(pd.PeriodIndex, daily.index)
    daily[Column.TIMESTAMP] = idx.to_timestamp()
    daily[Column.DEVICE] = device or "SITE"
    # plot
    ax = Plot().plot_metric(
        daily,
        Column.TIMESTAMP,
        Metric.MEAN_POWER,
    )
    title = (
        title
        or f"Daily Mean Power - {metric}" + (f" - {device}" if device else "-")
    )
    ax.set_title(title) # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Date") # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel(metric) # pyright: ignore[reportUnknownMemberType]
    plt.show() # pyright: ignore[reportUnknownMemberType]

def plot_monthly_mean_power(
        df: pd.DataFrame, 
        metric: Metric = Metric.AC_POWER, 
        device: Optional[str] = None, 
        title: Optional[str] = None
    ) -> None:
    """
    Plot monthly mean power for a given metric at site or device level.

    This function aggregates time-series power data to monthly means using
    calendar months and visualizes the resulting trend. The aggregation is
    performed over all available observations for the specified metric,
    optionally filtered to a single device. When no device is specified,
    the plot represents site-level monthly averages.

    Internally, timestamps are converted to monthly periods for aggregation
    and then converted back to timestamps for plotting. A device label is
    injected into the aggregated data to maintain compatibility with the
    time-series plotting backend.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time-series measurements. Must include
        at least the columns specified by ``Column.TIMESTAMP``,
        ``Column.METRIC``, ``Column.VALUE``, and ``Column.DEVICE``.
    metric : Metric, optional
        Metric to aggregate and plot. Defaults to ``Metric.AC_POWER``.
    device : str or None, optional
        Device identifier to filter the data before aggregation. If ``None``,
        data from all devices are aggregated to produce a site-level monthly
        mean. Defaults to ``None``.
    title : str or None, optional
        Custom title for the plot. If ``None``, a title is generated
        automatically based on the selected metric and device.

    Returns
    -------
    None
        Displays the plot and does not return a value.
    """
    df_ = df.copy()
    ts = df_[Column.TIMESTAMP] = pd.to_datetime(
        df_[Column.TIMESTAMP], 
        errors="coerce"
    )
    df_ = df_[(df_[Column.METRIC] == metric) & df_[Column.VALUE].notna()]

    if device is not None:
        df_ = df_[df_[Column.DEVICE] == device]

    df_["month"] = ts.dt.to_period("M")
    monthly = (
        df_
        .groupby( # pyright: ignore[reportUnknownMemberType]
            "month"
        )[Column.VALUE]
        .mean()
        .rename(Metric.MEAN_POWER)
        .to_frame()
    )
    # plot (convert period to timestamp for x-axis)
    idx = cast(pd.PeriodIndex, monthly.index)
    monthly[Column.TIMESTAMP] = idx.to_timestamp()
    monthly[Column.DEVICE] = device or "SITE"
    # plot
    ax = Plot().plot_metric(
        monthly,
        Column.TIMESTAMP,
        Metric.MEAN_POWER,
    )
    title = (
        title 
        or f"Monthly Mean Power - {metric}" 
        + (f" - {device}" if device else "")
    )
    ax.set_title(title) # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Month") # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel(metric) # pyright: ignore[reportUnknownMemberType]
    plt.show() # pyright: ignore[reportUnknownMemberType]

def plot_histogram(df: pd.DataFrame, text: str) -> None:
    """
    Plot a histogram of hourly observations.

    This function renders a histogram of the "hours" column in the provided
    Polars DataFrame, showing the distribution of observations across the
    24-hour day. The plot is displayed immediately using Matplotlib.

    Parameters
    ----------
    df : polars.DataFrame
        Input DataFrame containing a column named "hours" representing
        hour-of-day values.
    text : str
        Title text to display above the histogram.
    """
    plt.figure(figsize=(16,5)) # pyright: ignore
    plt.hist( # pyright: ignore
        df["hours"], bins=12, color='skyblue', edgecolor='k'
    ) 
    plt.title(f"{text}") # pyright: ignore
    plt.xlabel("Hour of Day (Local Time)") # pyright: ignore
    plt.ylabel("Frequency") # pyright: ignore
    plt.xlim(0, 24) # pyright: ignore
    plt.grid(alpha=0.3) # pyright: ignore
    plt.show() # pyright: ignore

def block_filter(df_daylight: pd.DataFrame, target: Field) -> pd.DataFrame:
    """
    Apply a block-level validity filter based on a target metric.

    This function enforces block-level consistency by retaining all 
    metrics for device-timestamp pairs where a specified target metric 
    is present, non-null, and within a physically plausible range. Rows 
    associated with module-level devices are excluded.

    The intent is to ensure that if a block (e.g., inverter) is 
    considered valid at a given time, all of its associated measurements
    are kept together.

    Parameters
    ----------
    df_daylight : pandas.DataFrame
        Long-format DataFrame already filtered to daylight observations.
        Must include device identifiers, timestamps, metric names, and 
        values.
    target : Field
        Name of the target metric used to determine block-level 
        validity.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only rows belonging to valid
        device-timestamp blocks, with all original metric columns 
        preserved.
    """
    has_target = (
        df_daylight[Column.METRIC].eq(target) 
        & df_daylight[Column.VALUE].notna()
    ).any()
    print("Has target rows after daylight filter:", has_target)
    CAP = 1.20 * DEVICE_RATING   # 20% safety band

    # 0) Start from your daylight-filtered long table
    df = df_daylight.copy()
    df[Column.TIMESTAMP] = pd.to_datetime(
        df[Column.TIMESTAMP],
        errors="coerce"
    )

    # 1) Drop module level devices
    df_block = df[df[Column.TYPE] == "Inverter"].copy()

    # 2) Define “good” device-time pairs where target exists and is  
    #    within sane (0..CAP)
    m_target = (
        df_block[Column.METRIC].eq(target) 
        & (df_block[Column.VALUE].notna())
    )
    m_cap = (
        df_block[Column.VALUE]
        .between( # pyright: ignore[reportUnknownMemberType]
            0, CAP
        )
    )
    good = (df_block.loc[m_target & m_cap, [*KEYS]].drop_duplicates())

    # 3) Keep **all metrics** for those good BLOCK device-time pairs
    df_clean = (
        df_block
        .merge(
            good.assign(_keep=1), 
            on=KEYS, 
            how="inner"
        )[df_block.columns]
    )
    print(
        "Rows before:", 
        len(df_daylight), 
        "| after block+cap filter:", 
        len(df_clean)
    )
    return df_clean

def print_heatmap(
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix Heatmap"
    ) -> None:
    """
    Display a heatmap visualization of a correlation matrix.

    This function renders a heatmap for a precomputed correlation 
    matrix, allowing quick visual inspection of the strength and 
    direction of pairwise relationships between variables. The heatmap 
    uses a diverging color scale to highlight positive and negative 
    correlations.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square DataFrame containing correlation coefficients, where both
        rows and columns correspond to the same set of variables.
    title : str, optional
        Title displayed above the heatmap. Defaults to
        ``"Correlation Matrix Heatmap"``.

    Returns
    -------
    None
        Displays the heatmap and does not return a value.
    """
    plt.figure(figsize=(15, 10)) # pyright: ignore[reportUnknownMemberType]
    sns.heatmap( # pyright: ignore[reportUnknownMemberType]
        corr_matrix, cmap='coolwarm', annot=False
    )
    plt.title(f'{title}') # pyright: ignore[reportUnknownMemberType]
    plt.show() # pyright: ignore[reportUnknownMemberType]

def plot_hour_density(
        df: pd.DataFrame, 
        title: str = "Irradiance-hour check"
    ) -> None:
    """
    Plot the distribution of observation times over the local day.

    This function converts the timestamp column into a fractional hour
    representation (hour + minute / 60) and visualizes the density of
    observations across the 24-hour day using a histogram. It is 
    intended as a diagnostic plot to verify temporal coverage, detect 
    sampling bias, and identify missing night or daylight periods in 
    time-series data (e.g., SCADA or sensor measurements).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a timestamp column specified by
        ``Column.TIMESTAMP``.
    title : str, optional
        Title displayed on the plot. Defaults to 
        "Irradiance-hour check".

    Returns
    -------
    None
        Displays the histogram and does not return a value.
    """
    ts = pd.to_datetime(df[Column.TIMESTAMP])
    hours = ts.dt.hour + ts.dt.minute/60
    plt.figure(figsize=(14,4)) # pyright: ignore[reportUnknownMemberType]
    sns.histplot(
        x=hours, 
        bins=np.linspace(0, 24, 25), 
        kde=False, 
        color="skyblue"
    )
    plt.title(title) # pyright: ignore[reportUnknownMemberType]
    plt.xlabel( # pyright: ignore[reportUnknownMemberType]
        "Hour of day (local)"
    )
    plt.show() # pyright: ignore[reportUnknownMemberType]

def label_failures(
        df: pd.DataFrame, 
        time_col: Column = Column.TIMESTAMP, 
        p_ac: Metric = Metric.AC_POWER, 
        p_dc: Metric = Metric.DC_POWER, 
        ac_lim: Metric = Metric.ACTIVE_LIMIT, 
        frac: float = 0.60,
        min_ref: float = 2000.0,
        dc_frac: float = 0.50, 
        lim_frac: float = 0.80,
        k: int = 24, 
        m: int = 20
    ) -> pd.DataFrame:
    """
    Label potential failure events in time-series power data using
    rule-based, persistence-aware heuristics.

    This function applies a set of relative, distribution-based rules to
    identify sustained low-power behavior indicative of potential 
    failures. Reference levels are derived internally from the data 
    using rolling monthly-time-slot statistics, and failure labels are 
    assigned only when conditions persist over multiple consecutive 
    observations.

    All numeric thresholds used in this function (including 
    ``min_ref=2000``) are **illustrative defaults only**. They do 
    **not** correspond to any specific site capacity, inverter rating, 
    or proprietary operational threshold, and should be treated as 
    tunable hyperparameters rather than domain-revealing constants.

    Failure detection logic (high level):
    - Compute a month x 5-minute slot reference using the 95th 
      percentile of observed AC (and optionally DC) power.
    - Flag instantaneous low-power events when AC power drops below a 
      fixed fraction of the reference.
    - Apply guard conditions to exclude low-irradiance or constrained 
      regimes.
    - Require persistence over ``m`` out of ``k`` consecutive intervals 
      before labeling a failure.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time-series measurements. Must 
        include a timestamp column and at least an AC power metric.
    time_col : Column, optional
        Column identifying the timestamp used for temporal ordering and 
        feature extraction. Defaults to ``Column.TIMESTAMP``.
    p_ac : Metric, optional
        Metric representing AC power. Defaults to ``Metric.AC_POWER``.
    p_dc : Metric, optional
        Metric representing DC power. Used as an optional guard if 
        present in the DataFrame. Defaults to ``Metric.DC_POWER``.
    ac_lim : Metric, optional
        Metric representing an active power limit or curtailment signal.
        Used as a guard when available. Defaults to 
        ``Metric.ACTIVE_LIMIT``.
    frac : float, optional
        Fraction of the reference AC power below which an observation is
        considered instantaneously low. Defaults to ``0.60``.
    min_ref : float, optional
        Minimum reference AC power required for failure evaluation.
        This value is an **example placeholder only** and does not 
        reflect any real system rating or confidential threshold. 
        Defaults to ``2000.0``.
    dc_frac : float, optional
        Fraction of the DC reference power required to pass the DC 
        guard. Defaults to ``0.50``.
    lim_frac : float, optional
        Fraction of the AC reference power required to pass the 
        active-limit guard. Defaults to ``0.80``.
    k : int, optional
        Size of the rolling window (number of time steps) used for 
        persistence evaluation. Defaults to ``24``.
    m : int, optional
        Minimum number of low-power observations within the rolling 
        window required to label a failure. Defaults to ``20``.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame augmented with intermediate guard 
        signals and two binary labels:
        - ``fail0``: instantaneous low-power indicator
        - ``failure``: persistence-based failure label
    """
    w = df.copy()
    ts = w[time_col] = pd.to_datetime(w[time_col], errors="coerce")
    w = w.sort_values(time_col)
    w["month"] = ts.dt.month
    w["slot5"] = (ts.dt.hour * 60 + ts.dt.minute) // 5

    # slot+month reference for AC
    ref_ac = (
        w.groupby( # pyright: ignore[reportUnknownMemberType]
            ["month", "slot5"]
        )[p_ac]
        .quantile(0.95)
        .rename("ref_ac_95m")
    )
    w = w.merge(ref_ac, on=["month", "slot5"], how="left")
    # DC guard if column exists
    has_dc = p_dc in w.columns
    if has_dc:
        ref_dc = (
            w.groupby( # pyright: ignore[reportUnknownMemberType]
                ["month", "slot5"]
            )[p_dc]
            .quantile(0.95)
            .rename("ref_dc_95m")
        )
        w = w.merge(ref_dc, on=["month", "slot5"], how="left")
        guard_dc = w[p_dc] >= (dc_frac * w["ref_dc_95m"])
    else:
        guard_dc = True

    # Guards
    guard_ref = w["ref_ac_95m"] >= min_ref
    guard_lim = (
        (~w[ac_lim].isna()) & (w[ac_lim] >= lim_frac * w["ref_ac_95m"])
        if ac_lim in w.columns
        else True
    )
    # Instant low vs normal
    low_now = w[p_ac] < frac * w["ref_ac_95m"]
    # Combine guards
    w["fail0"] = (guard_ref & guard_dc & guard_lim & low_now).astype(int)
    # Persistence rule
    roll = w["fail0"].rolling(k, min_periods=1).sum()
    w["failure"] = (roll >= m).astype(int)
    print(
        f"Failures: {int(w['failure'].sum())} / {len(w)} "
        f"({w['failure'].mean()*100:.1f}%)"
    )
    
    return w

def add_time_features(
        df: pd.DataFrame, 
        time_col: Column = Column.TIMESTAMP
    ) -> pd.DataFrame:
    """
    Add cyclic time-based features derived from a timestamp column.

    This function converts the specified timestamp column to pandas
    ``datetime`` (coercing invalid values to ``NaT``) and augments the
    DataFrame with sinusoidal encodings of hour-of-day and day-of-year.
    These features are commonly used in machine-learning models to
    represent cyclical temporal patterns without introducing artificial
    discontinuities.

    The following columns are added to the DataFrame:
    - ``hour_sin`` / ``hour_cos``: cyclic encoding of hour-of-day
    - ``doy_sin`` / ``doy_cos``: cyclic encoding of day-of-year

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a timestamp column.
    time_col : Column, optional
        Column identifying the timestamp to use for feature generation.
        Defaults to ``Column.TIMESTAMP``.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with additional cyclic time 
        features appended.
    """
    df = df.copy()
    ts = df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy  = ts.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 366)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 366)

    return df

def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form metric data to a wide-format table.

    This function pivots a long-form DataFrame containing metric/value 
    pairs into a wide-format DataFrame with one column per metric. Rows 
    are indexed by the columns specified in ``KEYS`` (e.g., device and 
    timestamp), and metric values are aggregated using the median where 
    multiple observations exist for the same index/metric combination.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in long format. Must contain the columns 
        specified by ``Column.METRIC`` and ``Column.VALUE``, as well as 
        all columns listed in ``KEYS``.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one column per metric and one row per
        unique combination of index keys.
    """
    idx = [*KEYS]
    wide = df.pivot_table( # pyright: ignore[reportUnknownMemberType]
        index=idx, 
        columns=Column.METRIC, 
        values=Column.VALUE, 
        aggfunc="median"
    ).reset_index()

    return wide

def _safe_proba(
        model: Union[XGBClassifier, RandomForestClassifier], 
        X: pd.DataFrame
    ) -> np.ndarray:
    """
    Safely compute class probability estimates for binary classifiers.

    This helper attempts to obtain class probabilities via
    ``model.predict_proba`` and normalizes the output to a two-column
    ``(n_samples, 2)`` array representing ``P(class=0)`` and
    ``P(class=1)``. If the model returns a single probability column,
    it is interpreted as the probability of the positive class and
    complemented accordingly.

    If probability prediction is not supported or fails for any reason,
    the function falls back to using hard class predictions from
    ``model.predict`` and converts them into a pseudo-probability
    representation.

    Parameters
    ----------
    model : XGBClassifier or RandomForestClassifier
        Fitted binary classification model supporting ``predict`` and,
        optionally, ``predict_proba``.
    X : pd.DataFrame
        Input feature matrix used for prediction.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, 2)`` containing probability 
        estimates for the negative and positive classes respectively. In
        fallback mode, values correspond to hard-label–derived 
        pseudo-probabilities.
    """
    try:
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p
        # If only one column, treat it as prob of class 1
        return np.column_stack([1 - p[:, 0], p[:, 0]])
    except Exception:
        # Fallback: use predict() as hard labels
        y_hat = model.predict(X)
        y_hat = np.asarray(y_hat, dtype=float).ravel()
        return np.column_stack([1 - y_hat, y_hat])
    
def train_xgb(
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        base_params: Optional[Dict[Any, Any]] = None, 
        prog_params: Optional[Dict[Any, Any]] = None, 
        early_stopping_rounds: Optional[int] = 50, 
        plot: bool = False, 
        verbose: bool = False
    ) -> Dict[str, Any]:
    """
    Train baseline and progressive XGBoost classifiers with evaluation 
    metrics.

    This function trains two binary XGBoost classification models:

    1. A baseline model trained with fixed hyperparameters and no 
       evaluation history tracking.
    2. A progressive model trained with a larger estimator budget and 
       optional early stopping, using a held-out validation set to 
       monitor performance.

    Class imbalance is handled by computing and applying a
    ``scale_pos_weight`` based on the training labels. Model performance
    is evaluated on the test set using ROC-AUC and PR-AUC. Optionally, 
    the validation PR-AUC trajectory over boosting rounds is plotted.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.DataFrame
        Training binary target labels.
    X_test : pd.DataFrame
        Test feature matrix used for evaluation and early stopping.
    y_test : pd.DataFrame
        Test binary target labels.
    base_params : dict, optional
        Hyperparameters for the baseline XGBoost model. If ``None``, a 
        default parameter set is used.
    prog_params : dict, optional
        Hyperparameters for the progressive XGBoost model. If ``None``, 
        a default parameter set is used.
    early_stopping_rounds : int or None, optional
        Number of boosting rounds without improvement on the validation 
        PR-AUC before early stopping is triggered for the progressive 
        model. If ``None`` or non-positive, early stopping is disabled. 
        Defaults to ``50``.
    plot : bool, optional
        If ``True``, plot validation PR-AUC versus boosting rounds for 
        the progressive model. Defaults to ``False``.
    verbose : bool, optional
        Verbosity flag controlling diagnostic output during training. 
        Defaults to ``False``.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"xgb"``: trained baseline ``XGBClassifier``
        - ``"xgb_prog"``: trained progressive ``XGBClassifier``
        - ``"hist"``: evaluation history from the progressive model
        - ``"metrics"``: dictionary with test-set ROC-AUC and PR-AUC
        - ``"p_test"``: predicted positive-class probabilities on the 
                        test set

    Notes
    -----
    - The progressive model is trained with an evaluation set and may 
      stop early depending on validation performance.
    - Probability predictions are obtained using a safety wrapper to 
      handle model-specific edge cases.
    - Hyperparameters and plotting behavior are intended for exploratory
      modeling and may require tuning for production use.
    """
    # Default base parameters
    if base_params is None:
        base_params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            eval_metric="auc",
        )
    # Stage 1: Base model
    if prog_params is None:
        prog_params = dict(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            eval_metric=["auc", "aucpr"],
        )
    # Class imbalance - weight
    pos_w = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    base_params = dict(base_params, scale_pos_weight=pos_w)
    prog_params = dict(prog_params, scale_pos_weight=pos_w)
    # Train base model
    xgb = XGBClassifier(**base_params)
    xgb.fit(X_train, y_train)

    p_test = _safe_proba(xgb, X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, p_test)
    pr_auc = average_precision_score(y_test, p_test)

    # Stage 2: Progressive model with eval history 
    xgb_prog = XGBClassifier(**prog_params)
    callbacks: List[EarlyStopping] = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.append(
            EarlyStopping(
                rounds=early_stopping_rounds,
                metric_name="aucpr",
                data_name="validation_0",
                save_best=True,
            )
        )
    # Fit progressive model
    xgb_prog.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=verbose,
    )
    hist = xgb_prog.evals_result()
    if plot:
        # Plot validation PR-AUC vs trees
        val_aucpr = hist["validation_0"]["aucpr"]
        plt.figure() # pyright: ignore[reportUnknownMemberType]
        plt.plot(val_aucpr) # pyright: ignore[reportUnknownMemberType]
        plt.xlabel( # pyright: ignore[reportUnknownMemberType]
            "Boosting rounds"
        )
        plt.ylabel( # pyright: ignore[reportUnknownMemberType]
            "Validation PR-AUC"
        )
        plt.title( # pyright: ignore[reportUnknownMemberType]
            "XGB validation PR-AUC vs trees"
        )
        plt.tight_layout()
        plt.show() # pyright: ignore[reportUnknownMemberType]

    return {
        "xgb": xgb,
        "xgb_prog": xgb_prog,
        "hist": hist,
        "metrics": {"roc_auc": roc_auc, "pr_auc": pr_auc},
        "p_test": p_test,
    }

def flush_buffer(
        buf: List[Dict[str, Any]], 
        csv_path: Address
    ) -> None:
    """
    Append buffered records to a CSV file and clear the buffer.

    This function writes the contents of an in-memory buffer to disk in
    CSV format. If the target file does not already exist, it is created
    and a header row is written. Records are appended otherwise. After a
    successful write, the buffer is cleared in place.

    If the buffer is empty, the function returns without performing any
    file operations.

    Parameters
    ----------
    buf : list[dict[str, Any]]
        List of row dictionaries representing records to be written.
        Each dictionary should have a consistent set of keys.
    csv_path : Address
        Path to the CSV file to which records will be appended.

    Returns
    -------
    None
        Writes buffered records to disk and clears the buffer.
    """
    if not buf:
        return
    df_buf = pd.DataFrame(buf)
    header = not os.path.exists(csv_path)
    df_buf.to_csv(csv_path, mode="a", index=False, header=header)
    print(f"[save] Wrote {len(buf)} rows to {csv_path}")
    buf.clear()

def run_xgb_sweep(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        input_path: Address, 
        output_csv: Address
    ) -> None:
    """
    Run a hyperparameter sweep for XGBoost classifiers and record 
    results.

    This function performs a grid-based hyperparameter sweep over 
    multiple XGBoost model configurations. For each configuration, a 
    binary classification model is trained, evaluated on both training 
    and test sets, and key performance metrics (ROC-AUC and PR-AUC) are 
    computed. Results are incrementally written to a CSV file to support
    long-running experiments and safe resumption.

    The sweep is resumable: if the output CSV already exists, previously
    completed model configurations are detected and skipped 
    automatically.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training binary target labels.
    X_test : pd.DataFrame
        Test feature matrix used for evaluation.
    y_test : pd.Series
        Test binary target labels.
    input_path : Address
        Path to the input dataset associated with the sweep. This value 
        is used for logging and traceability only and is not read inside
        this function.
    output_csv : Address
        Path to the CSV file where sweep results will be appended. The 
        file is created if it does not already exist.

    Returns
    -------
    None
        Writes sweep results to disk and does not return a value.

    Notes
    -----
    - Hyperparameters, evaluation metrics, and model identifiers are 
      logged for comparative analysis.
    - Results are flushed to disk in chunks to reduce memory usage and 
      to allow recovery from interruptions.
    - Any failed model configurations are recorded with an error message
      instead of metrics.
    """
    print("\n==============================")
    print("Running XGB sweep")
    print("Input parquet:", input_path)
    print("Output csv:", output_csv)
    print("==============================\n")
    # Hyperparams to sweep
    max_depths       = [3, 4, 6, 8, 10]
    learning_rates   = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08]
    n_estimators_lst = [200, 400, 800, 1000]
    min_child_wts    = [1, 5, 10, 20, 40]
    colsample_vals   = [0.6, 0.8, 0.9]
    subsample_vals   = [0.7, 0.8, 0.9]
    # Resume support
    done: Set[Any] = set()
    if os.path.exists(output_csv):
        try:
            prev = pd.read_csv( # pyright: ignore[reportUnknownMemberType]
                output_csv, 
                usecols=["model"]
            )
            done = set(prev["model"].astype(str).tolist())
            print(
                f"[resume] Found {len(done)} completed models. Skipping them."
            )
        except Exception as e:
            print(
                f"[resume] Could not read existing CSV ({e}). Starting fresh."
            )
    # Sweep iterator
    grid_iter = itertools.product(
        max_depths,
        learning_rates,
        n_estimators_lst,
        min_child_wts,
        colsample_vals,
        subsample_vals
    )
    buffer: List[Dict[str, Any]] = []
    counter = 0
    CHUNK_SIZE = 100
    start_time = time.time()

    for md, lr, est, mcw, col, sub in grid_iter:
        model_name = f"XGB_d{md}_lr{lr}_n{est}_mcw{mcw}_col{col}_sub{sub}"
        # Skip previously completed
        if model_name in done:
            continue
        print(f"\n[{counter+1}] Training {model_name}")
        # Base + progressive params
        base_params = dict(
            n_estimators=est,
            max_depth=md,
            learning_rate=lr,
            subsample=sub,
            colsample_bytree=col,
            min_child_weight=mcw,
            tree_method="hist",
            random_state=42,
            eval_metric="auc",
        )
        prog_params = dict(
            n_estimators=max(600, est*2),
            max_depth=md,
            learning_rate=lr,
            subsample=sub,
            colsample_bytree=col,
            min_child_weight=mcw,
            tree_method="hist",
            random_state=42,
            eval_metric=["auc", "aucpr"],
        )
        try:
            # Train model
            res = train_xgb(
                X_train, y_train,
                X_test, y_test,
                base_params=base_params,
                prog_params=prog_params,
                early_stopping_rounds=None,
                plot=False,
                verbose=False
            )
            model = res["xgb"]
            # Compute metrics safely
            p_tr = _safe_proba(model, X_train)[:, 1]
            p_te = _safe_proba(model, X_test)[:, 1]

            auc_tr = roc_auc_score(y_train, p_tr)
            auc_te = roc_auc_score(y_test,  p_te)
            pr_tr  = average_precision_score(y_train, p_tr)
            pr_te  = average_precision_score(y_test,  p_te)
            print(
                f"{model_name} | AUROC tr={auc_tr:.3f}, te={auc_te:.3f} | "
                f"PRAUC tr={pr_tr:.3f}, te={pr_te:.3f}"
            )
            # Buffer the results
            buffer.append({
                "model": model_name,
                "max_depth": md,
                "learning_rate": lr,
                "n_estimators": est,
                "min_child_weight": mcw,
                "colsample_bytree": col,
                "subsample": sub,
                "AUROC_train": auc_tr,
                "AUROC_test": auc_te,
                "AUROC_gap": auc_tr - auc_te,
                "PRAUC_train": pr_tr,
                "PRAUC_test": pr_te,
                "PRAUC_gap": pr_tr - pr_te,
                "AUROC_rel_drop": (auc_tr - auc_te) / max(auc_tr, 1e-9),
                "PRAUC_rel_drop": (pr_tr - pr_te) / max(pr_tr, 1e-9),
            })
        except Exception as e:
            print(f"[warn] {model_name} FAILED: {e}")
            buffer.append({
                "model": model_name,
                "max_depth": md,
                "learning_rate": lr,
                "n_estimators": est,
                "min_child_weight": mcw,
                "colsample_bytree": col,
                "subsample": sub,
                "error": str(e)
            })
        counter += 1
        # Save every CHUNK_SIZE models
        if counter % CHUNK_SIZE == 0:
            flush_buffer(buffer, output_csv)
    # Final flush
    flush_buffer(buffer, output_csv)
    print("\nSweep complete.")
    print(f"Total models trained: {counter}")
    print(f"Results saved to: {output_csv}")
    print(f"Total time: {time.time() - start_time:.1f} seconds\n")

def train_rf(
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        rf_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
    """
    Train a Random Forest classifier and evaluate binary classification 
    metrics.

    This function fits a ``RandomForestClassifier`` using the provided 
    training data and evaluates its performance on both the training and
    test sets. Class probability estimates are obtained using a safety 
    wrapper to ensure consistent output shape across models. Performance 
    is reported using ROC-AUC and PR-AUC, along with simple 
    generalization gap diagnostics.

    If no hyperparameters are provided, a default configuration suitable 
    for imbalanced binary classification is used.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training binary target labels.
    X_test : pd.DataFrame
        Test feature matrix used for evaluation.
    y_test : pd.Series
        Test binary target labels.
    rf_params : dict, optional
        Hyperparameters passed directly to ``RandomForestClassifier``.
        If ``None``, a default parameter set is used.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"rf"``: the trained ``RandomForestClassifier`` instance
        - ``"metrics"``: a dictionary with training and test ROC-AUC and
          PR-AUC values, as well as generalization gaps

    Notes
    -----
    - Class imbalance is handled via 
      ``class_weight="balanced_subsample"`` in the default 
      configuration.
    - The function is intended for exploratory modeling and 
      benchmarking; further tuning may be required for production use.
    """
    if rf_params is None:
        rf_params = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    p_tr = _safe_proba(rf, X_train)[:, 1]
    p_te = _safe_proba(rf, X_test)[:, 1]
    auc_tr = roc_auc_score(y_train, p_tr)
    auc_te = roc_auc_score(y_test,  p_te)
    pr_tr  = average_precision_score(y_train, p_tr)
    pr_te  = average_precision_score(y_test,  p_te)

    return {
        "rf": rf,
        "metrics": {
            "AUROC_train": auc_tr,
            "AUROC_test": auc_te,
            "PRAUC_train": pr_tr,
            "PRAUC_test": pr_te,
            "AUROC_gap": auc_tr - auc_te,
            "PRAUC_gap": pr_tr - pr_te,
        }
    }

def run_rf_sweep(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        input_path: Address, 
        output_csv: Address
    ) -> None:
    """
    Run a hyperparameter sweep for Random Forest failure classification.

    This function performs a grid-based hyperparameter sweep over 
    multiple ``RandomForestClassifier`` configurations. For each 
    configuration, a model is trained on the provided training data, 
    evaluated on both training and test sets, and performance metrics 
    are recorded. Results are written incrementally to a CSV file to 
    support long-running experiments and safe resumption.

    The sweep is resumable: if the output CSV already exists, previously
    completed model configurations are detected and skipped 
    automatically.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training binary target labels.
    X_test : pd.DataFrame
        Test feature matrix used for evaluation.
    y_test : pd.Series
        Test binary target labels.
    input_path : Address
        Path to the input dataset associated with the sweep. This 
        parameter is used for logging and traceability only and is not 
        read inside this function.
    output_csv : Address
        Path to the CSV file where sweep results will be appended. The 
        file is created if it does not already exist.

    Returns
    -------
    None
        Writes sweep results to disk and does not return a value.

    Notes
    -----
    - Performance is evaluated using ROC-AUC and PR-AUC on both training 
      and test sets, along with generalization gap diagnostics.
    - Results are flushed to disk in fixed-size chunks to reduce memory 
      usage and to allow recovery from interruptions.
    - Failed model configurations are recorded with an error message 
      instead of metrics.
    """
    print("\n==============================")
    print("Running RF sweep")
    print("Input parquet:", input_path)
    print("Output csv:", output_csv)
    print("==============================\n")
    # Hyperparam sweep
    n_estimators_lst   = [200, 300, 600, 800]
    max_depths         = [10, 12, 14, 16, 18]
    min_samples_leafs  = [50, 75, 100, 125]
    max_features_vals  = ["sqrt"]
    class_weights      = ["balanced_subsample", "balanced"]
    # Resume support
    done: Set[Any] = set()
    if os.path.exists(output_csv):
        try:
            prev = pd.read_csv( # pyright: ignore[reportUnknownMemberType]
                output_csv, 
                usecols=["model"]
            )
            done = set(prev["model"].astype(str).tolist())
            print(
                f"[resume] Found {len(done)} completed models. Skipping them."
            )
        except Exception as e:
            print(
                f"[resume] Could not read existing CSV ({e}). Starting fresh."
            )
    grid_iter = itertools.product(
        n_estimators_lst,
        max_depths,
        min_samples_leafs,
        max_features_vals,
        class_weights
    )

    buffer: List[Dict[str, Any]] = []
    counter = 0
    CHUNK_SIZE = 100
    start_time = time.time()

    for est, md, leaf, mf, cw in grid_iter:
        model_name = f"RF_n{est}_d{md}_leaf{leaf}_mf{mf}_cw{cw}"
        if model_name in done:
            continue

        print(f"\n[{counter+1}] Training {model_name}")
        rf_params = dict(
            n_estimators=est,
            max_depth=md,
            min_samples_leaf=leaf,
            max_features=mf,
            class_weight=cw,
            n_jobs=-1,
            random_state=42
        )
        try:
            res = train_rf(
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                rf_params=rf_params
            )
            m = res["metrics"]

            print(
                f"{model_name} | AUROC tr={m['AUROC_train']:.3f}, "
                f"te={m['AUROC_test']:.3f} | "
                f"PRAUC tr={m['PRAUC_train']:.3f}, te={m['PRAUC_test']:.3f}"
            )

            buffer.append({
                "model": model_name,
                "n_estimators": est,
                "max_depth": md,
                "min_samples_leaf": leaf,
                "max_features": mf,
                "class_weight": cw,
                **m,
                "AUROC_rel_drop": m["AUROC_gap"] / max(m["AUROC_train"], 1e-9),
                "PRAUC_rel_drop": m["PRAUC_gap"] / max(m["PRAUC_train"], 1e-9),
            })

        except Exception as e:
            print(f"[warn] {model_name} FAILED: {e}")
            buffer.append({
                "model": model_name,
                "n_estimators": est,
                "max_depth": md,
                "min_samples_leaf": leaf,
                "max_features": mf,
                "class_weight": cw,
                "error": str(e)
            })
        counter += 1
        if counter % CHUNK_SIZE == 0:
            flush_buffer(buffer, output_csv)
    flush_buffer(buffer, output_csv)
    print("\nRF sweep complete.")
    print(f"Total models trained: {counter}")
    print(f"Results saved to: {output_csv}")
    print(f"Total time: {time.time() - start_time:.1f} seconds\n")

def filter_for_device(source: Address, out: Address) -> None:
    """
    Load raw data, inspect a sample device, and write a filtered subset 
    to disk.

    This function reads data from the given source address, prints a 
    small preview and row count for a predefined sample device, and then
    writes the corresponding subset to the output address.

    Parameters
    ----------
    source : Address
        Input data location to read from.
    out : Address
        Output location where the filtered subset will be written.

    Raises
    ------
    RuntimeError
        If the input data cannot be read.
    """
    # Read in data
    lf = load_lazyframe(source)
    # Look at head of data
    preview = (
        lf
        .filter(pl.col(Column.DEVICE) == PDM_SAMPLE_DEVICE)
        .select([Column.TIMESTAMP, Column.DEVICE, Column.METRIC, Column.VALUE])
        .limit(10)
        .collect(engine="streaming")
    )
    print(preview)
    # Create lazy subset for sample inverter
    lf_sample = lf.filter(pl.col(Column.DEVICE) == PDM_SAMPLE_DEVICE)
    # View row count
    count = lf_sample.select(pl.count()).collect(engine="streaming")
    print(count)
    # Save the file
    t1 = time.time()
    with Open(out, mode="w") as o:
        o.write(lf_sample, reverse_mapping=True)

    print(f"Wrote file, {out}, in {time.time()-t1} secs")

def filter_for_daylight(source: Address, out: Address) -> None:
    """
    Preprocess inverter time-series data and apply daylight-based 
    filtering.

    This function loads raw inverter data, performs exploratory 
    validation, removes stationary columns, filters invalid 
    device-timestamp blocks based on a target metric, applies a 
    sunrise/sunset daylight filter using astronomical data, and writes 
    the cleaned dataset to disk.

    Several diagnostic summaries and plots are produced to verify data 
    quality before and after filtering.

    Parameters
    ----------
    source : Address
        Input data location containing raw inverter measurements.
    out : Address
        Output location where the cleaned, daylight-filtered dataset is 
        written.

    Raises
    ------
    RuntimeError
        If the input data cannot be read.
    """
    # Read the data
    print(
        "Ready to read the data for inverter "
        f"{PDM_SAMPLE_DEVICE} from: {source}"
    )
    df = load_pandas(source)
    # View snippet of data
    print(df.head())
    # Verify sample device occurences
    n_unique = df[Column.DEVICE].nunique(dropna=True)
    print(n_unique)
    # Confirm occurences match sample selected
    names = sorted(df[Column.DEVICE].unique())
    print("Listing unique device names:")
    counter = 1
    for name in names:
        if counter < 10:
            print(counter, " : ",name)
        else:
            print(counter,": ",name)
        counter += 1    

    # Drop stationary columns
    drop_cols = stationary_cols(df)
    drop_cols.append(Column.TIMESTAMP)
    print(f"\nColumns to drop: {drop_cols}\n")
    df.drop(columns=drop_cols, inplace=True)
    # Confirm there are no more stationary columns
    checl_cols_to_drop = stationary_cols(df)
    if len(checl_cols_to_drop) == 0:
        print("**** No more columbs to drop currently ****")       
    # Filter data
    target = Metric.AC_POWER
    mask = (df[Column.METRIC].eq(target)) & (df[Column.VALUE].notna())
    good =  df.loc[mask, Column.TIMESTAMP].drop_duplicates()
    t1 = time.time()
    df_filter = df.merge(good.assign(keep=1), on=KEYS, how="inner")
    print(f"Time taken: {time.time()-t1} secs")
    flag = (
        df_filter[df_filter[Column.METRIC] == target][Column.VALUE]
        .isna()
        .sum()
    )
    if flag == 0:
        print('No missing values\n=======================\n')
    else:
        print('Missing values')
    print(f"Number of rows: {df_filter.shape[0]}")
    # Drop unused columns
    keep_cols: List[str] = [*KEYS, Column.METRIC, Column.VALUE]
    df_long = df_filter[keep_cols].copy()
    df_long[Column.TIMESTAMP] = pd.to_datetime(
        df_long[Column.TIMESTAMP], 
        errors="coerce"
    )
    df_long = df_long.sort_values(KEYS)
    print("df_long shape:", df_long.shape) 
    # Irradiance hours
    ts = pd.to_datetime(df_long[Column.TIMESTAMP])
    df_long["hours"] = ts.dt.hour + ts.dt.minute / 60
    # Plot histogram
    plot_histogram(
        df_long, 
        "Data Frequency Accross Hours Before Irradiance Filter"
    )
    # Obtain site data and timezone
    site = LocationInfo(SITE_NAME, COUNTRY, SITE_TZ, LAT, LON)
    tz_local = ZoneInfo(SITE_TZ)
    # Obtain min and max date
    date_min = df_long[Column.TIMESTAMP].min()
    date_max = df_long[Column.TIMESTAMP].max()
    print(date_min)
    print(date_max)
    # Build sun table
    t1 = time.time()
    sun_table = build_sun_table(date_min, date_max, site, tz_local)
    print(f"\nTime taken to build sun table: {time.time()-t1} secs")
    # Merge on calendar date
    t1 = time.time()
    df_long["date_local"] = pd.to_datetime(df_long[Column.TIMESTAMP]).dt.date
    m = df_long.merge(sun_table, on="date_local", how="left")
    print(f"\nTime taken to merge sun table: {time.time()-t1} secs")

    # Keep rows within sunrise..sunset
    mask = (
        m["sunrise_local"].notna() &
        m["sunset_local"].notna()  &
        (m["event_local_time"] >= m["sunrise_local"]) &
        (m["event_local_time"] <= m["sunset_local"])
    )
    df_daylight = m.loc[mask, [*KEYS, Column.METRIC, Column.VALUE]].copy()
    print(
        "\nBefore:", 
        len(df_long), 
        "After daylight filter:", 
        len(df_daylight)
    )
    df_daylight.head()

    # Compute distribution after filter
    df_ = df_daylight.copy()
    df_[Column.TIMESTAMP] = pd.to_datetime(
        df_[Column.TIMESTAMP], 
        errors="coerce"
    )
    ts = pd.to_datetime(df_[Column.TIMESTAMP], errors="coerce")
    df_["hours"] = ts.dt.hour + ts.dt.minute / 60
    # Plot Histogram
    plot_histogram(df_, "Data Frequency Across Hours After Irradiance Filter")
    df_clean = block_filter(df_daylight, target)

    # Save the cleaned data
    to_save = df_clean
    with Open(out, mode="w") as o:
        o.write(to_save, reverse_mapping=True)
    print("Saved cleaned data shape:", to_save.shape)
    print(f"\nSaved cleaned data to: {out}\n")

    # Plot daily mean power
    plot_daily_mean_power(
        df_clean, 
        metric=Metric.AC_POWER, 
        device=None, 
        title=None
    )    

def label_timeseries_dataset(
        source: Address, 
        out: Address, 
        plot: bool = False
    ) -> None:
    """
    Prepare, analyze, and label a time-series dataset from long to wide 
    format.

    This function implements an end-to-end preprocessing pipeline for
    inverter-level time-series data. It loads long-form metric data from
    disk, optionally generates exploratory plots, pivots the data to a
    wide format, engineers time-based features, performs basic 
    data-quality analysis, reduces highly correlated features, and 
    applies rule-based failure labeling. The resulting labeled dataset 
    is written to disk.

    The function is designed as a procedural analysis step and produces
    side effects (plots, console summaries, and file output) rather than
    returning intermediate results.

    High-level steps:
    - Load time-series data from ``source`` using a lazy Polars reader
    - Optionally generate exploratory temporal plots
    - Pivot data from long to wide format (metrics as columns)
    - Normalize and label failures using heuristic rules
    - Add cyclic time features (hour-of-day, day-of-year)
    - Inspect and clean missing values
    - Remove highly correlated features
    - Apply persistence-aware failure labeling
    - Save the final labeled dataset to ``out``

    Parameters
    ----------
    source : Address
        Location of the input dataset in long format.
    out : Address
        Destination where the processed and labeled dataset will be 
        written.
    plot : bool, optional
        If ``True``, generate exploratory plots during processing
        (e.g., hourly density, daily and monthly mean power).
        Defaults to ``False``.

    Returns
    -------
    None
        Writes the processed dataset to disk and does not return a 
        value.
    """
    df = load_pandas(source)
    # Plot data frequeny (hourly)
    if plot:
        print(f"Plotting data frequency (hourly) for file: {source}\n")
        plot_hour_density(df)
        # Plot daily mean
        print(f"Plotting daily mean power for file: {source}\n")
        plot_daily_mean_power(df, metric=Metric.AC_POWER)
        # Plot monthly mean
        print(f"Plotting monthly mean power for file: {source}\n")
        plot_monthly_mean_power(df, metric=Metric.AC_POWER)

        print("Done plotting.\n")

    # Pivot: From long to wide format 
    wide = long_to_wide(df)
    print(wide.head())
    print(f"\nNew shape: {wide.shape}\n")

    #  Normalization and Labeling
    FAIL_THRESHOLD = 0.40
    LIMIT_RATIO = 0.9
    # Compute normalized power
    wide["norm_power"] = wide[Metric.AC_POWER] / DEVICE_RATING
    wide["is_limited"] = (
        wide[Metric.AC_POWER] <= LIMIT_RATIO * wide[Metric.ACTIVE_LIMIT]
    )
    # Failure label (simple version)
    wide["failure"] = (
        (wide["norm_power"] <= FAIL_THRESHOLD) 
        & (~wide["is_limited"])
    ).astype(int)
    print("\n", wide["failure"].value_counts(), "\n")
    perc_failure = wide["failure"].value_counts()
    print(f"Percent of failure: {(perc_failure[1]/wide.shape[0] * 100):.2f}%")

    # Adding time features
    wide = add_time_features(wide)
    print("\n******Added time features.*****\n")
    print(f"Wide.shape: {wide.shape}")
    print("\nWide.head()", wide.head(), "\n")  

    # Handling Nans
    print(f"\nwide.info(): {wide.info()}")
    print(f"wide.isna().sum(): {wide.isna().sum()}")
    wide_df = wide.copy()
    metric_cols: List[str] = [
        col for col in wide_df.columns if col not in KEYS
    ]
    print("\nlen(metric_cols), metric_cols: ", len(metric_cols), metric_cols)
    num_rows = len(wide_df)
    nan_count = wide_df[metric_cols].isna().sum(axis=0)
    nan_percentage = (
        nan_count.astype("float64") / float(num_rows) * 100
    ).round(2)
    print(f"\nPercentage of Nans:\n{nan_percentage}")
    summary = pd.DataFrame({
        'num_rows': num_rows,
        'nan_count': nan_count,
        'nan_percentage': nan_percentage,
        'non_nulls' : num_rows - nan_count,
        'dtype' : wide_df[metric_cols].dtypes.astype(str),
    }).sort_values(by='nan_percentage', ascending=False)
    print(f"\nsummary.head(): \n{summary.head()}")
    # Drop rows with any missing values
    wide_df = wide_df.dropna() # pyright: ignore[reportUnknownMemberType]
    # Display the shape of the new dataframe
    print(
        f"\nShape of dataframe after dropping rows with NaNs: {wide_df.shape}"
    )
    print(
        "\nNAN count after filling: ", 
        int(wide_df[metric_cols].isna().sum().sum())
    )
    # Exploring data after cleaning 
    print(f"wide_df.head(): \n{wide_df.head()}")
    # Calculate the correlation matrix
    correlation_matrix = wide_df[metric_cols].corr()
    # Display the correlation matrix
    print(correlation_matrix)
    print_heatmap(
        correlation_matrix, 
        title="Correlation Matrix of Metric Features"
    )
    # Set the correlation threshold
    threshold = 0.6
    # Calculate the absolute correlation matrix
    abs_correlation_matrix = correlation_matrix.abs()
    # Select upper triangle of correlation matrix
    upper = (
        abs_correlation_matrix
        .where(
            np.triu(np.ones(abs_correlation_matrix.shape), k=1).astype(bool)
        )
    )
    # Find features with correlation greater than the threshold, 
    # excluding the target
    to_drop = [
        column for column in upper.columns 
        if any(upper[column] > threshold) and column != Metric.AC_POWER 
    ]
    print("Features to drop:", to_drop)
    # Drop the identified features from the dataframe
    wide_df_reduced = wide_df.drop(columns=to_drop)
    # Display the shape of the new dataframe
    print(
        "Shape of dataframe after dropping highly correlated features:", 
        wide_df_reduced.shape
    )
    wide_df = wide_df_reduced
    # Update metric_cols to reflect the columns in wide_df
    metric_cols = [col for col in wide_df.columns if col not in KEYS]
    # Calculate the correlation matrix
    correlation_matrix = wide_df[metric_cols].corr()
    # Display the correlation matrix
    print(correlation_matrix)
    print(f"\nwide_df.columns\n", wide_df.columns)
    # Visualize the correlation matrix as a heatmap
    print_heatmap(
        correlation_matrix, 
        title='Correlation Matrix of Metric Features'
    )
    print(f"wide_df.shape: {wide_df.shape}")
    # Add failure features
    w = label_failures(wide_df)
    print(f"\nLabeled data w.head(): {w.head()}\n")    
    print(f"\nw.cols\n: {w.columns}")

    # Save the labeled data 
    with Open(out, mode="w") as o:
        o.write(w, reverse_mapping=True)
          
def cluster(source: Address, out_km: Address, out_dbscan: Address) -> None:
    """
    Perform unsupervised clustering on time-series feature data and 
    write results.

    This function loads a preprocessed dataset from disk, applies 
    feature scaling, and performs unsupervised clustering using both 
    K-Means and DBSCAN. Cluster assignments are appended to the dataset 
    and written back to disk as separate outputs for each clustering 
    method.

    The function is designed as an exploratory analysis step and 
    operates procedurally: it produces console output, performs 
    in-memory clustering, and writes results to disk rather than 
    returning values.

    High-level steps:
    - Load a dataset from ``source`` using a lazy Polars reader
    - Materialize the data into pandas for scikit-learn compatibility
    - Select a predefined set of numeric and engineered features
    - Standardize features using ``StandardScaler``
    - Apply K-Means clustering with a fixed number of clusters
    - Apply DBSCAN clustering to identify density-based structure and 
      noise
    - Write clustered datasets to ``out_km`` and ``out_dbscan`` 
      respectively

    Parameters
    ----------
    source : Address
        Path to the input dataset to be clustered.
    out_km : Address
        Destination path for the dataset augmented with K-Means cluster 
        labels.
    out_dbscan : Address
        Destination path for the dataset augmented with DBSCAN cluster 
        labels.

    Returns
    -------
    None
        Writes clustered datasets to disk and does not return a value.

    Notes
    -----
    - Feature selection, scaling, and clustering hyperparameters (e.g.,
      number of clusters, DBSCAN ``eps`` and ``min_samples``) are fixed 
      and intended for exploratory analysis.
    - Cluster labels are written using reverse column-name mapping to
      preserve external schema compatibility.
    """
    df = load_pandas(source)
    print(df.head())
    cols = df.columns
    print("\n", cols, "\n")

    weather_cols: List[str] = [
        Metric.AC_CURRENT_A,
        Metric.AC_CURRENT_MAX,
        Metric.AC_POWER,
        Metric.DC_BATT_BUS_VOLTAGE,
        Metric.DC_CURRENT_MAX,
        Metric.DC_BUS_VOLTAGE,
        Metric.ENERGY_DELIVERED,
        Metric.ENERGY_DELIVERED_DAILY,
        Metric.ENERGY_RECEIVED,
        Metric.FREQUENCY,
        Metric.VAR,
        Metric.DAILY_REACTIVE_POWER,
        Metric.MONTHLY_REACTIVE_POWER,
        'hour_sin', # engineered column 
        'hour_cos', # engineered column
        'doy_sin'   # engineered column
    ]

    scaler = StandardScaler()
    X_all = scaler.fit_transform( # pyright: ignore[reportUnknownMemberType]
        df[weather_cols]
    )
    print(X_all)
    # KMeans Clustering
    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42).fit(X_all)
    df_kmeans = df.copy()
    df_kmeans['cluster'] = kmeans.labels_.astype("int16")
    print("KMeans cluster counts: ")
    print(df_kmeans['cluster'].value_counts().sort_index())
    kmean_outfile = out_km
    with Open(out_km, mode='w') as o:
        o.write(df_kmeans, reverse_mapping=True)
    print("Wrote:", kmean_outfile, "| rows:", len(df_kmeans))
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.8, min_samples=20).fit(X_all)
    df_dbscan = df.copy()
    df_dbscan["db_cluster"] = dbscan.labels_.astype("int16")  # -1 = noise
    print("DBSCAN cluster counts (includes -1 for noise):")
    print(df_dbscan["db_cluster"].value_counts().sort_index())
    print(df_dbscan.head())
    df_dbscan["db_cluster"].unique()
    db_counts = (
        df_dbscan["db_cluster"]
        .value_counts(dropna=False)
        .sort_index()
    )
    print(db_counts)
    db_outfile = out_dbscan
    with Open(out_dbscan, mode='w') as o:
        o.write(df_dbscan, reverse_mapping=True)
    print("Wrote:", db_outfile, "| rows:", len(df_dbscan))

def run_xgb_failure_classification_sweeps(
        input_path_5_KM: Address, 
        input_path_5_DBSCAN: Address, 
        output_path_5_KM: Address, 
        output_path_5_DBSCAN: Address
    ) -> None:
    """
    Run XGBoost hyperparameter sweeps for failure classification on 
    clustered data.

    This function orchestrates end-to-end XGBoost training and 
    evaluation for failure classification using two pre-clustered 
    datasets: one derived from K-Means clustering and one from DBSCAN 
    clustering. Each dataset is loaded, split chronologically into 
    training and test sets, and passed to a grid-based XGBoost 
    hyperparameter sweep.

    The function is designed as a procedural pipeline step and produces
    side effects only (console output and result files). No values are 
    returned.

    High-level steps:
    - Load K-Means-clustered and DBSCAN-clustered datasets from disk
    - Prepare feature matrices and binary failure labels
    - Perform a chronological train-test split (no shuffling)
    - Run resumable XGBoost hyperparameter sweeps for each dataset
    - Persist sweep results to CSV files

    Parameters
    ----------
    input_path_5_KM : Address
        Path to the input dataset augmented with K-Means cluster labels.
    input_path_5_DBSCAN : Address
        Path to the input dataset augmented with DBSCAN cluster labels.
    output_path_5_KM : Address
        Path to the CSV file where sweep results for the K-Means dataset
        will be written.
    output_path_5_DBSCAN : Address
        Path to the CSV file where sweep results for the DBSCAN dataset
        will be written.

    Returns
    -------
    None
        Runs training and evaluation pipelines and writes sweep results
        to disk.
    """
    print("\n=== XGBoost Classifier ===\n")
    # 20% test chronological split by setting shuffle=False 
    TEST_SIZE = 0.2    
    # Run sweep for KMeans data
    print("\n--- KMeans data ---")
    df_km = load_pandas(input_path_5_KM)
    drop_cols: List[str] = ["fail0", "failure", *KEYS]
    X_km = df_km.drop(columns=drop_cols, errors="ignore")
    y_km = df_km["failure"].astype(int)
    X_train, X_test, y_train, y_test = cast(
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
        train_test_split(
            X_km, y_km, test_size=TEST_SIZE, shuffle=False
        )
    )
    run_xgb_sweep(
        X_train, 
        y_train, 
        X_test, 
        y_test,
        input_path_5_KM, 
        output_path_5_KM
    )
    # run sweep for DBSCAN data
    print("\n--- DBSCAN data ---")
    df_db = load_pandas(input_path_5_DBSCAN)
    X_db = df_db.drop(columns=drop_cols, errors="ignore")
    y_db = df_db["failure"].astype(int)

    X_train, X_test, y_train, y_test = cast(
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
        train_test_split(
            X_db, y_db, test_size=0.2, shuffle=False
        )
    )
    run_xgb_sweep(
        X_train, 
        y_train, 
        X_test, 
        y_test,
        input_path_5_DBSCAN, 
        output_path_5_DBSCAN
    )

def run_rf_failure_classification_sweeps(
        input_path_km: Address, 
        input_path_db: Address, 
        output_csv_km: Address, 
        output_csv_db: Address
    ) -> None:
    """
    Run Random Forest hyperparameter sweeps for failure classification
    on clustered datasets.

    This function orchestrates Random Forest-based failure 
    classification experiments on two pre-clustered datasets: one 
    augmented with K-Means cluster labels and one augmented with DBSCAN 
    cluster labels. Each dataset is loaded from disk, split 
    chronologically into training and test sets, and passed to a 
    resumable Random Forest hyperparameter sweep.

    The function is intended as a procedural pipeline step and produces
    side effects only (console output and result CSV files).

    High-level steps:
    - Load K-Means-clustered and DBSCAN-clustered datasets
    - Prepare feature matrices and binary failure labels
    - Perform a chronological train-test split (no shuffling)
    - Run Random Forest hyperparameter sweeps for each dataset
    - Persist sweep results to CSV files

    Parameters
    ----------
    input_path_km : Address
        Path to the input dataset augmented with K-Means cluster labels.
    input_path_db : Address
        Path to the input dataset augmented with DBSCAN cluster labels.
    output_csv_km : Address
        Path to the CSV file where sweep results for the K-Means dataset
        will be written.
    output_csv_db : Address
        Path to the CSV file where sweep results for the DBSCAN dataset
        will be written.

    Returns
    -------
    None
        Runs training and evaluation pipelines and writes sweep results
        to disk.
    """
    print("\n=== Random Forest Classifier ===\n")
    TEST_SIZE = 0.2

    drop_cols = ["fail0", "failure", *KEYS]
    # KM data
    print("\n--- KMeans data ---")
    df_km = load_pandas(input_path_km)
    X_km = df_km.drop(columns=drop_cols, errors="ignore")
    y_km = df_km["failure"].astype(int)
    X_train, X_test, y_train, y_test = cast(
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
        train_test_split(
            X_km, y_km, test_size=TEST_SIZE, shuffle=False
        )
    )
    run_rf_sweep(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        input_path_km, 
        output_csv_km
    )

    # DBSCAN data
    print("\n--- DBSCAN data ---")
    df_db = load_pandas(input_path_db)
    X_db = df_db.drop(columns=drop_cols, errors="ignore")
    y_db = df_db["failure"].astype(int)

    X_train, X_test, y_train, y_test = cast(
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],  
        train_test_split(
            X_db, y_db, test_size=TEST_SIZE, shuffle=False
        )
    )
    run_rf_sweep(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        input_path_db, 
        output_csv_db
    )

def main():
    """
    Run the end-to-end failure detection and modeling pipeline.

    This function serves as the entry point for the full workflow, 
    including data filtering, feature engineering, failure labeling, 
    clustering, and supervised model training. It orchestrates 
    sequential pipeline stages and writes intermediate and final 
    artifacts to disk.
    """
    # Define plotting theme
    use_dark_theme()
    # Inverter filter
    output_path_1 = DATA_ROOT / "pdm_1.parquet"
    filter_for_device(RAW_DATA_ROOT, output_path_1)
    # Irradiance filter
    output_path_2 = DATA_ROOT / "pdm_2.parquet"
    filter_for_daylight(output_path_1, output_path_2)
    # Label failures and Feature Engineering
    output_path_3 = DATA_ROOT / "pdm_labeled.parquet"
    label_timeseries_dataset(output_path_2, output_path_3, plot=True)
    # Clustering
    output_path_4_KM = DATA_ROOT / 'pdm_labeledDataKmeans.parquet'
    output_path_4_DBSCAN = DATA_ROOT / 'pdm_labeledDataDBSCAN.parquet'
    cluster(output_path_3, output_path_4_KM, output_path_4_DBSCAN)
    print(f"Pipeline execution ready.")
    # XGBoost Model Training
    output_path_5_KM = DATA_ROOT / 'pdm_XGB_model_km.csv'
    output_path_5_DBSCAN = DATA_ROOT / 'pdm_XGB_model_dbscan.csv'
    print(
        f'\nmodel ready for training on inputs:'
        f'\nKMeans data: {output_path_4_KM} and,'
        f'\nDBSCAN data: {output_path_4_DBSCAN}\n'
        '\nOutputs:\nKMeans model results:' 
        f'{output_path_5_KM} and,'
        f'\nDBSCAN model results: {output_path_5_DBSCAN}\n'
    )
    run_xgb_failure_classification_sweeps(
        output_path_4_KM, 
        output_path_4_DBSCAN, 
        output_path_5_KM, 
        output_path_5_DBSCAN
    )
    # Random Forest 
    output_path_6_KM = DATA_ROOT / 'pdm_RF_model_km.csv'
    output_path_6_DBSCAN = DATA_ROOT / 'pdm_RF_model_dbscan.csv'
    
    print(
        f'\nRandom Forest model ready for training on inputs:'
        f'\nKMeans data: {output_path_5_KM} and,'
        f'\nDBSCAN data: {output_path_5_DBSCAN}\n\n'
        'Outputs:\nKMeans model results:' 
        f'{output_path_6_KM} and,'
        f'\nDBSCAN model results: {output_path_6_DBSCAN}\n')
    
    run_rf_failure_classification_sweeps(
        output_path_5_KM, 
        output_path_5_DBSCAN, 
        output_path_6_KM, 
        output_path_6_DBSCAN
    )

if __name__ == "__main__":
    main()
