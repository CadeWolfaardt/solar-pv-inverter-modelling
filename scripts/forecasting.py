# stdlib
import argparse
from pathlib import Path
from collections import defaultdict
from typing import (
    Literal, 
    Dict, 
    List, 
    Tuple, 
    Optional, 
    Union, 
    Iterable,
    cast
)
# thirdpartylib
import polars as pl
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.api import VAR # pyright: ignore[reportMissingTypeStubs]
from statsmodels.tsa.vector_ar.var_model import ( # pyright: ignore
    VARResultsWrapper
)
from statsmodels.tsa.statespace.sarimax import ( # pyright: ignore
    SARIMAX,
    SARIMAXResultsWrapper
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from chronos import ChronosPipeline # pyright: ignore[reportMissingTypeStubs]
# projectlib
from pv_inverter_modeling.data.schemas import Metric, Column
from pv_inverter_modeling.utils.typing import (
    Address, 
    Verbosity, 
    Field,
    ForecastTrainingResult,
    ModelRegistryEntry
)
from pv_inverter_modeling.utils.paths import validate_address
from pv_inverter_modeling.utils.logging import Logger
from pv_inverter_modeling.data.loaders import load_lazyframe
from pv_inverter_modeling.config.private_map import REVERSE_ENTITY_MAP
from pv_inverter_modeling.visualization.timeseries import use_dark_theme
from pv_inverter_modeling.models.io import save_model
from pv_inverter_modeling.evaluation.metrics import strict_r2, safe_mape
from pv_inverter_modeling.models.forecasting import LSTMRegressor
from pv_inverter_modeling.config.env import DATA_ROOT

def map_cols(cols: Iterable[Field]) -> List[str]:
    return [REVERSE_ENTITY_MAP[c] for c in cols]

# Constants
TARGET = Metric.AC_POWER
TARGET_FALLBACK = Metric.DC_POWER
FEATURES = {
    Metric.GHI_IRRADIANCE,
    Metric.POA_IRRADIANCE,
    Metric.REAR_IRRADIANCE,
    Metric.HUMIDITY,
    Metric.WIND_SPEED,
    Metric.WIND_DIRECTION,
    Metric.AMB_TEMP,
    Metric.PV_TEMP,
    Metric.TRACKER_ANGLE,
    Metric.TRACKER_ANGLE_SETPOINT,
    Metric.DC_CURRENT,
    Metric.DC_VOLTAGE,
}
FEATS = map_cols(FEATURES)
USECOLS = [
    Column.TIMESTAMP,
    Column.TYPE,
    Column.METRIC,
    Column.VALUE,
]

def skip_model_training(reason: str) -> Tuple[None, None]:
    """
    Log a message explaining why a model training step was skipped and
    return a sentinel result that allows the training pipeline to
    continue execution.

    This helper is intended for use in model-training orchestration
    code where individual model failures should not halt the entire
    experiment or pipeline.

    Parameters
    ----------
    reason : str
        Human-readable explanation for why the model training was 
        skipped.

    Returns
    -------
    Tuple[None, None]
        Sentinel return value indicating that no metrics and no model
        were produced.
    """
    print(reason)
    return None, None

def slugify(value: str) -> str:
    """
    Convert an arbitrary string into a filesystem- and identifier-safe 
    slug.

    Rules:
    - Alphanumeric characters are preserved
    - '-' and '_' are preserved
    - Whitespace and path separators are converted to '-'
    - All other characters are converted to '_'
    - Leading/trailing separators are stripped
    - Returns 'unknown' if the result is empty
    """
    safe = (
        ch if ch.isalnum() or ch in "-_"
        else "-" if ch.isspace() or ch in "/\\:"
        else "_"
        for ch in value
    )
    slug = "".join(safe).strip("-_")
    return slug or "unknown"

def plot_series_comparison(
        series: Dict[str, pd.Series],
        *,
        title: str,
        ylabel: str,
        ax: Optional[Axes] = None,
        alpha: float = 0.8,
    ) -> Axes:
    """
    Plot multiple aligned time-series on a shared datetime axis.

    This helper is intended for model evaluation and forecasting
    diagnostics (e.g. train vs test vs prediction), where each
    series already represents an aggregated signal indexed by time.

    Parameters
    ----------
    series : dict[str, pandas.Series]
        Mapping from label to time-indexed series. All series must
        share compatible datetime indices.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. A new one is created if None.
    alpha : float, default 0.8
        Base transparency applied to all lines.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the rendered plot.
    """
    use_dark_theme()
    if ax is None:
        _, ax = plt.subplots( # pyright: ignore[reportUnknownMemberType]
            figsize=(12, 5)
        )

    for label, s in series.items():
        ax.plot( # pyright: ignore[reportUnknownMemberType]
            s.index,
            s.to_numpy(),
            label=label,
            alpha=alpha,
        )

    ax.set_title(title) # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("UTC time") # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel(ylabel) # pyright: ignore[reportUnknownMemberType]
    ax.legend() # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout() # pyright: ignore[reportUnknownMemberType]

    return ax

def clean_data(source: Address) -> pl.LazyFrame:
    """
    Load and sanitize raw telemetry data into a normalized Polars 
    LazyFrame.

    This function performs minimal, deterministic preprocessing required 
    by downstream aggregation pipelines:
    - Coerces timestamps to UTC-aware `Datetime[us]`
    - Coerces metric values to floating point
    - Drops rows missing required structural fields

    No aggregation or feature engineering is performed here; the output
    preserves the original row-level granularity while enforcing 
    consistent temporal and numeric semantics.

    Parameters
    ----------
    source : Address
        Path or string pointing to the telemetry data source (e.g. 
        Parquet, IPC, or similar Polars-supported format).

    Returns
    -------
    pl.LazyFrame
        Cleaned lazy dataframe with UTC timestamps and numeric values, 
        suitable for downstream aggregation and streaming execution.
    """
    lf = (
        load_lazyframe(source)
        .with_columns(
            # Enforce UTC-aware timestamps for correct temporal 
            # semantics
            pl.col(Column.TIMESTAMP)
                .cast(pl.Datetime("us"), strict=False)
                .dt.replace_time_zone("UTC"),
            # Coerce numeric values; invalid parses become null
            pl.col(Column.VALUE).cast(pl.Float64, strict=False),
        )
        # Drop rows missing any required fields
        .drop_nulls(
            subset=[Column.TIMESTAMP, Column.METRIC, Column.VALUE, Column.TYPE]
        )
    )
    return lf

def reduce_feature_lists(
        ts_dict: Dict[pd.Timestamp, List[float]]
    ) -> pd.Series:
    """
    Reduce per-timestamp lists of numeric values to a single scalar
    using nanmean.

    Parameters
    ----------
    ts_dict : Dict[pd.Timestamp, List[float]]
        Mapping from timestamp to list of observed values.

    Returns
    -------
    pd.Series
        Series indexed by timestamp with reduced scalar values.
    """
    ts_sorted = sorted(ts_dict.keys())
    values = [
        np.nan if not ts_dict[ts]
        else float(np.nanmean(ts_dict[ts]))
        for ts in ts_sorted
    ]
    s = pd.Series(values, index=pd.to_datetime(ts_sorted))
    s.index.name = Column.TIMESTAMP
    return s

def add_cyclical_time_features(x: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encodings for time-of-day and wind direction.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix indexed by UTC timestamps.

    Returns
    -------
    pd.DataFrame
        Feature matrix with added cyclical features.
    """
    ind = pd.DatetimeIndex(x.index)
    minutes = (ind.hour * 60 + ind.minute).astype(float)
    day_minutes = 24.0 * 60.0

    x = x.copy()
    x["tod_sin"] = np.sin(2 * np.pi * minutes / day_minutes)
    x["tod_cos"] = np.cos(2 * np.pi * minutes / day_minutes)

    if Metric.WIND_DIRECTION in x.columns:
        radians = np.deg2rad(x[Metric.WIND_DIRECTION].astype(float))
        x["wind_dir_sin"] = np.sin(radians)
        x["wind_dir_cos"] = np.cos(radians)

    return x

def apply_daylight_filter(
        x: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter timestamps to daylight-relevant periods using target
    and irradiance signals.

    Daylight is defined as:
      - target > 0 OR
      - sum of irradiance features > 0

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Filtered (x, y).
    """
    irr_cols = map_cols([Metric.GHI_IRRADIANCE,
        Metric.POA_IRRADIANCE,
        Metric.REAR_IRRADIANCE,
    ])
    irradiance_cols = list(set(irr_cols).intersection(x.columns))

    if not irradiance_cols:
        return x, y

    mask = (
        (y > 0)
        | (
            x[irradiance_cols]
            .fillna(0) # pyright: ignore[reportUnknownMemberType]
            .sum(axis=1) > 0
        )
    )
    return x.loc[mask], y.loc[mask]

def align_features_to_target(
        x: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
    """
    Align feature matrix to the target timeline and fill gaps.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to y.index.
    """
    return x.reindex(y.index).ffill().bfill()

def read_aggregated_timeseries(
        source: Address
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aggregate raw telemetry into a site-level feature matrix and target
    time series suitable for forecasting or downstream modeling.

    This function performs the following high-level steps:
    1. Loads and cleans raw telemetry using a shared preprocessing 
       routine
    2. Aggregates the target metric (site-level AC power) by timestamp
    3. Aggregates selected exogenous metrics into per-timestamp feature 
       values
    4. Aligns features to the target timeline and fills gaps
    5. Engineers cyclical time-of-day and wind-direction features
    6. Filters the resulting series to daylight-relevant timestamps

    The returned feature matrix and target series share a common UTC
    DatetimeIndex and are safe to use directly in time-series models.

    Parameters
    ----------
    source : Address
        Path or string pointing to the telemetry data source (e.g. 
        Parquet, IPC, or similar Polars-supported format).

    Returns
    -------
    x : pd.DataFrame
        Feature matrix indexed by UTC timestamp, including engineered
        temporal and directional features.

    y : pd.Series
        Target time series (site-level AC power) indexed by UTC 
        timestamp.

    Raises
    ------
    RuntimeError
        If no valid target values or no exogenous features are found.
    """
    # Load and normalize raw telemetry (lazy, streaming-safe)
    clean = clean_data(source)
    # Aggregate target metric (site-level AC power)
    y_df = (
        clean
        .filter(
            (pl.col(Column.METRIC) == REVERSE_ENTITY_MAP[TARGET])
            & pl.col(Column.TYPE).is_in(["Inverter", "Meter"])
        )
        .group_by(Column.TIMESTAMP)
        .agg(pl.sum(Column.VALUE).alias("value_sum"))
        .collect(engine="streaming")
        .to_pandas()
    )
    if y_df.empty:
        raise RuntimeError("No target values found.")
    # Construct target series with a UTC DatetimeIndex
    y = pd.Series(
        y_df["value_sum"].astype(float).values,
        index=pd.to_datetime(y_df[Column.TIMESTAMP], utc=True),
        name="y_site_ac_power",
    )
    y.index.name = Column.TIMESTAMP
    # Aggregate exogenous metrics into per-timestamp value lists
    x_df = (
        clean
        .filter(pl.col(Column.METRIC).is_in(FEATS))
        .group_by([Column.METRIC, Column.TIMESTAMP])
        .agg(pl.col(Column.VALUE).alias("values"))
        .collect(engine="streaming")
        .to_pandas()
    )
    # Accumulate feature values as lists prior to reduction
    features: Dict[str, Dict[pd.Timestamp, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for metric, ts, values in x_df.itertuples(index=False):
        features[str(metric)][pd.to_datetime(ts, utc=True)].extend(
            float(v) for v in values
        )
    # Reduce per-timestamp value lists and assemble feature matrix
    x = pd.DataFrame({
        m: reduce_feature_lists(ts_dict)
        for m, ts_dict in features.items()
        if ts_dict
    })

    if x.empty:
        raise RuntimeError("No exogenous features found.")
    # Align features to target timeline and engineer derived features
    x = align_features_to_target(x, y)
    x = add_cyclical_time_features(x)
    # Filter to daylight-relevant timestamps
    x, y = apply_daylight_filter(x, y)

    return x, y

def read_aggregated_timeseries_by_device_type(
        source: Address
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Aggregate telemetry into per-device-type feature matrices and target
    time series suitable for device-specific modeling.

    For each device type present in the data, this function:
    1. Aggregates a primary target metric by timestamp
    2. Falls back to a secondary target metric if the primary is 
       unavailable
    3. Aggregates selected exogenous metrics into per-timestamp feature 
       values
    4. Aligns features to the target timeline and fills gaps
    5. Engineers cyclical time-of-day and wind-direction features
    6. Filters to daylight-relevant timestamps
    7. Discards device types with insufficient target history

    Each device type is treated independently and may have its own
    timeline and feature availability.

    Parameters
    ----------
    source : Address
        Path or string pointing to the telemetry data source (e.g. 
        Parquet, IPC, or similar Polars-supported format).

    Returns
    -------
    Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Mapping from device type to a `(X, y)` tuple where:
        - `X` is a feature matrix indexed by UTC timestamp
        - `y` is the corresponding target time series

        Device types with no valid target data or insufficient history
        are omitted from the output.

    Raises
    ------
    None
        Device types without usable data are silently skipped.
    """
    # Load and normalize raw telemetry (lazy, streaming-safe)
    clean = clean_data(source)
    # Aggregate primary and fallback target metrics by device type
    y_ac = (
        clean
        .filter(pl.col(Column.METRIC) == REVERSE_ENTITY_MAP[TARGET])
        .group_by([Column.TYPE, Column.TIMESTAMP])
        .agg(pl.sum(Column.VALUE).alias("value_sum"))
        .collect(engine="streaming")
        .to_pandas()
    )
    y_dc = (
        clean
        .filter(pl.col(Column.METRIC) == REVERSE_ENTITY_MAP[TARGET_FALLBACK])
        .group_by([Column.TYPE, Column.TIMESTAMP])
        .agg(pl.sum(Column.VALUE).alias("value_sum"))
        .collect(engine="streaming")
        .to_pandas()
    )
    # Aggregate exogenous metrics into per-timestamp value lists
    x_df = (
        clean
        .filter(pl.col(Column.METRIC).is_in(FEATS))
        .group_by([Column.TYPE, Column.METRIC, Column.TIMESTAMP])
        .agg(pl.col(Column.VALUE).alias("values"))
        .collect(engine="streaming")
        .to_pandas()
    )
    # Output container: one (X, y) pair per device type
    out: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    # Process each device type independently
    for dt in set(y_ac[Column.TYPE]) | set(y_dc[Column.TYPE]):
        dt: str
        # Prefer primary target; fall back if unavailable
        y_src =  y_ac[y_ac[Column.TYPE] == dt]
        if y_src.empty:
            y_src = y_dc[y_dc[Column.TYPE] == dt]
        if y_src.empty:
            continue
        # Construct target series with a UTC DatetimeIndex
        y = pd.Series(
            y_src["value_sum"].astype(float).values,
            index=pd.to_datetime(y_src[Column.TIMESTAMP], utc=True),
            name="y_target",
        )
        y.index.name = Column.TIMESTAMP
        # Accumulate exogenous feature values for this device type
        features: Dict[str, Dict[pd.Timestamp, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        sub = x_df[x_df[Column.TYPE] == dt]
        for _, metric, ts, values in sub.itertuples(index=False):
            features[str(metric)][pd.to_datetime(ts, utc=True)].extend(
                float(v) for v in values
            )
        if not features:
            continue
        # Reduce per-timestamp value lists and assemble feature matrix
        x = pd.DataFrame({
            m: reduce_feature_lists(ts_dict)
            for m, ts_dict in features.items()
            if ts_dict
        })
        # Align, engineer derived features, and filter to daylight
        x = align_features_to_target(x, y)
        x = add_cyclical_time_features(x)
        x, y = apply_daylight_filter(x, y)
        # Enforce a minimum history length for downstream modeling
        if len(y) >= 60:
            out[dt] = (x, y)

    return out

def decompose_and_forecast(
        X: pd.DataFrame, 
        y: pd.Series, 
        outputs_dir: Path
    ) -> Tuple[Dict[str, float], LinearRegression, XGBRegressor]:
    """
    Decompose a time series into trend and stable components, fit 
    separate models to each component, and produce an out-of-sample
    forecast with evaluation artifacts.

    This function implements a simple two-stage forecasting strategy:
    1. A smooth trend component is extracted from the target series 
       using a rolling mean and modeled with linear regression over 
       time.
    2. The residual (stable) component is modeled using a 
       gradient-boosted tree regressor conditioned on exogenous 
       features.
    3. The two predicted components are recombined to form the final
       forecast.

    The data is split temporally into a training and test segment
    (70% / 30%), metrics are computed on the test set, and both 
    numerical outputs and visual diagnostics are written to disk.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix indexed by timestamp. Must be aligned one-to-one
        with the target series.
    y : pandas.Series
        Target time series indexed by timestamp.
    outputs_dir : pathlib.Path
        Directory in which forecast outputs, plots, and metrics will be
        written.

    Returns
    -------
    metrics : dict[str, float]
        Dictionary containing evaluation metrics computed on the test 
        set (MAE, RMSE, MAPE, R²).
    trend_model : sklearn.linear_model.LinearRegression
        Fitted linear regression model used to predict the trend 
        component.
    stable_model : xgboost.XGBRegressor
        Fitted gradient-boosted tree model used to predict the stable
        residual component.

    Raises
    ------
    RuntimeError
        If fewer than 60 observations are available after filtering.
    AssertionError
        If the feature matrix and target series are not index-aligned.
    """
    # Validate output directory and ensure index alignment
    outputs_dir = validate_address(outputs_dir)
    assert(X.index == y.index).all()
    # Require a minimum amount of data to ensure a meaningful train/test 
    # split
    n = len(y)
    if n < 60: 
        raise RuntimeError(
            "Insufficient data rows (< 60) after filtering for daylight."
        )
    # Split data temporally to preserve causal ordering
    split_idx = int(n*0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # Estimate a smooth trend using a rolling mean whose window adapts
    # to the amount of available training data
    window = max(12, int(len(y_train)*0.05))
    trend_train = (
        y_train
        .rolling(window=window, min_periods = max(3, window // 3))
        .mean()
        .ffill()
        .bfill()
    )
    # Define the stable component as the residual after trend removal
    stable_train = y_train - trend_train
    # Fit a simple linear model to the trend as a function of time index
    t_train = np.arange(len(trend_train)).reshape(-1,1)
    lin = LinearRegression()
    lin.fit(t_train, trend_train.to_numpy())
    # Fit a gradient-boosted tree model to the stable component using
    # exogenous features
    stable_model: XGBRegressor = XGBRegressor(
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        reg_lambda = 1.0,
        n_jobs = 4,
    )
    stable_model.fit(X_train,stable_train)
    # Predict both components on the test horizon and recombine them
    t_test = np.arange(
        len(trend_train), 
        len(trend_train) + len(X_test)
    ).reshape(-1,1)
    trend_test_pred = pd.Series(
        lin.predict(t_test), 
        index=X_test.index
    )
    stable_test_pred = pd.Series(
        stable_model.predict(X_test), 
        index=X_test.index
    )
    y_pred_test = trend_test_pred + stable_test_pred
    # Compute standard regression metrics on the test set
    mae = mean_absolute_error(y_test,y_pred_test)
    rmse = root_mean_squared_error(y_test, y_pred_test)
    mape = safe_mape(y_test, y_pred_test)
    r2 = strict_r2(y_test, y_pred_test)
    # Persist forecast outputs for downstream inspection
    df_out = pd.DataFrame({
        "y_actual" : y_test,
        "y_pred" : y_pred_test,
        "trend_pred": trend_test_pred,
        "stable_pred": stable_test_pred,
    })
    df_out.index.name = Column.TIMESTAMP
    df_out.to_csv(outputs_dir / "site_forecast.csv")
    # Generate and save a visual comparison of actual vs forecasted 
    # values
    fig, ax = plt.subplots( # pyright: ignore[reportUnknownMemberType]
        figsize=(12, 5)
    )
    ax = plot_series_comparison(
        {
            "train actual": y_train,
            "test actual": y_test,
            "test forecast": y_pred_test,
        },
        title="Site AC Power: Actual vs Forecast",
        ylabel="AC Power",
        ax=ax
    )
    fig.savefig( # pyright: ignore[reportUnknownMemberType]
        outputs_dir / "site_forecast.png",
        dpi=150,
    )
    plt.close()
    # Write evaluation metrics to disk in a simple text format
    metrics = {"MAE": mae,"RMSE": rmse, "MAPE": mape, "R2": r2}
    with open(outputs_dir / "metrics.txt", "w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    return metrics, lin, stable_model

def train_and_forecast_var(
        X: pd.DataFrame,
        y: pd.Series, 
        outputs_dir: Path, 
        maxlags: int = 8
    ) -> ForecastTrainingResult[VARResultsWrapper]:
    """
    Train a Vector Autoregression (VAR) model and generate 
    one-step-ahead rolling forecasts on a holdout set.

    This function fits a VAR model using the target series ``y`` 
    together with a selected subset of exogenous variables from ``X``. 
    The data is split chronologically into a training set (first 70%) 
    and a test set (remaining 30%). Forecasts are generated iteratively 
    in a rolling fashion, where each prediction is fed back into the 
    model history.

    Model performance is evaluated on the test set using MAE, RMSE,
    safe MAPE, and strict R². Forecasts, plots, and metrics are written
    to disk under ``outputs_dir / "var"``.

    Parameters
    ----------
    X : pd.DataFrame
        Exogenous feature matrix indexed by timestamp.
    y : pd.Series
        Target time series indexed by timestamp. Must align exactly with 
        ``X``.
    outputs_dir : Path
        Base directory where model outputs, plots, and metrics are 
        saved.
    maxlags : int, default=8
        Maximum number of lags to consider during VAR model selection.
        The final lag order is chosen via AIC.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing evaluation metrics computed on the test 
        set: ``MAE``, ``RMSE``, ``MAPE``, and ``R2``. Returns ``None`` 
        if the model is skipped due to insufficient data.
    model_result : Optional[VARResultsWrapper]
        Fitted VAR results object. Returns ``None`` if the model is 
        skipped due to insufficient data.

    Notes
    -----
    - This function assumes a univariate target with multivariate 
      inputs.
    - Missing values are forward- and backward-filled prior to modeling.
    - Forecasting is performed in an autoregressive rolling manner.
    - R² is computed using a strict definition and may return NaN for
      near-zero variance targets.
    """
    # Ensure target and feature indices are perfectly aligned
    assert(X.index == y.index).all()
    # Candidate exogenous variables to include in the VAR model
    candidate_cols = map_cols([
        Metric.POA_IRRADIANCE,
        Metric.GHI_IRRADIANCE,
        Metric.DC_CURRENT,
        Metric.DC_VOLTAGE,
        Metric.AMB_TEMP,
    ])
    # Use only columns that are actually present in X
    used_cols = [c for c in candidate_cols if c in X.columns]
    # Combine target and features into a single DataFrame
    # Fill missing values to ensure VAR compatibility
    df = pd.concat(
        [y.rename("y"), X[used_cols]],
        axis=1
    ).ffill().bfill()
    n = len(df)
    # VAR models require sufficient observations to estimate lag 
    # structure
    if n < 60:
        return skip_model_training(
            "Insufficient data for VAR after filtering."
        )
     # Chronological train/test split (no shuffling)
    split_idx = int(n * 0.7)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    # Fit VAR model and select lag order using AIC
    model = VAR(train_df)
    res = model.fit( # pyright: ignore[reportUnknownMemberType]
        maxlags=maxlags, 
        ic="aic"
    )
    # Selected lag order
    lag = res.k_ar
    # Initialize rolling forecast history with training data
    history = train_df.values.tolist()
    preds: List[float] = []
    # One-step-ahead rolling forecast
    for _ in range(len(test_df)):
        input_arr = np.asarray(history[-lag:])
        forecast_vec = res.forecast(y = input_arr, steps = 1)[0]
         # First element corresponds to the target variable
        preds.append(float(forecast_vec[0]))
        # Append full forecast vector to history for next step
        history.append(forecast_vec.tolist())
    # Align predictions with test index
    y_test = test_df["y"]
    y_pred = pd.Series(preds, index=y_test.index)
    # Compute evaluation metrics
    mae = mean_absolute_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    r2 = strict_r2(y_test, y_pred)
    # Prepare output directory
    out_dir = validate_address(
        outputs_dir / "var",
        mkdir=True
    )
    # Save forecasts to CSV
    df_out = pd.DataFrame({"y_actual": y_test, "y_pred": y_pred})
    df_out.index.name = Column.TIMESTAMP
    df_out.to_csv(out_dir / "site_forecast_var.csv")
    # Plot train/test actuals and VAR forecast
    fig, ax = plt.subplots( # pyright: ignore[reportUnknownMemberType]
        figsize=(12, 5)
    )
    ax = plot_series_comparison(
        {
            "train actual": train_df["y"],
            "test actual": y_test,
            "test forecast (VAR)": y_pred,
        },
        title="Site AC Power: Actual vs Forecast (VAR)",
        ylabel="AC Power",
        ax=ax
    )
    fig.savefig( # pyright: ignore[reportUnknownMemberType]
        out_dir / "site_forecast_var.png",
        dpi=150,
    )
    plt.close()
    # Persist metrics to disk
    metrics = {"MAE": mae,"RMSE": rmse, "MAPE": mape, "R2": r2}
    with open(out_dir / "metrics.txt", "w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    return metrics, res

def train_and_forecast_lstm(
        X: pd.DataFrame,
        y: pd.Series,
        outputs_dir: Path,
        seq_len: int = 12,
        epochs: int = 40,
        batch_size: int = 64,
        hidden_size: int = 64
    ) -> ForecastTrainingResult[LSTMRegressor]:
    """
    Train an LSTM regressor and generate sequence-to-one forecasts on a
    chronological holdout set.

    This function constructs rolling input sequences of length 
    ``seq_len`` using the target variable and selected exogenous 
    features, trains an LSTM-based regression model on the first 70% of 
    the data, and evaluates performance on the remaining 30%.

    Features and target values are standardized using statistics 
    computed from the training split only. Predictions are 
    inverse-transformed back to the original target scale before 
    evaluation.

    Forecasts, plots, and evaluation metrics are written to disk under
    ``outputs_dir / "lstm"``.

    Parameters
    ----------
    X : pd.DataFrame
        Exogenous feature matrix indexed by timestamp.
    y : pd.Series
        Target time series indexed by timestamp. Must align with ``X``.
    outputs_dir : Path
        Base directory for saving model outputs, plots, and metrics.
    seq_len : int, default=12
        Length of the rolling input sequence (number of past time 
        steps).
    epochs : int, default=40
        Number of training epochs.
    batch_size : int, default=64
        Mini-batch size used during training.
    hidden_size : int, default=64
        Number of hidden units in the LSTM layer.

    Returns
    -------
    metrics : Optional[Dict[str, float]]
        Dictionary containing evaluation metrics computed on the test 
        set: ``MAE``, ``RMSE``, ``MAPE``, and ``R2``. Returns ``None`` 
        if training cannot proceed due to insufficient data.
    model : Optional[LSTMRegressor]
        Trained LSTM model. Returns ``None`` if training cannot proceed 
        due to insufficient data.

    Notes
    -----
    - Forecasting is performed in a direct (one-step-ahead) manner.
    - Sequences include the lagged target variable as an input feature.
    - R² is computed using a strict definition and may return NaN for
      near-zero variance targets.
    """
    # Feature selection and preprocessing
    candidate_cols = map_cols([
        Metric.POA_IRRADIANCE,
        Metric.GHI_IRRADIANCE,
        Metric.DC_CURRENT,
        Metric.DC_VOLTAGE,
        Metric.AMB_TEMP,
        Metric.HUMIDITY,
        Metric.TRACKER_ANGLE,
    ]) + ["tod_sin", "tod_cos"]
    # Use only features that exist in X
    used_cols = [c for c in candidate_cols if c in X.columns]
    # Combine target and features into a single aligned DataFrame
    df = pd.concat(
        [y.rename("y"), X[used_cols]],
        axis=1
    ).ffill().bfill()
    n = len(df)
    if n < seq_len + 30:
        return skip_model_training(
            "Insufficient data for LSTM after filtering."
        )
    # Chronological train/test split
    split_idx = int(n * 0.7)
    # Scaling (fit on training data only)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_vals = df[used_cols].values
    y_vals = df[["y"]].values
    scaler_X.fit(X_vals[:split_idx])
    scaler_y.fit(y_vals[:split_idx])
    X_scaled = scaler_X.transform(X_vals)
    y_scaled = scaler_y.transform(y_vals)
    # Sequence construction
    list_X: List[np.ndarray] = []
    list_y: List[float] = []
    for i in range(seq_len,n):
        # Each input sequence contains lagged target + lagged features
        list_X.append(np.hstack([y_scaled[i-seq_len:i],X_scaled[i-seq_len:i]]))
        list_y.append(y_scaled[i,0])
    seq_X = np.array(list_X)
    seq_y = np.array(list_y)
    # Map sequence targets back to original indices
    target_indices = np.arange(seq_len, n)
    train_mask = target_indices < split_idx
    test_mask = ~train_mask
    X_train, y_train = seq_X[train_mask], seq_y[train_mask]
    X_test, _ = seq_X[test_mask], seq_y[test_mask]
    # Torch tensors and data loaders
    device = torch.device("cpu")
    X_train_t = torch.tensor(X_train, dtype = torch.float32, device = device)
    y_train_t = torch.tensor(
        y_train, 
        dtype=torch.float32, 
        device= device
    ).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device= device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), 
        batch_size=batch_size, 
        shuffle=True
    )
    # Model training
    input_size = X_train_t.shape[-1]
    model = LSTMRegressor(
        input_size=input_size, 
        hidden_size=hidden_size
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for xb,yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred,yb)
            loss.backward()
            optimizer.step() # pyright: ignore[reportUnknownMemberType]
    # Inference
    model.eval()
    with torch.no_grad():
        y_pred_s: NDArray[np.floating] = (
            model(X_test_t)
            .cpu()
            .numpy()
            .squeeze()
        )
    # Inverse-transform predictions back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).squeeze()
    # Evaluation
    y_actual_full = df["y"].to_numpy()
    y_test = y_actual_full[target_indices[test_mask]]
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    r2 = strict_r2(y_test, y_pred)
    # Outputs and visualization
    out_dir = validate_address(outputs_dir / "lstm", mkdir=True)
    idx = df.index[target_indices[test_mask]]
    df_out = pd.DataFrame({"y_actual": y_test, "y_pred": y_pred})
    df_out.index.name = Column.TIMESTAMP
    df_out.to_csv(out_dir / "site_forecast_lstm.csv")
    # Plot train/test actuals and LSTM forecast
    fig, ax = plt.subplots( # pyright: ignore[reportUnknownMemberType]
        figsize=(12, 5)
    )
    y_test_pd = pd.Series(y_test, index=idx)
    y_pred_pd = pd.Series(y_pred, index=idx)
    ax = plot_series_comparison(
        {
            "train actual": df.iloc[:split_idx]["y"],
            "test actual": y_test_pd,
            "test forecast (LSTM)": y_pred_pd,
        },
        title="Site AC Power: Actual vs Forecast (LSTM)",
        ylabel="AC Power",
        ax=ax
    )
    fig.savefig( # pyright: ignore[reportUnknownMemberType]
        out_dir / "site_forecast_lstm.png",
        dpi=150,
    )
    plt.close()
    # Persist metrics
    metrics = {"MAE": mae,"RMSE": rmse, "MAPE": mape, "R2": r2}
    with open(out_dir / "metrics.txt", "w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    return metrics, model

def train_and_forecast_chronos(
            y: pd.Series,
            outputs_dir: Path,
            model_name: str = "amazon/chronos-t5-mini",
            horizon: Optional[int] = None,
            num_samples: int = 20,
    ) -> ForecastTrainingResult[ChronosPipeline]:
    """
    Train (load) a Chronos foundation model and generate probabilistic
    time-series forecasts on a holdout horizon.

    This function uses a pretrained Chronos model to forecast the final
    portion of the target series ``y``. The model is not fine-tuned; instead,
    it conditions on historical context values and generates multiple
    forecast samples, which are then reduced to a single deterministic
    forecast via averaging.

    The forecast horizon is automatically inferred if not provided and is
    capped to the maximum prediction length supported by the selected
    Chronos model.

    Forecasts, plots, and evaluation metrics are written to disk under
    ``outputs_dir / "chronos"``.

    Parameters
    ----------
    y : pd.Series
        Target time series indexed by timestamp.
    outputs_dir : Path
        Base directory for saving model outputs, plots, and metrics.
    model_name : str, default="amazon/chronos-t5-mini"
        Hugging Face model identifier for the Chronos pipeline.
    horizon : Optional[int], default=None
        Forecast horizon. If ``None``, defaults to the maximum of 12 or
        30% of the series length.
    num_samples : int, default=20
        Number of probabilistic forecast samples generated by Chronos.

    Returns
    -------
    metrics : Optional[Dict[str, float]]
        Dictionary containing evaluation metrics computed on the forecast
        horizon: ``MAE``, ``RMSE``, ``MAPE``, and ``R2``. Returns ``None`` if
        forecasting is skipped or fails.
    model : Optional[ChronosPipeline]
        Loaded Chronos pipeline. Returns ``None`` if forecasting is skipped
        or fails.

    Notes
    -----
    - Chronos is used in inference-only mode (no fine-tuning).
    - Forecast samples are averaged to obtain a point forecast.
    - R² is computed using a strict definition and may return NaN for
      near-zero variance targets.
    """
    # Horizon determination
    if horizon is None:
        horizon = max(12, int(len(y)*0.3))
    try:
        # Load pretrained Chronos pipeline
        pipe = cast(
            ChronosPipeline,
            ChronosPipeline
            .from_pretrained( # pyright: ignore[reportUnknownMemberType]
                model_name,
                device_map = "cpu",
                torch_dtype = torch.float32,
            )
        )
        # Cap horizon to model-supported prediction length (if 
        # specified)
        model_pred_len = getattr(
            pipe.model.config, 
            "prediction_length", 
            None
        )
        if model_pred_len is not None and horizon > int(model_pred_len):
            horizon = int(model_pred_len)
        # Context / target split
        train_len = max(1, len(y) - horizon)
        context_values = y.iloc[:train_len].astype(float).values
        target_index = y.index[train_len:]
        if len(target_index) != horizon:
            # Fallback to last `horizon` timestamps if alignment drifts
            target_index = y.index[-horizon:]
        if horizon <= 0:
            return skip_model_training(
                "[Chronos] Horizon is non-positive "
                "after adjustment; skipping."
            )
        # Forecast generation
        context_tensor = torch.tensor(
            context_values, 
            dtype = torch.float32
        )
        samples = pipe.predict(
            context_tensor,
            prediction_length = horizon,
            num_samples=num_samples,
            temperature = 1.0)
        # Normalize output to a torch.Tensor
        if hasattr(samples, "detach"):
            samples_t = samples
        else:
            samples_t = torch.tensor(samples)
        # Remove optional batch dimension
        if samples_t.ndim == 3:
            samples_t = samples_t.squeeze(0)
        # Reduce probabilistic samples to a point forecast
        if samples_t.ndim == 2:
            preds_t = samples_t.mean(dim=0)
        elif samples_t.ndim == 1:
            preds_t = samples_t
        else:
            return skip_model_training(
                "Unexpected Chronos predictions shape: "
                f"{tuple(samples_t.shape)}"
            )
        preds = preds_t.detach().cpu().numpy()
        y_pred = pd.Series(preds, index=target_index)
    except Exception as exc:
        # Any failure in model loading or forecasting should not
        # halt the broader training pipeline
        return skip_model_training(
            f"[Chronos] Forecast failed: {exc}"
        )
    # Evaluation
    y_test = y.iloc[train_len:]
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    r2 = strict_r2(y_test, y_pred)
    # Outputs and visualization
    out_dir = validate_address(outputs_dir / "chronos", mkdir=True)
    df_out = pd.DataFrame({"y_actual": y_test, "y_pred": y_pred})
    df_out.index.name = Column.TIMESTAMP
    df_out.to_csv(out_dir / "site_forecast_chronos.csv")
    fig, ax = plt.subplots( # pyright: ignore[reportUnknownMemberType]
        figsize=(12, 5)
    )
    ax = plot_series_comparison(
        {
            "train actual": y.iloc[:train_len],
            "test actual": y_test,
            "test forecast (Chronos)": y_pred,
        },
        title="Site AC Power: Actual vs Forecast (Chronos)",
        ylabel="AC Power",
        ax=ax
    )
    fig.savefig( # pyright: ignore[reportUnknownMemberType]
        out_dir / "site_forecast_chronos.png",
        dpi=150,
    )
    plt.close()
    # Persist metrics
    metrics = {"MAE": mae,"RMSE": rmse, "MAPE": mape, "R2": r2}
    with open(out_dir / "metrics.txt", "w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    return metrics, pipe

def train_and_forecast_sarima(
        X: pd.DataFrame,
        y: pd.Series,
        outputs_dir: Path,    
    ) -> ForecastTrainingResult[SARIMAXResultsWrapper]:
    """
    Train a SARIMA model with exogenous regressors and generate 
    forecasts on a chronological holdout set.

    This function fits a SARIMA model to the first 70% of the target 
    series using selected exogenous variables, then forecasts the 
    remaining 30%. Model performance is evaluated using MAE, RMSE, safe 
    MAPE, and strict R².

    Forecasts, plots, and evaluation metrics are written to disk under
    ``outputs_dir / "sarima"``.

    Parameters
    ----------
    X : pd.DataFrame
        Exogenous feature matrix indexed by timestamp.
    y : pd.Series
        Target time series indexed by timestamp. Must align exactly with 
        ``X``.
    outputs_dir : Path
        Base directory for saving model outputs, plots, and metrics.

    Returns
    -------
    metrics : Optional[Dict[str, float]]
        Dictionary containing evaluation metrics computed on the test
        split (chronological holdout) of the target series: ``MAE``,
        ``RMSE``, ``MAPE``, and ``R2``. Returns ``None`` if forecasting 
        is skipped or fails.
    model : Optional[SARIMAXResultsWrapper]
        Fitted SARIMAX results wrapper. Returns ``None`` if forecasting 
        is skipped or fails.
    """
    # Ensure strict alignment between features and target
    assert(X.index == y.index).all()
    n = len(y)
    if n < 60:
        return skip_model_training(
            "Insufficient data rows (<60) for SARIMA."
        )
    split_idx = int(n * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # Select candidate exogenous features and retain only those present
    candidate_cols = map_cols([
        Metric.POA_IRRADIANCE,
        Metric.GHI_IRRADIANCE,
        Metric.DC_VOLTAGE,
        Metric.AMB_TEMP,
        Metric.HUMIDITY,
        Metric.TRACKER_ANGLE,
    ]) + ["tod_sin", "tod_cos"]
    used_cols = [c for c in candidate_cols if c in X.columns]
    # Forward/backward fill to ensure SARIMA compatibility
    X_train_sarima = X_train[used_cols].ffill().bfill()
    X_test_sarima = X_test[used_cols].ffill().bfill()
    y_train_sarima = y_train.rename("y_target")
    # Fixed SARIMA configuration; seasonal period corresponds to daily 
    # cycle
    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, 12*24)
    try:
        model = SARIMAX(
            y_train_sarima,
            exog=X_train_sarima,
            order=order,
            seasonal_order=seasonal_order,
            enforce_statiopnarity=False,
            enforce_invertibility=False,
        )
        model_fit = cast(
            SARIMAXResultsWrapper,
            model.fit(disp=True),  # pyright: ignore[reportUnknownMemberType]
        )
        # Forecast over the test horizon using exogenous inputs
        y_pred = model_fit.forecast(
            steps=len(y_test), 
            exog=X_test_sarima
        )
        y_pred.index = y_test.index

    except Exception as e:
        print(f"SARIMA Model fitting or forecasting failed: {e}")
        metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
        out_dir = validate_address(outputs_dir / "sarima", mkdir=True)
        df_out = pd.DataFrame({
            "y_actual": y_test,
            "y_pred": pd.Series(index=y_test.index)
        })
        df_out.index.name = Column.TIMESTAMP
        df_out.to_csv(out_dir / "site_forecast_sarima.csv")
        
        with open(out_dir / "metrics.txt", "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        
        return metrics, None
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    r2 = strict_r2(y_test, y_pred)

    out_dir = validate_address(outputs_dir / "sarima", mkdir=True)
    df_out = pd.DataFrame({"y_actual":y_test, "y_pred": y_pred})
    df_out.index.name = Column.TIMESTAMP
    df_out.to_csv(out_dir / "site_forecast_sarima.csv")

    fig, ax = plt.subplots( # pyright: ignore[reportUnknownMemberType]
        figsize=(12, 5)
    )
    ax = plot_series_comparison(
        {
            "train actual": y_train,
            "test actual": y_test,
            "test forecast (SARIMA)": y_pred,
        },
        title="Site AC Power: Actual vs Forecast (SARIMA)",
        ylabel="AC Power",
        ax=ax
    )
    fig.savefig( # pyright: ignore[reportUnknownMemberType]
        out_dir / "site_forecast_sarima.png",
        dpi=150,
    )
    plt.close()

    metrics = {"MAE":mae, "RMSE":rmse,"MAPE":mape,"R2":r2}
    with open(out_dir / "metrics.txt","w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    return metrics, model_fit

MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {
    "baseline": {
        "kind": "baseline",
        "fn": decompose_and_forecast,
        "args": lambda X, y, out: (X, y, out),
        "save": [
            ("baseline_linear", lambda res: res[1]),
            ("baseline_stable", lambda res: res[2]),
        ],
    },
    "var": {
        "kind": "forecast",
        "fn": train_and_forecast_var,
        "args": lambda X, y, out: (X, y, out),
        "save": [
            ("var", lambda res: res[1]),
        ],
        "skip_if_none": True,
    },
    "lstm": {
        "kind": "forecast",
        "fn": train_and_forecast_lstm,
        "args": lambda X, y, out: (X, y, out),
        "save": [
            ("lstm", lambda res: res[1]),
        ],
        "skip_if_none": True,
    },
    "chronos": {
        "kind": "forecast",
        "fn": train_and_forecast_chronos,
        "args": lambda X, y, out: (y, out),
        "save": [
            ("chronos", lambda res: res[1]),
        ],
        "skip_if_none": True,
    },
    "sarima": {
        "kind": "forecast",
        "fn": train_and_forecast_sarima,
        "args": lambda X, y, out: (X, y, out),
        "save": [
            ("sarima", lambda res: res[1]),
        ],
        "skip_if_none": True,
    },
}

def run_model_spec(
        *,
        name: str,
        spec: ModelRegistryEntry,
        X: pd.DataFrame,
        y: pd.Series,
        out_dir: Path,
        device_type: Optional[str] = None,
    ) -> Optional[Dict[str, Union[str, float]]]:
    """
    Execute a single model specification from the model registry, 
    persist its trained artifacts, and return a flattened metrics record 
    suitable for aggregation.

    The function dispatches execution based on the model specification
    type (baseline or forecast), handles optional model skipping, saves
    any produced model artifacts, and extracts evaluation metrics into a
    single dictionary.

    Parameters
    ----------
    name : str
        Registry key identifying the model.
    spec : ModelRegistryEntry
        Model registry entry defining the training function, argument
        mapping, artifact extraction logic, and skip behavior.
    X : pd.DataFrame
        Feature matrix indexed by timestamp.
    y : pd.Series
        Target time series indexed by timestamp.
    out_dir : Path
        Base directory in which model outputs and artifacts are written.
    device_type : Optional[str], default=None
        Optional device or system identifier to include in the returned
        metrics record.

    Returns
    -------
    Optional[Dict[str, Union[str, float]]]
        Dictionary containing the model name, evaluation metrics, and
        optional device identifier. Returns ``None`` if the model is
        skipped according to its registry configuration.
    """
    if spec["kind"] == "baseline":
        # Baseline models always return metrics and multiple fitted 
        # models
        result = spec["fn"](*spec["args"](X, y, out_dir))
        metrics, _, _ = result
        for save_name, extractor in spec["save"]:
            save_model(extractor(result), save_name, out_dir)

    else:
        # Forecast models may optionally return no metrics (e.g. 
        # skipped)
        result = spec["fn"](*spec["args"](X, y, out_dir))
        metrics, _ = result
        if metrics is None:
            if spec.get("skip_if_none"):
                return None
            raise RuntimeError(f"Model '{name}' returned no metrics")
        for save_name, extractor in spec["save"]:
            save_model(extractor(result), save_name, out_dir)
    # Construct a flattened metrics row for downstream aggregation
    row: Dict[str, Union[str, float]] = {
        "model": name,
        **metrics,
    }
    if device_type is not None:
        row["device_type"] = device_type

    return row

def forecast_pipeline(
        data: Address,
        output_dir: Address,
        models: Literal["all", "baseline", "var", "lstm", "chronos", "sarima"],
        per_device_type: bool,
        *,
        verbosity: Verbosity = 1,
        write_log: bool = False,
    ) -> None:
    """
    Run an end-to-end forecasting pipeline for site-level and optionally
    per-device-type time series data.

    This pipeline reads aggregated telemetry, executes a selected set of
    forecasting models defined in ``MODEL_REGISTRY``, persists trained
    artifacts and evaluation metrics, and writes summary tables to disk.
    Models may be executed once at the site level or independently for
    each device type.

    Model execution, argument mapping, skip behavior, and artifact
    persistence are entirely driven by the model registry.

    Parameters
    ----------
    data : Address
        Input data source from which aggregated time series are read.
    output_dir : Address
        Base directory in which all model artifacts, plots, and metrics
        summaries are written.
    models : {"all", "baseline", "var", "lstm", "chronos", "sarima"}
        Set of models to execute. Use ``"all"`` to run all registered
        models.
    per_device_type : bool
        If ``True``, models are executed independently for each device
        type with sufficient data. If ``False``, models are executed 
        once at the site level.
    verbosity : Verbosity, default=1
        Logging verbosity level.
    write_log : bool, default=False
        If ``True``, write log output to disk in addition to stdout.

    Returns
    -------
    None
        All outputs are persisted to disk; no value is returned.
    """
    output_dir = validate_address(output_dir, mkdir=True)
    log = Logger(verbose=verbosity, log_dir=output_dir, write_log=write_log)
    # Normalize model selection into a set of registry keys
    if models == "all":
        selected = {"baseline", "var", "lstm", "chronos", "sarima"}
    else:
        selected = {models}
    # Per-device-type execution
    if per_device_type:
        log("Reading and aggregating time series per device_type...", 1)
        per_map = read_aggregated_timeseries_by_device_type(data)
        log(f"Found {len(per_map)} device types with sufficient data.", 1)
        overall_rows: List[Dict[str, Union[str, float]]] = []
        for dt, (x, y) in per_map.items():
            dt_slug = slugify(dt)
            out_dir = validate_address(
                output_dir / f"device_type_{dt_slug}",
                mkdir=True,
            )
            log(
                f"Running models for device_type='{dt}' "
                f"(rows={len(y)}).",
                1,
            )
            for name in selected:
                spec = MODEL_REGISTRY[name]
                row = run_model_spec(
                    name=name,
                    spec=spec,
                    X=x,
                    y=y,
                    out_dir=out_dir,
                    device_type=dt,
                )
                if row is not None:
                    overall_rows.append(row)
        if overall_rows:
            pd.DataFrame(overall_rows).to_csv(
                output_dir / "metrics_summary_per_device_type.csv",
                index=False,
            )
        log(
            f"Completed per-device-type forecasting. "
            f"Artifacts saved under: {output_dir}",
            1,
        )
    # Site-level execution
    log("Reading and aggregating time series (site-level)...", 1)
    X, y = read_aggregated_timeseries(data)
    log(
        f"Aggregated rows: {len(y)}; "
        f"features: {list(X.columns)}",
        1,
    )
    site_rows: List[Dict[str, Union[str, float]]] = []
    for name in selected:
        spec = MODEL_REGISTRY[name]
        log(f"Training and forecasting ({name})...", 1)
        out_dir = validate_address(
            output_dir / name,
            mkdir=True,
        )
        row = run_model_spec(
            name=name,
            spec=spec,
            X=X,
            y=y,
            out_dir=out_dir,
        )
        if row is not None:
            site_rows.append(row)
            log(f"{name} completed.", 1)
        else:
            log(f"{name} skipped.", 1)
    if site_rows:
        pd.DataFrame(site_rows).to_csv(
            output_dir / "metrics_summary.csv",
            index=False,
        )

    log("Completed Metrics Summary:", 1)
    for row in site_rows:
        model = row["model"]
        metrics_str = ", ".join(
            f"{k}: {v:.4f}"
            for k, v in row.items()
            if k != "model"
        )
        log(f"[{model}] {metrics_str}", 1)

    log(f"Artifacts saved in: {output_dir}", 1)

def parse_args() -> argparse.Namespace:
    """Parse input arguments for Forecasting."""
    parser = argparse.ArgumentParser(
        description="Run Forecasting pipeline",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help=(
            "Verbosity level: "
            "0 = silent, "
            "1 = info, "
            "2 = debug"
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Which model to train, acceptable inputs are "
            "all, baseline, var, lstm, chronos, sarima."
        ),
    )
    parser.add_argument(
        "--per_device_type",
        type=bool,
        default=False,
        help="Whether train and forecast stratified by device type.",
    )
    parser.add_argument(
        "--write_log",
        type=bool,
        default=False,
        help="Whether to store message/info outputs to a log file.",
    )

    return parser.parse_args()

def main() -> None:
    """
    Entry point for running the forecasting pipeline from the command 
    line.

    This function parses command-line arguments, initializes output
    directories and logging, and invokes the end-to-end forecasting
    pipeline to train, forecast, and evaluate the selected models on
    aggregated time-series data. All artifacts, plots, and evaluation
    metrics are written to disk under the configured output directory.
    """
    # Define directory for model, image and data ouputs
    output_dir = Path.cwd() / "outputs" / "forecasting"
    # Parse input arguments
    args = parse_args()
    # Train, predict, and evaluate models specified
    forecast_pipeline(
        data=DATA_ROOT / "forecast_data.parquet",
        output_dir=output_dir,
        models=args.models,
        per_device_type=args.per_device_type,
        verbosity=args.verbosity,
        write_log=args.write_log,
    )

if __name__ == "__main__":
    main()




