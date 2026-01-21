# stdlib
from typing import (
    Union, 
    Iterable, 
    Tuple, 
    List, 
    Sequence, 
    Optional, 
    cast
)
# thirdpartylib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split # pyright: ignore[reportUnknownVariableType] 
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    root_mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from statsmodels.tsa.seasonal import (# pyright: ignore[reportMissingTypeStubs]
    seasonal_decompose # pyright: ignore[reportUnknownVariableType]
)
from keras.models import ( # pyright: ignore[reportMissingTypeStubs]
    Sequential, 
    Model
)
from keras.layers import ( # pyright: ignore[reportMissingTypeStubs]
    LSTM, 
    Dense, 
    Reshape,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D
)
from keras.optimizers import RMSprop # pyright: ignore[reportMissingTypeStubs]
# projectlib
from pv_inverter_modeling.data.loaders import load_pandas
from pv_inverter_modeling.utils.typing import Address, PlotLineSpec
from pv_inverter_modeling.utils.paths import validate_address
from pv_inverter_modeling.data.schemas import Column, Metric
from pv_inverter_modeling.visualization.timeseries import use_dark_theme
from pv_inverter_modeling.evaluation.metrics import safe_mape
from pv_inverter_modeling.config.env import SAMPLE_DEVICE, DATA_ROOT

def fill_missing_times_to_padded_array(
        series: pd.Series, 
        freq: str ='5min'
    ) -> np.ndarray:
    """
    Convert an irregularly-sampled time series into a fixed-length, 
    day-aligned tensor.

    The input series is assumed to be indexed by timestamps. For each 
    calendar day, the function constructs a complete, evenly spaced time 
    grid (default: 5-minute intervals), inserts the observed values 
    where available, and fills any missing timestamps with 0.

    This is primarily useful for preparing time series data for sequence 
    models (e.g., RNNs, LSTMs, Transformers) that require uniform 
    temporal spacing and fixed-length inputs.

    Parameters
    ----------
    series : pd.Series
        A numeric pandas Series indexed by datetime-like values. The 
        index may be naive or timezone-aware; naive indices are assumed 
        to be UTC.

    freq : str, optional
        Target sampling frequency used to build the daily grid (default 
        is ``"5min"``). The function expects this frequency to yield 
        exactly 288 samples per day.

    Returns
    -------
    np.ndarray
        A 3-D NumPy array of shape ``(num_days, 288, 1)``, where each 
        slice along axis 0 corresponds to one day of data, padded and 
        aligned to the specified frequency.

    Notes
    -----
    - Days that do not produce exactly 288 time steps (e.g., due to DST 
      transitions or incomplete coverage) are skipped.
    - Missing observations within a day are filled with ``0.0``.
    - All output values are returned as ``float32``.
    """
    # Ensure datetime index and timezone-aware
    series = series.copy()
    series.index = ind = pd.to_datetime(series.index)
    if ind.tz is None:
        series.index = ind = ind.tz_localize('UTC')
    sequences: List[np.ndarray] = []
    groups = cast(
        Iterable[Tuple[pd.Timestamp, pd.Series]],
        series.groupby(ind.date) # pyright: ignore[reportUnknownMemberType]
    )
    for day, group in groups:
        start = pd.Timestamp(day).tz_localize(ind.tz)
        end = start + pd.Timedelta(days=1)
        # 288 5-minute intervals in a full day
        full_range = pd.date_range(
            start=start, 
            end=end - pd.Timedelta(minutes=5), 
            freq=freq, 
            tz=ind.tz
        )
        # Create empty series with full timestamps
        daily_series: pd.Series[float] = pd.Series(
            index=full_range, 
            dtype='float32'
        )
        # Fill known values from group
        daily_series.update(group)
        # Fill missing with 0
        daily_values = (
            daily_series
            .fillna(0.0) # pyright: ignore[reportUnknownMemberType]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )
        # Ensure it's exactly 288 steps
        if daily_values.shape[0] == 288:
            sequences.append(daily_values)
    padded_array = np.stack(sequences)  # (num_days, 288, 1)
    print("Padded shape:", padded_array.shape)
    
    return padded_array

def plot_days_line_with_time(padded_array: np.ndarray) -> None:
    """
    Plot multiple days of time-series data over a shared time-of-day 
    axis.

    This method visualizes a padded time-series tensor where each 
    slice along the first dimension represents one full day of data 
    sampled at 5-minute intervals. Each day is plotted as a separate 
    line over the same time-of-day axis, enabling visual comparison 
    of daily patterns, variability, and anomalies.

    The input array is expected to have been produced by a 
    preprocessing step that aligns each day to a fixed-length 
    sequence (e.g., ``(num_days, 288, 1)`` for 5-minute resolution).

    Parameters
    ----------
    padded_array : numpy.ndarray
        A 3D NumPy array of shape ``(num_days, 288, 1)``, where:
        - axis 0 corresponds to different days,
        - axis 1 corresponds to 5-minute time steps within a day,
        - axis 2 contains the value dimension.

    Returns
    -------
    None
        This method produces a plot for visualization and does not
        return a value.

    Notes
    -----
    - All days share the same time-of-day x-axis (00:00-23:55).
    - Individual lines are plotted with partial transparency to
    highlight overlapping patterns.
    - Only the first few days are labeled in the legend to reduce
    visual clutter.
    """
    # Number of full days represented in the padded array
    num_days = padded_array.shape[0]
    # Create time-of-day labels for one full day at 5-minute 
    # resolution
    time_of_day = pd.date_range(
        "00:00",
        "23:55",
        freq="5min"
    ).strftime("%H:%M")
    # Initialize the figure for plotting
    plt.figure(figsize=(15, 10)) # pyright: ignore[reportUnknownMemberType]
    # Plot each day as a separate line on the same time-of-day axis
    for i in range(num_days):
        plt.plot(  # pyright: ignore[reportUnknownMemberType]
            time_of_day,
            padded_array[i, :, 0],
            alpha=0.3,
            label=f"Day {i + 1}" if i < 5 else ""
        )
    # Add plot title and axis labels
    plt.title(  # pyright: ignore[reportUnknownMemberType]
        f"{num_days} Days of Data (Each Line Shows "
        "5-Minute Intervals Over a Day)"
    )
    plt.xlabel("Time of Day")  # pyright: ignore[reportUnknownMemberType]
    plt.ylabel("Value")        # pyright: ignore[reportUnknownMemberType]
    # Reduce x-axis tick density for readability (every 2 hours)
    step = 24  # 24 × 5-minute intervals = 2 hours
    plt.xticks(  # pyright: ignore[reportUnknownMemberType]
        time_of_day[::step],
        rotation=45
    )
    # Add grid and adjust layout
    plt.grid(True)  # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()
    # Render the plot
    plt.show()  # pyright: ignore[reportUnknownMemberType]

def _plot_lines(line: PlotLineSpec) -> None:
    """
    Render a single line on the current Matplotlib axes.

    This helper draws a line plot based on a ``PlotLineSpec``. If 
    explicit x-values are provided, the line is plotted as 
    ``(x, y)``; otherwise, the y-values are plotted against their 
    implicit index. Optional styling attributes such as label, 
    linestyle, alpha, and color are applied when present.

    Parameters
    ----------
    line : PlotLineSpec
        Plot specification defining the data to plot and optional 
        visual styling attributes.

    Returns
    -------
    None
        This method renders the line on the active Matplotlib axes 
        and does not return a value.
    """
    # Plot using explicit x-values if provided
    if "x" in line:
        plt.plot(  # pyright: ignore[reportUnknownMemberType]
            line["x"],
            line["y"],
            label=line.get("label"),
            linestyle=line.get("linestyle"),
            alpha=line.get("alpha"),
            color=line.get("color"),
            marker=line.get("marker"),
            linewidth=line.get("linewidth")
        )
    # Otherwise, plot y-values against their implicit index
    else:
        plt.plot(  # pyright: ignore[reportUnknownMemberType]
            line["y"],
            label=line.get("label"),
            linestyle=line.get("linestyle"),
            alpha=line.get("alpha"),
            color=line.get("color"),
            marker=line.get("marker"),
            linewidth=line.get("linewidth")
        )

def _scatter_lines(line: PlotLineSpec) -> None:
    """
    Render a single scatter plot on the current Matplotlib axes.

    This helper draws a scatter plot based on a ``PlotLineSpec``. 
    Unlike line plots, scatter plots require both explicit x- and 
    y-values. An error is raised if the x-values are missing. 
    Optional styling attributes such as label, alpha, and color 
    are applied when present.

    Parameters
    ----------
    line : PlotLineSpec
        Plot specification defining the x- and y-values to plot and
        optional visual styling attributes.

    Returns
    -------
    None
        This method renders the scatter plot on the active 
        Matplotlib axes and does not return a value.

    Raises
    ------
    ValueError
        If the plot specification does not include explicit 
        x-values.
    """
    # Scatter plots require explicit x- and y-values
    if "x" not in line:
        raise ValueError("scatter plots require both x and y")
    plt.scatter(  # pyright: ignore[reportUnknownMemberType]
        line["x"],
        line["y"],
        label=line.get("label"),
        alpha=line.get("alpha"),
        color=line.get("color"),
        marker=line.get("marker")
    )

def plot_helper(
        *,
        figsize: Tuple[int, int],
        lines: Sequence[PlotLineSpec],
        title: str,
        grid: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        tightlayout: bool = False,
    ) -> None:
    """
    Render a matplotlib figure composed of multiple plot 
    specifications.

    This internal helper centralizes common plotting boilerplate 
    such as figure creation, axis labeling, legend handling, and 
    layout control. Each entry in ``lines`` defines a single visual 
    element (line or scatter) and is dispatched to the appropriate 
    plotting backend.

    Parameters
    ----------
    figsize : tuple[int, int]
        Size of the matplotlib figure in inches.
    lines : Sequence[PlotLineSpec]
        Collection of plot specifications defining what to render.
        Each specification must include a ``plot_type`` field
        indicating whether the element should be rendered as a line
        or a scatter plot.
    title : str
        Title applied to the figure.
    grid : bool, default True
        Whether to display a background grid.
    xlabel : str, optional
        Label for the x-axis. If None, no label is applied.
    ylabel : str, optional
        Label for the y-axis. If None, no label is applied.
    tightlayout : bool, default False
        If True, apply ``plt.tight_layout`` to reduce label and
        element overlap.

    Returns
    -------
    None
        This method renders a plot and does not return a value.
    """
    # Create a new figure with the requested dimensions
    plt.figure(figsize=figsize) # pyright: ignore[reportUnknownMemberType]
    # Render each plot element according to its declared type
    for line in lines:
        if line["plot_type"] == "line":
            _plot_lines(line)
        elif line["plot_type"] == "scatter":
            _scatter_lines(line)
    # Apply figure-level annotations
    plt.title(title) # pyright: ignore[reportUnknownMemberType]
    if xlabel is not None:
        plt.xlabel(xlabel) # pyright: ignore[reportUnknownMemberType]
    if ylabel is not None:
        plt.ylabel(ylabel) # pyright: ignore[reportUnknownMemberType]
    # Display legend and grid if requested
    plt.legend() # pyright: ignore[reportUnknownMemberType]
    plt.grid(grid) # pyright: ignore[reportUnknownMemberType]
    # Optionally compact layout to avoid overlaps
    if tightlayout:
        plt.tight_layout()
    # Render the figure
    plt.show() # pyright: ignore[reportUnknownMemberType]

class timeseries_data(object):
    """
    Utility class for loading, cleaning, filtering, and reshaping
    inverter time-series data.

    This class provides a structured interface for working with
    time-series telemetry data stored in Parquet format. It encapsulates
    common preprocessing steps required for analysis and modeling,
    including column selection, missing-value handling, timestamp
    normalization, device- and metric-level filtering, and 
    transformation from long-format to wide, time-indexed 
    representations.

    The class is designed to support downstream workflows such as:
    - exploratory data analysis,
    - correlation and statistical analysis,
    - feature engineering,
    - machine learning and time-series modeling.

    Data is loaded eagerly into a pandas ``DataFrame`` using the
    project's standardized I/O layer, with column projection applied at
    load time to reduce memory usage.

    Responsibilities
    ----------------
    - Load a subset of required columns from Parquet datasets
    - Sanitize missing values
    - Normalize timestamps to timezone-aware datetime objects
    - Filter data by device type and device name
    - Select and align metric time series by component
    - Produce wide, time-indexed DataFrames suitable for modeling

    Attributes
    ----------
    fname : Address
        Name of the Parquet dataset file.
    data : pandas.DataFrame
        Raw dataset loaded from disk with missing values replaced by
        zeros.
    data_change_time : pandas.DataFrame
        Timestamp-normalized copy of the dataset created by
        ``makecopy_changetimetype``.

    Notes
    -----
    - All filtering operations return new DataFrames; the original
      loaded dataset remains unchanged.
    - Timestamp alignment is performed using outer joins to preserve
      uneven time coverage across metrics.
    - When duplicate timestamps exist for a metric, the most recent
      value is retained.
    - This class assumes a schema containing at least the following
      columns:
      ``Column.TIMESTAMP``, ``Column.DEVICE``, ``Column.TYPE``,
      ``Column.METRIC``, and ``Column.VALUE``.
    """
    
    def __init__(self, filename: Address, path: Address) -> None:
        """
        Initialize the dataset handler and load the required data 
        columns.

        This constructor validates the provided dataset path, loads the
        specified Parquet file into a pandas ``DataFrame`` using the
        project's standardized I/O layer, and performs initial data
        sanitation steps.

        Only a predefined subset of columns required for downstream
        processing is loaded, reducing memory usage and I/O overhead. 
        Any missing values in the loaded dataset are replaced with 
        zeros.

        Parameters
        ----------
        filename : Address
            Name of the Parquet file to load.
        path : Address
            Base directory containing the dataset file.

        Returns
        -------
        None
            This method initializes internal state and does not return a
            value.

        Attributes
        ----------
        fname : Address
            Name of the dataset file.
        data : pandas.DataFrame
            Loaded dataset containing only the selected columns:
            ``Column.TIMESTAMP``, ``Column.DEVICE``, ``Column.TYPE``,
            ``Column.METRIC``, and ``Column.VALUE``.
        data_change_time : pandas.DataFrame
            Placeholder DataFrame to store a timestamp-converted copy of
            the dataset created by downstream processing methods.

        Notes
        -----
        - Column selection is applied at load time to minimize memory 
          usage.
        - Missing values are replaced with ``0`` via
        ``replace_nan_with_zeros``.
        - Path validation is performed using ``validate_address``.
        """
        self.fname = filename
        self.data = load_pandas(
            validate_address(path) / self.fname,
            columns=[
                Column.TIMESTAMP,
                Column.DEVICE, 
                Column.TYPE,
                Column.METRIC,
                Column.VALUE
            ]
        )
        self.data_change_time = pd.DataFrame()
        self.replace_nan_with_zeros()
                
    def replace_nan_with_zeros(self) -> None:
        """
        Replace all missing (NaN) values in the underlying DataFrame 
        with 0.

        This method performs an in-place fill operation on 
        ``self.data``, replacing all ``NaN`` values with ``0``. A 
        diagnostic message is printed indicating whether any missing 
        values were present prior to replacement.

        Returns
        -------
        None
            This method mutates the underlying DataFrame in place and 
            does not return a value.

        Notes
        -----
        - The operation is performed in place using 
          ``fillna(..., inplace=True)``.
        - This method assumes ``self.data`` is a pandas ``DataFrame``.
        """
        has_nan = self.data.isna().any().any()
        print(f"Does the DataFrame contain any NaN values? {has_nan}")
        self.data.fillna( # pyright: ignore[reportUnknownMemberType]
            0, 
            inplace=True
        )     

    def sort_on_column(
            self,
            dataframein: pd.DataFrame,
            column_name: str,
        ) -> pd.DataFrame:
        """
        Return a DataFrame sorted by the specified column.

        This method returns a new DataFrame sorted in ascending order by
        the given column name. The input DataFrame is not modified.

        Parameters
        ----------
        dataframein : pandas.DataFrame
            Input DataFrame to be sorted.
        column_name : str
            Name of the column to sort by.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame sorted by ``column_name`` in ascending 
            order.

        Notes
        -----
        - Sorting is performed using ``DataFrame.sort_values``.
        - The original DataFrame remains unchanged.
        """
        sorted_df = dataframein.sort_values(by=column_name)

        return sorted_df

    def filter_devicetype(
            self,
            datain: pd.DataFrame,
            device_type: str,
        ) -> pd.DataFrame:
        """
        Filter a DataFrame by device type.

        This method returns a new DataFrame containing only the rows 
        where the ``"device_type"`` column matches the specified device 
        type. The input DataFrame is not modified.

        Parameters
        ----------
        datain : pandas.DataFrame
            Input DataFrame to be filtered. Must contain a 
            ``"device_type"`` column.
        device_type : str
            Device type value to filter on.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing only rows with the specified 
            device type.

        Notes
        -----
        - Filtering is performed using boolean indexing.
        - The returned DataFrame is a copy of the filtered result.
        """
        dataout = datain[datain["device_type"] == device_type].copy()

        return dataout

    def filter_devicename(
            self,
            datain: pd.DataFrame,
            device_name: str,
        ) -> pd.DataFrame:
        """
        Filter a DataFrame by device name.

        This method returns a new DataFrame containing only the rows 
        where the ``"device_name"`` column matches the specified device 
        name. The input DataFrame is not modified.

        Parameters
        ----------
        datain : pandas.DataFrame
            Input DataFrame to be filtered. Must contain a 
            ``"device_name"`` column.
        device_name : str
            Device name value to filter on.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing only rows with the specified 
            device name.

        Notes
        -----
        - Filtering is performed using boolean indexing.
        - Consider calling ``.copy()`` on the result if the returned
        DataFrame will be modified downstream.
        """  
        dataout = datain[datain['device_name'] == device_name]

        return dataout

    def makecopy_changetimetype(self) -> pd.DataFrame:
        """
        Return a copy of the dataset with the timestamp column converted 
        to timezone-aware ``datetime`` objects.

        This method creates a shallow copy of ``self.data`` and converts 
        the timestamp column (``Column.TIMESTAMP``) from its string
        representation into a pandas ``datetime64[ns, UTC]`` dtype using
        the expected ISO 8601 format.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame
            A copy of the original dataset with the timestamp column 
            parsed as timezone-aware datetimes in UTC.

        Notes
        -----
        - The expected timestamp format is ``"%Y-%m-%dT%H:%M:%S.%fZ"``.
        - The original ``self.data`` is not modified.
        - The converted DataFrame is also stored on the instance as
          ``self.data_change_time``.
        """
        date_time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        self.data_change_time = self.data.copy()
        self.data_change_time[Column.TIMESTAMP] = pd.to_datetime(
            self.data[Column.TIMESTAMP], 
            format=date_time_format, 
            utc=True
        )

        return self.data_change_time

    def filter_data_from_device_name(self, device_name: str) -> pd.DataFrame:
        """
        Filter the dataset by device name.

        This method returns a subset of ``self.data_change_time`` 
        containing only the rows where the device identifier column
        (``Column.DEVICE``) matches the specified device name.

        Parameters
        ----------
        device_name : str
            Device name or identifier to filter on.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing only rows associated with the 
            specified device.

        Notes
        -----
        - This method assumes ``self.data_change_time`` has already been
        created (e.g., via ``makecopy_changetimetype``).
        - The returned DataFrame is a view of the filtered result; call
          ``.copy()`` if the result will be modified downstream.
        """
        x = self.data_change_time[
            self.data_change_time[Column.DEVICE] == device_name
        ]

        return x

    def filter_data_with_column(
            self,
            dfin: pd.DataFrame,
            column_name: str,
        ) -> pd.DataFrame:
        """
        Filter a DataFrame by metric name.

        This method returns a new DataFrame containing only the rows 
        where the metric column (``Column.METRIC``) matches the 
        specified column name. The input DataFrame is not modified.

        Parameters
        ----------
        dfin : pandas.DataFrame
            Input DataFrame to be filtered. Must contain a metric column
            identified by ``Column.METRIC``.
        column_name : str
            Metric (or column) name value to filter on.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing only rows matching the specified
            metric name.

        Notes
        -----
        - Filtering is performed using boolean indexing.
        - The returned DataFrame is a copy of the filtered result.
        """
        dataout = dfin[dfin[Column.METRIC] == column_name].copy()
        
        return dataout
      
    def filter_column_from_dataset(
            self,
            datain: pd.DataFrame,
            componentin: str,
        ) -> pd.DataFrame:
        """
        Extract and align metric values for a given component into a
        time-indexed DataFrame.

        This method filters the input dataset to all metrics whose names
        contain the specified component identifier, pivots the data into
        a wide format with one column per metric, and aligns all values 
        on a shared timestamp index.

        Each metric is outer-joined on the timestamp axis, ensuring that
        uneven or partially missing time coverage across metrics is
        preserved. Missing values introduced by alignment are filled 
        with ``0`` in the returned DataFrame.

        Parameters
        ----------
        datain : pandas.DataFrame
            Input dataset containing time-series observations. Must 
            include at least the following columns:
            - ``Column.METRIC``: metric or signal name
            - ``Column.TIMESTAMP``: timestamp of the observation
            - ``Column.VALUE``: numeric value of the observation

        componentin : str
            Substring used to identify and select relevant metrics. Any
            metric name containing this string will be included in the
            output.

        Returns
        -------
        pandas.DataFrame
            A time-indexed DataFrame where each column corresponds to a
            metric matching ``componentin`` and each row corresponds to 
            a timestamp. Missing values are filled with ``0``.

        Notes
        -----
        - If multiple values exist for the same metric and timestamp, 
          the last observed value is retained.
        - The output DataFrame is indexed by ``Column.TIMESTAMP`` and 
          sorted in ascending time order.
        - This structure is suitable for correlation analysis, feature
          engineering, and time-series modeling.
        """
        # Identify all metric names that contain the requested component
        metrics = [
            m for m in pd.unique(datain[Column.METRIC])
            if componentin in str(m)
        ]
        # Initialize an empty DataFrame that will be progressively 
        # joined with each metric-specific time series
        out = pd.DataFrame()
        # Process each matching metric independently
        for m in metrics:
            # Select timestamp-value pairs for the current metric
            df_m = (
                datain.loc[
                    datain[Column.METRIC] == m,
                    [Column.TIMESTAMP, Column.VALUE]
                ]
                # Rename the value column to the metric name
                .rename(columns={Column.VALUE: m})
                # Ensure temporal ordering
                .sort_values(Column.TIMESTAMP)
                # Remove duplicate timestamps, keeping the most recent 
                # value
                .drop_duplicates(subset=[Column.TIMESTAMP], keep="last")
                # Use timestamp as the index for alignment
                .set_index(Column.TIMESTAMP)
            )
            # Outer-join with the accumulated DataFrame to preserve all
            # timestamps across metrics
            out = df_m if out.empty else out.join(df_m, how="outer")

        # Replace missing values introduced by alignment with zeros
        return out.fillna(0)  # pyright: ignore[reportUnknownMemberType]

    def run_on_selected_device(self, devicein: str) -> pd.DataFrame:
        """
        Extract and prepare AC-related time-series data for a specific 
        inverter device.

        This method performs a fixed preprocessing pipeline on the 
        internal dataset to isolate data for a single inverter device 
        and extract all AC-related metrics. The resulting DataFrame is 
        time-indexed, wide-formatted (one column per metric), and 
        suitable for downstream analysis such as correlation, feature 
        engineering, or modeling.

        The processing steps are:
        1. Create a copy of the dataset with the timestamp column 
           converted
        to timezone-aware datetime objects.
        2. Filter the dataset to inverter-type devices only.
        3. Select data for the specified inverter device.
        4. Extract and align all metrics whose names contain ``"AC_"``.

        Parameters
        ----------
        devicein : str
            Device name or identifier corresponding to the inverter to 
            be selected.

        Returns
        -------
        pandas.DataFrame
            A time-indexed DataFrame containing AC-related metrics for 
            the specified inverter device, with missing values filled 
            with ``0``.
        """
        # Create a copy of the data with the timestamp column converted 
        # to timezone-aware datetime objects (stored internally on the 
        # instance)
        data_t = self.makecopy_changetimetype()
        # Restrict the dataset to inverter-type devices only
        data_inverter = self.filter_devicetype(data_t, "Inverter")
        # Select data corresponding to the specified inverter device
        device_inverter = self.filter_devicename(data_inverter, devicein)
        # Extract and align all AC-related metrics into a wide, 
        # time-indexed DataFrame
        device_AC = self.filter_column_from_dataset(device_inverter, "AC_")

        return device_AC

class stable_model(object):
    """
    End-to-end LSTM-based modeling and visualization utility for
    inverter time-series prediction.

    This class encapsulates the full workflow required to train,
    evaluate, and visualize a sequence-to-sequence LSTM model for
    next-day forecasting using historical inverter data sampled at
    fixed time intervals.

    The workflow includes:
    - preprocessing and detrending of time-series data,
    - padding and alignment into fixed-length daily sequences,
    - construction of supervised learning datasets using a rolling
      multi-day window,
    - model definition, training, and evaluation,
    - computation of regression accuracy metrics,
    - and generation of diagnostic visualizations.

    The class is designed to operate on pandas ``DataFrame`` or
    ``Series`` inputs and stores intermediate artifacts (scalers,
    train/test splits, trained models) as instance attributes to
    support iterative experimentation and analysis.

    Attributes
    ----------
    data : pandas.DataFrame or pandas.Series
        Raw inverter time-series data provided at initialization.
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler used for normalization and inverse transforms.
    X_train, X_test : numpy.ndarray
        Training and testing input sequences.
    y_train, y_test : numpy.ndarray
        Training and testing target sequences.
    lstm_stable_model : tensorflow.keras.Model
        Trained LSTM model instance produced by the modeling pipeline.

    Notes
    -----
    - The modeling approach assumes fixed-resolution time steps
      (e.g., 5-minute intervals) and full-day alignment.
    - Sequence construction follows a sliding-window strategy where
      multiple past days are used to predict the next full day.
    - Plotting utilities are integrated to support qualitative model
      evaluation alongside numeric metrics.
    """

    def __init__(self, inverter_data: Union[pd.DataFrame, pd.Series]):
        """
        Initialize the instance with inverter time-series data.

        This constructor stores the provided pandas object for
        downstream analysis and modeling. The input may be either a
        ``DataFrame`` or ``Series`` depending on the stage of the
        processing pipeline.

        Parameters
        ----------
        inverter_data : pandas.DataFrame or pandas.Series
            Inverter time-series data to be used by this instance.

        Returns
        -------
        None
            This method initializes internal state and does not return a
            value.
        """
        self.data = inverter_data

    def plot_result(
            self,
            y_pred_flat: np.ndarray,
            y_test_flat: np.ndarray,
            y_pred_scaled: np.ndarray,
            y_test: np.ndarray,
        ) -> None:
        """
        Visualize model predictions against ground-truth values.

        This method inverse-transforms predicted and actual values back
        to their original scale and produces a set of diagnostic plots
        to assess model performance. The visualizations include:
        - a single-day forecast comparison,
        - full-series line comparisons in scaled space,
        - full-series scatter comparisons,
        - and an actual-versus-predicted scatter plot.

        Parameters
        ----------
        y_pred_flat : numpy.ndarray
            Flattened model predictions in scaled space.
        y_test_flat : numpy.ndarray
            Flattened ground-truth values in scaled space.
        y_pred_scaled : numpy.ndarray
            Scaled model predictions with daily structure preserved.
        y_test : numpy.ndarray
            Scaled ground-truth values with daily structure preserved.

        Returns
        -------
        None
            This method produces plots for visual inspection and does
            not return a value.
        """

        # Inverse-transform predictions and targets back to original 
        # scale
        y_pred_orig = self.scaler.inverse_transform(
            y_pred_flat.reshape(-1, 1)
        ).reshape(y_pred_scaled.shape)
        y_test_orig = self.scaler.inverse_transform(
            y_test_flat.reshape(-1, 1)
        ).reshape(y_test.shape)
        # Select a single test day for detailed inspection
        day_index = 0
        # Plot 1: Full-day forecast vs actual (original scale)
        plot_1 = [
            PlotLineSpec(
                y=y_test_orig[day_index].squeeze(),
                plot_type="line",
                label="Actual",
            ),
            PlotLineSpec(
                y=y_pred_orig[day_index].squeeze(),
                plot_type="line",
                label="Predicted",
            ),
        ]
        plot_helper(
            figsize=(12, 5),
            lines=plot_1,
            title="Full Day Forecast",
            grid=True,
        )
        # Plot 2: Full flattened time-series comparison (line)
        plot_2 = [
            PlotLineSpec(
                x=np.arange(len(y_test_flat)),
                y=y_test_flat,
                plot_type="line",
                label="Actual stableData",
            ),
            PlotLineSpec(
                x=np.arange(len(y_pred_flat)),
                y=y_pred_flat,
                plot_type="line",
                label="Predicted stableData",
                linestyle="--",
            ),
        ]
        plot_helper(
            figsize=(12, 5),
            lines=plot_2,
            title="LSTM Forecasting of stableData from Past stableData",
            xlabel="Time",
            ylabel="stableData",
            tightlayout=True,
        )
        # Plot 3: Full flattened time-series comparison (scatter)
        plot_3 = [
            PlotLineSpec(
                x=np.arange(len(y_test_flat)),
                y=y_test_flat,
                plot_type="scatter",
                label="Actual stableData",
            ),
            PlotLineSpec(
                x=np.arange(len(y_pred_flat)),
                y=y_pred_flat,
                plot_type="scatter",
                label="Predicted stableData",
            ),
        ]
        plot_helper(
            figsize=(12, 5),
            lines=plot_3,
            title="LSTM Forecasting of stableData from Past stableData",
            xlabel="Time",
            ylabel="stableData",
            tightlayout=True,
        )
        # Plot 4: Actual vs predicted scatter with identity line
        plot_4 = [
            PlotLineSpec(
                x=y_test_flat,
                y=y_pred_flat,
                plot_type="scatter",
                label="Actual stableData",
                alpha=0.5,
            ),
            PlotLineSpec(
                x=[min(y_test_flat), max(y_test_flat)],
                y=[min(y_test_flat), max(y_test_flat)],
                plot_type="line",
                label="Identity",
                linestyle="--",
                color="red",
            ),
        ]
        plot_helper(
            figsize=(12, 5),
            lines=plot_4,
            title="Actual vs Predicted",
            grid=False,
            xlabel="Actual",
            ylabel="Predicted",
        )
 
    def accuracy_matrix(
            self,
            y_actual_rescaled: np.ndarray,
            y_pred_rescaled: np.ndarray,
        ) -> None:
        """
        Compute and display accuracy and error metrics for model 
        predictions.

        This method compares rescaled ground-truth values against 
        rescaled model predictions and prints a collection of 
        descriptive statistics and regression error metrics. It is 
        intended for diagnostic and evaluation purposes rather than 
        returning structured results.

        Parameters
        ----------
        y_actual_rescaled : numpy.ndarray
            Array of ground-truth target values after inverse scaling.
        y_pred_rescaled : numpy.ndarray
            Array of model-predicted values after inverse scaling.

        Returns
        -------
        None
            This method prints evaluation metrics and does not return a
            value.
        """
        # Descriptive statistics for actual values
        print("Actual min:", np.min(y_actual_rescaled))
        print("Actual max:", np.max(y_actual_rescaled))
        print("Actual std:", np.std(y_actual_rescaled))
        # Descriptive statistics for predicted values
        print("Predicted min:", np.min(y_pred_rescaled))
        print("Predicted max:", np.max(y_pred_rescaled))
        print("Predicted std:", np.std(y_pred_rescaled))
        # Redundant but explicit combined summary 
        # (kept for clarity/debugging)
        print(
            "Actual - min:",
            np.min(y_actual_rescaled),
            "max:",
            np.max(y_actual_rescaled),
            "std:",
            np.std(y_actual_rescaled),
        )
        print(
            "Pred - min:",
            np.min(y_pred_rescaled),
            "max:",
            np.max(y_pred_rescaled),
            "std:",
            np.std(y_pred_rescaled),
        )
        # Root Mean Squared Error (RMSE)
        rmse = root_mean_squared_error(y_actual_rescaled, y_pred_rescaled)
        print(f"RMSE: {rmse}")
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_actual_rescaled, y_pred_rescaled)
        print(f"MAE: {mae}")
        # Mean Absolute Percentage Error (MAPE), excluding zero targets
        mask = y_actual_rescaled != 0
        mape = safe_mape(y_actual_rescaled[mask], y_pred_rescaled[mask])
        print(f"MAPE: {mape}%")
        # R² score computed on flattened arrays
        r2 = r2_score(
            y_actual_rescaled.reshape(-1),
            y_pred_rescaled.reshape(-1),
        )
        print(f"R² Score (flattened): {r2}")
        # R² score computed on original array shapes
        r2 = r2_score(y_actual_rescaled, y_pred_rescaled)
        print(f"R² Score: {r2}")

    def form_7days_XY_data(self) -> None:
        """
        Construct supervised learning datasets using a 7-day sliding 
        window over detrended time-series data.

        This method prepares input-target pairs suitable for sequence 
        models (e.g., LSTMs) by:
        1. Detrending a target metric using seasonal decomposition.
        2. Padding the time series into fixed-length daily sequences
        (5-minute resolution, 288 steps per day).
        3. Scaling the data using Min-Max normalization.
        4. Creating supervised samples where the previous 7 days are 
           used to predict the next day.
        5. Splitting the resulting dataset into training and testing 
           sets.
        
        The resulting tensors are stored on the instance for downstream
        model training and evaluation.

        Returns
        -------
        None
            This method populates the following instance attributes:
            - ``self.X_train`` : training input sequences
            - ``self.X_test``  : testing input sequences
            - ``self.y_train`` : training targets
            - ``self.y_test``  : testing targets
            - ``self.scaler``  : fitted ``MinMaxScaler`` for inverse 
              transforms

        Notes
        -----
        - Each day consists of 288 time steps (5-minute intervals).
        - Inputs use a flattened 7-day window (2016 steps) with a single
        feature channel.
        - Targets correspond to the full next-day sequence (288 steps).
        """
        # Copy data to avoid mutating the original dataset
        stable_pred_data = self.data.copy()
        # Target metric and seasonal period
        target_col = Metric.AC_POWER
        p = 365
        # Decompose the time series and remove the trend component
        decomposition = seasonal_decompose(
            stable_pred_data[target_col],
            model="additive",
            period=p,
            extrapolate_trend="freq",  # pyright: ignore[reportArgumentType]
        )
        detrended_series = stable_pred_data[target_col] - decomposition.trend
        # Pad the series into fixed daily sequences (288 steps per day)
        X_padded = fill_missing_times_to_padded_array(detrended_series)
        # Optional visualization of daily patterns
        plot_days_line_with_time(X_padded)
        # Scale values using Min–Max normalization
        X_flat = X_padded.reshape(-1, 1)
        scaler = MinMaxScaler()
        X_scaled_flat = (
            scaler
            .fit_transform(X_flat)  # pyright: ignore[reportUnknownMemberType]
        )
        self.scaler = scaler
        # Restore daily structure and add channel dimension
        X_scaled = X_scaled_flat.reshape(X_padded.shape)
        X_scaled = X_scaled[..., np.newaxis]
        # Build supervised samples: 7 previous days -> next day
        window_size = 7
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for i in range(window_size, len(X_scaled)):
            past_7_days = X_scaled[i - window_size : i].reshape(-1)
            next_day = X_scaled[i]
            X_list.append(past_7_days)
            y_list.append(next_day)

        x = np.array(X_list)
        y = np.array(y_list)
        # Reshape inputs and targets for sequence models
        x = x.reshape((x.shape[0], 7 * 288, 1))
        y = y.reshape((y.shape[0], 288, 1))
        # Split data while preserving temporal order
        X_train, X_test, y_train, y_test = cast(
            Tuple[
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
            ],
            train_test_split(x, y, test_size=0.2, shuffle=False),
        )
        # Store datasets on the instance
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # Basic shape verification
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

    def form_7day_model(self) -> Model:
        """
        Construct an LSTM-based model for next-day time-series 
        prediction using a 7-day input window.

        The model consumes a flattened sequence of 7 days of data
        (2016 time steps at 5-minute resolution) and predicts the full
        next-day sequence (288 time steps). The architecture uses 
        stacked LSTM layers to learn temporal dependencies, followed by 
        a dense projection and reshaping step to produce the daily 
        forecast.

        Returns
        -------
        tensorflow.keras.Model
            A compiled Keras model mapping inputs of shape ``(2016, 1)``
            to outputs of shape ``(288, 1)``.
        """
        # Define a sequential LSTM model for sequence-to-sequence prediction
        model = Sequential([
            # Encode the 7-day input sequence
            LSTM(128, return_sequences=True, input_shape=(2016, 1)),
            # Reduce the encoded sequence to a fixed-length representation
            LSTM(64),
            # Project to the full next-day horizon
            Dense(288),
            # Reshape output to (time_steps, channels)
            Reshape((288, 1)),
        ])

        return model

    def LSTM_stable_model(
            self,
            display_flag: bool = True,
        ) -> Model:
        """
        Train and evaluate an LSTM model for next-day stable time-series
        prediction.

        This method executes the full modeling pipeline:
        - prepares supervised training and test data using a rolling
          7-day window,
        - constructs the LSTM architecture,
        - compiles and trains the model,
        - evaluates predictions using error metrics,
        - and optionally generates diagnostic plots.

        The trained model is stored on the instance for later reuse.

        Parameters
        ----------
        display_flag : bool, default True
            If True, generate diagnostic plots comparing predicted and
            actual values after model evaluation.

        Returns
        -------
        tensorflow.keras.Model
            The trained LSTM model.
        """
        # Prepare supervised (X, y) datasets using a 7-day rolling 
        # window
        self.form_7days_XY_data()
        # Build the LSTM architecture
        model = self.form_7day_model()
        # Configure optimizer and compile the model
        optimizer = RMSprop(learning_rate=0.01)
        model.compile(  # pyright: ignore[reportUnknownMemberType]
            optimizer=optimizer,  # pyright: ignore[reportArgumentType]
            loss="mse",
            metrics=["mae"],
        )
        # Display model architecture summary
        print(model.summary())  # pyright: ignore[reportUnknownMemberType]
        # Train the model
        num_epoch = 150
        model.fit(  # pyright: ignore[reportUnknownMemberType]
            self.X_train,
            self.y_train,
            epochs=num_epoch,
            batch_size=16,
            validation_split=0.1,
            verbose=1,  # pyright: ignore[reportArgumentType]
        )
        # Generate predictions on the test set
        y_pred_scaled = cast(
            np.ndarray,
            model.predict(  # pyright: ignore[reportUnknownMemberType]
                self.X_test
            ),
        )
        # Flatten predictions and targets for metric computation
        y_pred_flat = y_pred_scaled.reshape(-1, 1)
        y_test_flat = self.y_test.reshape(-1, 1)
        # Compute and display evaluation metrics
        self.accuracy_matrix(y_test_flat, y_pred_flat)
        # Optionally generate diagnostic plots
        if display_flag:
            self.plot_result(
                y_pred_flat,
                y_test_flat,
                y_pred_scaled,
                self.y_test,
            )
        # Store trained model on the instance
        self.lstm_stable_model = model

        return model

class trend_model(object):
    """
    Trend-focused regression model for next-day time-series forecasting.

    This class implements a lightweight modeling pipeline designed to
    learn and forecast the *long-term trend component* of a high-
    frequency time series. Unlike full-signal models that attempt to
    capture both short-term variability and seasonal structure, this
    class isolates the trend using seasonal decomposition and trains a
    convolutional neural network to predict the next day of trend values
    from recent historical context.

    The workflow consists of:
    - extracting the trend component from a datetime-indexed time series
      using seasonal decomposition,
    - padding and aligning the trend into fixed-length daily sequences,
    - constructing supervised learning samples using a 7-day rolling
      window,
    - training a 1D convolutional regression model to predict the full
      next-day trend profile,
    - and storing trained artifacts for evaluation and reuse.

    This class is intended for scenarios where:
    - interpretability of long-term structure is preferred over exact
      short-term accuracy,
    - the trend component is modeled independently from residual or
      seasonal components,
    - or a simpler, more stable forecasting baseline is required.

    Attributes
    ----------
    X_train, X_test : numpy.ndarray
        Training and testing input sequences constructed from rolling
        7-day trend windows.
    y_train, y_test : numpy.ndarray
        Corresponding next-day trend targets.
    trend_data : numpy.ndarray
        Daily-aligned trend data after padding and preprocessing.
    tmodel : tensorflow.keras.Model
        Trained convolutional regression model for trend forecasting.

    Notes
    -----
    - The model assumes a fixed intra-day resolution (e.g., 288 time
      steps per day at 5-minute intervals).
    - Seasonal decomposition uses an additive model with a yearly period.
    - Predictions are deterministic and regression-based; no probabilistic
      uncertainty estimates are produced.
    - This class is complementary to full-signal sequence models (e.g.,
      LSTM-based approaches) rather than a replacement.
    """
    
    def setup_7days_train_data(
            self, 
            newdata: Union[pd.Series, np.ndarray]
        ) -> None:
        """
        Construct training and testing datasets using a 7-day sliding
        window over daily time-series data.

        This method converts a padded daily time-series array into
        supervised learning samples suitable for sequence models. Each
        training example consists of the previous 7 days (flattened)
        as input and the following day as the target.

        The resulting datasets are split into training and testing
        sets while preserving temporal order and stored on the
        instance for downstream model training.

        Parameters
        ----------
        newdata : pandas.Series or numpy.ndarray
            Daily-aligned time-series data where each row represents
            one day and each column represents intra-day time steps
            (e.g., 288 steps for 5-minute resolution).

        Returns
        -------
        None
            This method populates the following instance attributes:
            - ``self.X_train``
            - ``self.X_test``
            - ``self.y_train``
            - ``self.y_test``
        """
        # Ensure data is a NumPy array with shape 
        # (num_days, steps_per_day)
        data = np.squeeze(newdata)
        # Number of past days used as input
        window_size = 7
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        # Build supervised samples: previous 7 days -> next day
        for i in range(window_size, len(data)):
            # Flatten past 7 days into a single sequence
            past_7_days = data[i - window_size : i].reshape(-1)
            # Target is the full next day
            next_day = data[i]
            X_list.append(past_7_days)
            y_list.append(next_day)
        # Convert lists to NumPy arrays
        x = np.array(X_list)
        y = np.array(y_list)
        # Reshape inputs for sequence models (samples, time_steps, 
        # channels)
        x = x.reshape(-1, 2016, 1)
        # Split data while preserving temporal order
        X_train, X_test, y_train, y_test = cast(
            Tuple[
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
            ],
            train_test_split(x, y, test_size=0.2, shuffle=False),
        )
        # Store datasets on the instance
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def create_trend_regression_model(self, datain: pd.Series) -> None:
        """
        Train a convolutional regression model to predict next-day
        trend values from historical trend data.

        This method extracts the long-term trend component from the
        input time series using seasonal decomposition, reshapes the
        trend into fixed-length daily sequences, and constructs a
        supervised learning dataset using a 7-day sliding window.
        A 1D convolutional neural network is then trained to predict
        the full next-day trend profile.

        The trained model and intermediate datasets are stored on the
        instance for later evaluation and inference.

        Parameters
        ----------
        datain : pandas.Series
            Input time-series data indexed by datetime, containing the
            signal from which the trend component will be extracted.

        Returns
        -------
        None
            This method trains a model and stores it on the instance
            but does not return a value.

        Notes
        -----
        - Seasonal decomposition is performed using an additive model
          with a yearly period.
        - Each day is assumed to contain 288 time steps (5-minute
          resolution).
        - The model predicts the entire next-day trend sequence in a
          single forward pass.
        """
        # Period for seasonal decomposition (annual cycle)
        p = 365
        # Decompose the input time series into trend, seasonal, and 
        # residual components
        decomposition = seasonal_decompose(
            datain,
            model="additive",
            period=p,
            extrapolate_trend="freq",  # pyright: ignore[reportArgumentType]
        )
        # Extract the trend component of the decomposition
        decomp_trend: pd.Series = decomposition.trend
        # Pad and align the trend series into fixed-length daily sequences
        newdata = fill_missing_times_to_padded_array(decomp_trend)
        # Optional visualization of daily trend structure
        plot_days_line_with_time(newdata)
        # Store processed trend data on the instance
        self.trend_data = newdata
        # Build supervised training and testing datasets
        self.setup_7days_train_data(newdata)
        # Define a 1D convolutional regression model
        self.tmodel = Sequential([
            Conv1D(
                filters=64,
                kernel_size=7,
                activation="relu",
                input_shape=(self.X_train.shape[1], 1),
            ),
            Conv1D(filters=64, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation="relu"),
            GlobalAveragePooling1D(),
            Dense(512, activation="relu"),
            Dense(288),  # Predict full next-day trend sequence
        ])
        # Compile the model using mean squared error loss
        self.tmodel.compile(  # pyright: ignore[reportUnknownMemberType]
            optimizer="adam",
            loss="mse",
        )
        # Train the model on historical trend data
        self.tmodel.fit(  # pyright: ignore[reportUnknownMemberType]
            self.X_train,
            self.y_train,
            epochs=300,
            batch_size=16,
            validation_split=0.1,
        )
    
class forecast_algo(object):

    def __init__(self, filename: Address, path: Address):
        """
        Initialize the data interface and modeling components.

        This constructor instantiates the time-series data handler used 
        to load and filter inverter measurements from disk, as well as 
        the trend-modeling component responsible for extracting and 
        predicting long-term trend behavior.

        Parameters
        ----------
        filename : Address
            Name or identifier of the data file containing time-series
            measurements.
        path : Address
            Filesystem or storage path pointing to the location of the 
            data file.

        Returns
        -------
        None
            This constructor initializes internal state and does not 
            return a value.
        """
        # Initialize time-series data access layer
        self.mn8b = timeseries_data(filename, path)
        # Initialize handler for trend-based modeling
        self.trend_model_handler = trend_model()
        
    def read_data(self, inv: str = SAMPLE_DEVICE) -> None:
        """
        Load, decompose, and preprocess inverter time-series data for
        downstream trend and stable-component modeling.

        This method retrieves time-series data for a selected inverter,
        extracts the target power signal, and applies seasonal 
        decomposition to separate the long-term trend from the remaining 
        stable component. Both components are then aligned and padded 
        into fixed-length daily sequences suitable for sequence-based 
        models.

        The resulting processed arrays are stored as instance attributes
        for later model training and evaluation.

        Parameters
        ----------
        inv : str, default SAMPLE_DEVICE
            Identifier of the inverter device to load and process.

        Returns
        -------
        None
            This method populates internal state and does not return a 
            value.

        Attributes Set
        --------------
        self.filter_data : pandas.DataFrame
            Filtered raw time-series data for the selected inverter.
        self.stableval : pandas.Series
            Detrended signal obtained by subtracting the trend component
            from the original measurements.
        self.trend_newdata : numpy.ndarray
            Daily-aligned and padded trend component.
        self.stable_newdata : numpy.ndarray
            Daily-aligned and padded stable (detrended) component.
        """
        # Load time-series data for the selected inverter
        self.filter_data = self.mn8b.run_on_selected_device(inv)
        # Extract the target power measurement
        datain = self.filter_data["AC_POWER.MEASURED"]
        # Seasonal decomposition period (annual cycle, per literature)
        p = 365
        # Decompose the signal into trend, seasonal, and residual 
        # components
        decomposition = seasonal_decompose(
            datain,
            model="additive",
            period=p,
            extrapolate_trend="freq",  # pyright: ignore[reportArgumentType]
        )
        # Extract the long-term trend component
        trend: pd.Series = decomposition.trend
        # Compute the stable (detrended) component of the signal
        self.stableval = datain - trend.values
        # Pad and align both components into fixed-length daily sequences
        self.trend_newdata = fill_missing_times_to_padded_array(trend)
        self.stable_newdata = fill_missing_times_to_padded_array(
            self.stableval
        )

    def train_model(self) -> None:
        """
        Train both the stable (detrended) and trend prediction models.

        This method orchestrates the full training pipeline by:
        1. Initializing and training the LSTM-based stable model on
        detrended inverter data.
        2. Extracting the raw power time series and training a separate
        trend regression model to learn long-term behavior.

        The trained models are stored on the instance for downstream
        evaluation, visualization, or inference.

        Returns
        -------
        None
            This method trains models and updates internal state but 
            does not return a value.
        """
        # Train LSTM model on detrended (stable) component
        stable_model_handler = stable_model(self.filter_data)
        self.lstm = stable_model_handler.LSTM_stable_model()
        # Extract raw power time series for trend modeling
        datain: pd.Series = self.filter_data[Metric.AC_POWER]
        # Train regression model on the trend component
        self.trend_model_handler.create_trend_regression_model(datain)

    def model_predict(self, index: int) -> pd.DataFrame:
        """
        Generate a full next-day AC power prediction by combining
        seasonal and trend model components.

        This method performs an end-to-end inference step for a given
        day index by:
        1. Generating the seasonal (detrended) prediction component.
        2. Generating the trend prediction component.
        3. Reconstructing the full signal by summing seasonal and trend
           components.
        4. Visualizing predicted vs actual values in percentage space.
        5. Returning the predicted signal aligned to local time-of-day.

        Parameters
        ----------
        index : int
            Index of the day to predict. This index must be valid for
            both the seasonal and trend datasets.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the predicted AC power values for
            the selected day, indexed by local time-of-day at 5-minute
            resolution.
        """
        # Generate seasonal (stable) component prediction and ground 
        # truth
        stable_out, truth_stable_out = self.seasonal_prediction(index)
        # Generate trend component prediction and corresponding targets
        pred, y_trend = self.trend_prediction(index)
        # Combine seasonal and trend components to reconstruct
        # the full predicted and true signals
        p: np.ndarray = (
            pred[0] + np.asarray(stable_out).flatten()
        )
        o: np.ndarray = (
            y_trend[index] + np.asarray(truth_stable_out).flatten()
        )
        # Normalize both curves to percentage scale for visualization
        max_val = max(o.max(), p.max())
        o_pct = (o / max_val) * 100
        p_pct = (p / max_val) * 100
        # Plot reconstructed prediction vs ground truth (percentage 
        # scale)
        plot_helper(
            figsize=(14, 4),
            lines=[
                PlotLineSpec(
                    y=o_pct,
                    plot_type="line",
                    label="Actual (%)",
                    linewidth=2,
                ),
                PlotLineSpec(
                    y=p_pct,
                    plot_type="line",
                    label="Predicted (%)",
                    linewidth=2,
                ),
            ],
            title=f"AC Power Prediction vs Actual - Day {index}",
            xlabel="Time Step (5-minute intervals)",
            ylabel="AC Power (%)",
            grid=True,
            tightlayout=True,
        )
        # Convert predicted values back into a time-indexed DataFrame
        # using local time-of-day
        output = self.add_event_local_time(p)

        return output

    def seasonal_prediction(
            self,
            index: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and visualize a next-day seasonal (stable component)
        prediction for a specific test index.

        This method:
        1. Constructs supervised 7-day input/output samples from the
        stable (detrended) component of the time series.
        2. Runs inference for a selected test index using the trained
        stable LSTM model.
        3. Visualizes both the model prediction and the ground-truth
        target in scaled space.
        4. Converts the scaled outputs back to the original data scale
        using the reference stable signal.
        5. Returns both the rescaled prediction and rescaled ground 
        truth for downstream analysis or plotting.

        Parameters
        ----------
        index : int
            Index into the test dataset corresponding to the day to be
            predicted. Must be within the valid range of the constructed
            supervised dataset.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple ``(stable_out, truth_stable_out)`` where:
            - ``stable_out`` is the model-predicted next-day stable
            component in original data scale.
            - ``truth_stable_out`` is the corresponding ground-truth
            stable component in original data scale.
        """
        # Build supervised (X, y) datasets from stable 
        # (seasonal-adjusted) daily data
        Xstable, ystable = self.form_7days_stable_data(self.stable_newdata)
        # Basic shape diagnostics for sanity checking
        print("X shape:", Xstable.shape)
        print("y shape:", ystable.shape)
        # Run model inference for the specified index
        out: np.ndarray = self.test_stable_with_index(
            index,
            Xstable,
            ystable,
        )
        # Plot ground-truth stable data (scaled space)
        plot_helper(
            figsize=(15, 4),
            lines=[
                PlotLineSpec(
                    x=np.arange(len(ystable[index])),
                    y=ystable[index],
                    plot_type="scatter",
                    label="Ground Truth (Scaled)",
                    color="c",
                    marker="o",
                )
            ],
            title="AC Power Stable Component (Ground Truth)",
            xlabel="Data index",
            ylabel="Value",
            grid=False,
        )
        # Plot predicted stable data (scaled space)
        plot_helper(
            figsize=(15, 4),
            lines=[
                PlotLineSpec(
                    x=np.arange(len(out)),
                    y=out,
                    plot_type="scatter",
                    label="Prediction (Scaled)",
                    color="r",
                    marker="o",
                )
            ],
            title="AC Power Stable Component (Prediction)",
            xlabel="Data index",
            ylabel="Value",
            grid=False,
        )
        # Convert reference stable values to a 2D array for scaling
        # (required by convert_data)
        vals: np.ndarray = cast(np.ndarray, self.stableval.values)
        stable_data_2d: np.ndarray = vals.reshape(-1, 1)
        # Rescale prediction and ground truth back to original value 
        # space
        stable_out: np.ndarray = self.convert_data(
            stable_data_2d,
            out,
        )
        truth_stable_out: np.ndarray = self.convert_data(
            stable_data_2d,
            ystable[index],
        )

        return stable_out, truth_stable_out

    def trend_prediction(
            self,
            index: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and visualize a next-day trend prediction for a given 
        index.

        This method evaluates the trained trend regression model by:
        1. Constructing supervised 7-day trend inputs and targets.
        2. Selecting a single sample at the specified index.
        3. Running a forward pass through the trained trend model.
        4. Normalizing both prediction and ground truth to percentages
           relative to a shared maximum value.
        5. Plotting predicted versus actual trend profiles for visual
           comparison.

        Parameters
        ----------
        index : int
            Index of the day to predict within the supervised trend
            dataset. The index must be valid for the internally
            constructed 7-day sliding window data.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing:
            - ``pred`` : numpy.ndarray
                Model prediction with shape ``(1, 288)``, representing
                the predicted trend values for the selected day.
            - ``y_trend`` : numpy.ndarray
                Full array of ground-truth trend targets used for
                evaluation.
        """
        # Construct supervised trend datasets using a 7-day window
        X_trend, y_trend = self.setup_7days_trend_data(self.trend_newdata)
        # Select a single input sample and expand batch dimension
        x_single = np.expand_dims(
            X_trend[index],
            axis=0
        )  # Shape: (1, 2016, 1)
        # Run model inference for the selected day
        pred = cast(
            np.ndarray,
            self.trend_model_handler
                .tmodel
                .predict(x_single)  # pyright: ignore[reportUnknownMemberType]
        )
        # Debug: confirm prediction shape (expected: (1, 288))
        print(pred.shape)
        # Extract ground-truth and predicted daily sequences
        y_true: np.ndarray = y_trend[index]
        y_pred: np.ndarray = pred[0]
        # Normalize both curves to percentages using a shared maximum
        max_val = max(y_true.max(), y_pred.max())
        y_true_pct = (y_true / max_val) * 100
        y_pred_pct = (y_pred / max_val) * 100
        # Plot normalized actual vs predicted trend profiles
        plot_helper(
            figsize=(14, 4),
            lines=[
                PlotLineSpec(
                    y=y_true_pct,
                    plot_type="line",
                    label="Actual (%)",
                    linewidth=2,
                ),
                PlotLineSpec(
                    y=y_pred_pct,
                    plot_type="line",
                    label="Predicted (%)",
                    linewidth=2,
                ),
            ],
            title=f"AC Power Prediction vs Actual - Day {index}",
            xlabel="Time Step (5-minute intervals)",
            ylabel="AC Power (%)",
            grid=True,
            tightlayout=True,
        )

        return pred, y_trend
        
    def convert_data(
            self,
            main_data: np.ndarray,
            target_data: np.ndarray,
        ) -> np.ndarray:
        """
        Rescale normalized target data back to the original scale of a
        reference dataset and visualize the transformation.

        This method performs a manual inverse min-max scaling using the
        minimum and maximum values of ``main_data``. The scaled values 
        in ``target_data`` are assumed to lie in the range [0, 1] and 
        are mapped back to the original value range of ``main_data``.

        Two diagnostic plots are produced:
        1. The input (scaled) target data.
        2. The rescaled output data in the original value domain.

        Parameters
        ----------
        main_data : numpy.ndarray
            Reference array defining the original data scale. Its 
            minimum and maximum values are used to rescale 
            ``target_data``.
        target_data : numpy.ndarray
            Normalized data (typically model output) assumed to be 
            scaled to the range [0, 1].

        Returns
        -------
        numpy.ndarray
            Rescaled data mapped back to the original scale of
            ``main_data``.
        """
        # Compute min–max statistics from the reference data
        maxval: float = float(np.max(main_data))
        minval: float = float(np.min(main_data))
        value_range: float = maxval - minval
        # Inverse min–max scaling:
        # original_value = scaled_value * (max - min) + min
        out: np.ndarray = target_data * value_range + minval
        # Plot scaled (input) target data
        plot_helper(
            figsize=(15, 4),
            lines=[
                PlotLineSpec(
                    x=np.arange(len(target_data)),
                    y=target_data,
                    plot_type="line",
                    label="Scaled Input Data",
                    color="c",
                    marker="o",
                )
            ],
            title="AC Power Data (Scaled)",
            xlabel="Data index",
            ylabel="Value",
            grid=False,
        )
        # Plot rescaled output data
        plot_helper(
            figsize=(15, 4),
            lines=[
                PlotLineSpec(
                    x=np.arange(len(out)),
                    y=out,
                    plot_type="line",
                    label="Rescaled Output Data",
                    color="r",
                    marker="*",
                )
            ],
            title="AC Power Data (Rescaled)",
            xlabel="Data index",
            ylabel="Value",
            grid=False,
        )

        return out

    def form_7days_stable_data(
            self,
            datain: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct supervised learning data for stable (detrended) 
        modeling using a 7-day sliding window.

        This method takes daily-aligned stable time-series data, applies
        Min-Max scaling, and converts it into input-target pairs 
        suitable for sequence-to-sequence LSTM training. Each training 
        sample uses the previous 7 days (flattened) to predict the full 
        next day.

        Parameters
        ----------
        datain : numpy.ndarray
            Array of shape ``(num_days, 288)`` or ``(num_days, 288, 1)``
            containing stable (detrended) daily time-series data at
            5-minute resolution.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple ``(X, y)`` where:
            - ``X`` has shape ``(num_samples, 2016, 1)``, representing 
              the flattened 7-day input windows.
            - ``y`` has shape ``(num_samples, 288, 1)``, representing 
              the next-day targets.
        """
        # Flatten data for scaling while preserving temporal order
        X_flat = datain.reshape(-1, 1)
        # Fit Min–Max scaler on the full stable signal
        scaler = MinMaxScaler()
        X_scaled_flat = (
            scaler
            .fit_transform(X_flat)  # pyright: ignore[reportUnknownMemberType]
        )
        # Restore original daily structure and add channel dimension
        X_scaled = X_scaled_flat.reshape(datain.shape)
        X_scaled = X_scaled[..., np.newaxis]
        # Number of past days used as input
        window_size = 7
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        # Build supervised samples: previous 7 days -> next day
        for i in range(window_size, len(X_scaled)):
            # Flatten the previous 7 days into a single input sequence
            past_7_days = X_scaled[i - window_size : i].reshape(-1)
            # Target is the full next day sequence
            next_day = X_scaled[i]
            X_list.append(past_7_days)
            y_list.append(next_day)
        # Convert collected samples to NumPy arrays
        x = np.array(X_list)
        y = np.array(y_list)
        # Reshape inputs for LSTM: (samples, time_steps, channels)
        # 7 days × 288 steps/day = 2016 time steps
        x = x.reshape((x.shape[0], 7 * 288, 1))
        y = y.reshape((y.shape[0], 288, 1))

        return x, y

    def setup_7days_trend_data(
            self,
            newdata: Union[np.ndarray, pd.Series],
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct supervised learning data for trend modeling using a
        7-day sliding window.

        This method converts daily-aligned trend data into input-target
        pairs suitable for sequence models. Each input sample consists 
        of the previous 7 days of trend values (flattened into a single
        sequence), and the corresponding target is the full trend 
        profile of the next day.

        Unlike the stable-data pipeline, this method assumes the input
        data is already detrended and aligned into fixed-length daily
        sequences (e.g., 288 time steps per day).

        Parameters
        ----------
        newdata : numpy.ndarray or pandas.Series
            Daily-aligned trend data where each row represents one day 
            and each column represents intra-day time steps. Typical 
            shape is ``(num_days, 288)`` or ``(num_days, 288, 1)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple ``(X, y)`` where:
            - ``X`` has shape ``(num_samples, 2016, 1)``, representing 
              the flattened 7-day input windows.
            - ``y`` has shape ``(num_samples, 288)``, representing the
              next-day trend targets.
        """
        # Ensure data is a NumPy array with shape (num_days, 
        # steps_per_day)
        data = np.squeeze(newdata)
        # Number of past days used as input
        window_size = 7
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        # Build supervised samples: previous 7 days -> next day
        for i in range(window_size, len(data)):
            # Flatten past 7 days into a single sequence
            past_7_days = data[i - window_size : i].reshape(-1)
            # Target is the full next day
            next_day = data[i]
            X_list.append(past_7_days)
            y_list.append(next_day)
        # Convert lists to NumPy arrays
        x = np.array(X_list)
        y = np.array(y_list)
        # Reshape inputs for sequence models: (samples, time_steps, 
        # channels) 7 days × 288 steps/day = 2016 time steps
        x = x.reshape(-1, 2016, 1)

        return x, y

    def test_stable_with_index(
            self,
            index: int,
            X: np.ndarray,
            y: np.ndarray,
        ) -> np.ndarray:
        """
        Run a stable-component LSTM prediction for a specific index and
        visualize the result against the ground truth.

        This method extracts the historical input window corresponding
        to the requested index, performs a forward pass through the
        trained LSTM model, reshapes the prediction to daily resolution,
        and plots the predicted versus actual stable signal for visual
        inspection.

        Parameters
        ----------
        index : int
            Index of the target day to be predicted. This index must be
            valid for the supervised dataset and implicitly assumes
            that the preceding 7 days are available.
        X : numpy.ndarray
            Input feature array containing flattened 7-day windows.
            Expected shape: ``(n_samples, 2016, 1)``.
        y : numpy.ndarray
            Target array containing next-day stable values.
            Expected shape: ``(n_samples, 288, 1)`` or 
            ``(n_samples, 288)``.

        Returns
        -------
        numpy.ndarray
            The predicted stable component for the selected day,
            reshaped to ``(288, 1)``.
        """
        # Extract the supervised input window for the requested index.
        # NOTE: This slicing assumes X already encodes 7-day windows;
        # the first element is selected after slicing.
        a = X[index - 7 : index]
        # Ground-truth stable values for the target day
        b = y[index]
        # Select the actual 7-day input window and reshape for inference
        a = a[0]
        a = a.reshape(1, a.shape[0], 1)  # (1, 2016, 1)
        # Run model inference (single-day prediction)
        m = cast(
            np.ndarray,
            self.lstm.predict( # pyright: ignore[reportUnknownMemberType]
                a
            ),
        )
        # Reshape model output to daily resolution
        m = m.reshape(288, 1)
        # Plot predicted vs actual stable values for visual comparison
        plot_helper(
            figsize=(12, 6),
            lines=[
                PlotLineSpec(
                    x=np.arange(len(m)),
                    y=m,
                    plot_type="line",
                    label=f"Predicted (index {index})",
                    color="y",
                    marker="o",
                ),
                PlotLineSpec(
                    x=np.arange(len(b)),
                    y=b,
                    plot_type="line",
                    label=f"Actual (index {index})",
                    color="b",
                    marker="o",
                ),
            ],
            title="AC Power Stable Component Prediction",
            xlabel="Data index",
            ylabel="Value",
            grid=False,
        )

        return m

    def add_event_local_time(self, data: np.ndarray) -> pd.DataFrame:
        """
        Attach a local time-of-day index to a single-day prediction 
        array.

        This method converts a one-day prediction vector (assumed to be
        sampled at 5-minute resolution) into a pandas DataFrame indexed
        by local time-of-day. The resulting index spans a full 24-hour
        period from ``00:00`` to ``23:55`` with 288 total time steps.

        This is primarily intended for post-processing and inspection
        of model outputs, enabling time-aligned visualization and
        comparison with original time-series data.

        Parameters
        ----------
        data : numpy.ndarray
            One-dimensional NumPy array containing predicted or observed
            values for a single day. The array is expected to have
            length 288, corresponding to 5-minute intervals over
            24 hours.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by local time-of-day (5-minute
            resolution) with a single column named ``"value"``.
        """
        # Create a local time-of-day index covering one full day
        # at 5-minute resolution (24 * 60 / 5 = 288 steps)
        time_index = pd.date_range(
            start="00:00",
            end="23:55",   # Last 5-minute slot in a 24-hour day
            freq="5min",
        )
        # Optional sanity check: ensure index length matches data
        print(len(time_index))  # Expected: 288
        # Build a DataFrame using the generated time index
        df = pd.DataFrame(
            {"value": data},
            index=time_index,
        )
        # Name the index to match the project's timestamp convention
        df.index.name = Column.TIMESTAMP
        # Debug-friendly printout of the resulting structure
        print("-" * 12, f" PREDICT OUTPUT WITH {Column.TIMESTAMP} ", "-" * 12)
        print(df)

        return df

def main() -> None:
    """
    Execute the end-to-end forecasting workflow.

    This function serves as the primary entry point for running the
    forecasting pipeline. It performs the following steps in order:

    1. Applies the global dark theme configuration for all plots.
    2. Instantiates the forecasting algorithm with the required
       dataset path and filename.
    3. Loads and preprocesses the inverter time-series data.
    4. Trains both the seasonal (LSTM-based) and trend models.
    5. Runs a sample prediction for a selected day index and produces
       diagnostic plots.

    Returns
    -------
    None
        This function executes the workflow for its side effects
        (training, plotting, and prediction) and does not return a
        value.
    """
    # Apply consistent plotting style across the entire application
    use_dark_theme()
    # Initialize the forecasting pipeline with dataset location
    forecast = forecast_algo(
        "forecast_data.parquet",
        DATA_ROOT,
    )
    # Load and preprocess inverter data
    forecast.read_data()
    # Train seasonal and trend prediction models
    forecast.train_model()
    # Example day index for prediction and visualization
    index: int = 41
    # Run model inference and generate plots
    _ = forecast.model_predict(index)

if __name__ == "__main__":
    main()