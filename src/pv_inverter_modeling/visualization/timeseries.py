from datetime import datetime, timedelta
from typing import Union, Optional, Tuple
# thirdpartylib
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from matplotlib.axes import Axes
# projectlib
from pv_inverter_modeling.utils.typing import DataFrame, Field, Column
from pv_inverter_modeling.data.schemas import KEYS


def use_dark_theme() -> None:
    """
    Apply a shadcn-inspired dark theme to Matplotlib.

    This function updates Matplotlib's global rcParams to use a dark 
    color palette with subtle gridlines, muted text, and a modern line 
    color cycle suitable for time-series and dashboard-style plots.
    """
    # Dark theme
    mpl.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#171717",
        "axes.edgecolor": "#ffffff1a",
        "axes.labelcolor": "#fafafa",
        "axes.titlecolor": "#fafafa",
        "grid.color": "#ffffff1a",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.2,
        "xtick.color": "#a1a1a1",
        "ytick.color": "#a1a1a1",
        "lines.linewidth": 1.6,
        "text.color": "#e5e7eb",
        "legend.edgecolor": "#ffffff1a",
        "legend.facecolor": "#171717",
        "legend.fontsize": 9,
        "legend.frameon": True,
        "axes.grid": True,
        "axes.prop_cycle": cycler(color=[
            "#1447e6", 
            "#00bc7d", 
            "#fe9a00",
            "#ad46ff",
            "#ff2056",
        ]),
    })


class Plot(object):
    """
    Plotting utilities for time-series metrics with dark-style theming.

    This class provides a unified interface for plotting metrics from 
    pandas and Polars data structures, handling datetime normalization, 
    optional filtering by date range and device, and consistent visual 
    styling.
    """

    def __init__(self) -> None:
        """
        Initialize the plotter and apply the default dark-style 
        Matplotlib theme.
        """
        use_dark_theme()

    def __normalize_datetime(
            self, 
            value: Union[datetime, str, None],
            *,
            format: Optional[str] = None
        ) -> Tuple[Optional[pd.Timestamp], Optional[datetime]]:
        """
        Normalize a datetime-like input into both pandas and Python 
        datetime forms.

        Accepts a Python datetime, a string parseable by 
        pandas.to_datetime, or None. Returns a tuple containing:

        - a pandas.Timestamp representation (or None), and
        - a native datetime.datetime representation (or None).

        This dual representation is useful when working across pandas- 
        and non-pandas APIs (e.g., pandas vs. Polars).

        Parameters
        ----------
        value : datetime | str | None
            Datetime-like value to normalize.
        format : str, optional
            Optional datetime format string passed to pandas.to_datetime 
            when value is a string.

        Returns
        -------
        tuple[pandas.Timestamp | None, datetime.datetime | None]
            A tuple (timestamp, datetime) representing the same instant 
            in pandas and Python datetime formats, or (None, None) if 
            value is None.
        """
        if value is None:
            return None, None

        if isinstance(value, str):
            ts = pd.to_datetime(value, format=format)
            return ts, ts.to_pydatetime()
        else:
            ts = pd.Timestamp(value)
            return ts, value
        
    def __normalize_timedelta(
            self,
            value: Optional[Union[timedelta, str]],
    ) -> Tuple[Optional[pd.Timedelta], Optional[timedelta]]:
        """
        Normalize a timedelta-like input into both pandas and Python 
        timedelta forms.

        Accepts a Python datetime.timedelta, a string parseable by
        pandas.to_timedelta, or None. Returns a tuple containing:

        - a pandas.Timedelta representation (or None), and
        - a native datetime.timedelta representation (or None).

        This helper enables consistent handling of time windows across 
        pandas and non-pandas APIs.

        Parameters
        ----------
        value : timedelta | str | None
            Timedelta-like value to normalize.

        Returns
        -------
        tuple[pandas.Timedelta | None, datetime.timedelta | None]
            A tuple (timedelta, timedelta_py) representing the same 
            duration in pandas and Python timedelta formats, or 
            (None, None) if value is None.
        """
        if value is None:
            return None, None

        if isinstance(value, str):
            td = pd.to_timedelta( # pyright: ignore[reportUnknownMemberType]
                value
            )
            return td, td.to_pytimedelta()
        else:
            td = pd.Timedelta(value)
            return td, value

    def _plot_pandas(
            self,
            data: pd.DataFrame,
            x: Field,
            y: Field,
            ax: Axes,
            *,
            dropna: bool = True,
            devices: Optional[Tuple[str, ...]] = None,
            start: Optional[pd.Timestamp] = None,
            end: Optional[pd.Timestamp] = None,
            filtered: bool = False,
        ) -> Axes:
        """
        Plot a metric against time (or another field) from a pandas 
        DataFrame.

        This method performs optional filtering (by date range, device 
        selection, and missing values), then plots one line per device 
        onto the provided Matplotlib Axes. A shaded area is drawn 
        beneath each line, extending down to the global minimum of the 
        plotted values to provide visual context similar to shadcn-style
        line charts.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data containing at least the timestamp column, device 
            identifier, and the fields specified by x and y.
        x : Field
            Column name to use for the x-axis.
        y : Field
            Column name to use for the y-axis.
        ax : matplotlib.axes.Axes
            Axes object on which to draw the plot.
        dropna : bool, default True
            If True, drop rows with missing values in x or y before
            plotting.
        devices : tuple[str, ...], optional
            Optional subset of device identifiers to plot. If None, all 
            devices present in the data are plotted.
        start : pandas.Timestamp, optional
            Inclusive start timestamp used to filter the data.
        end : pandas.Timestamp, optional
            Exclusive end timestamp used to filter the data.
        filtered : bool, default False
            If True, assume the input data has already been filtered and
            skip all internal filtering steps. This is typically used 
            when the data originates from a Polars pipeline.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the rendered plot.
        """
        df = data.copy()
        if not filtered:
            # Filter dates
            if start is not None:
                df = df[df[Column.TIMESTAMP] >= start]
            if end is not None:
                df = df[df[Column.TIMESTAMP] < end]
            if devices is not None:
                mask = (
                    df[Column.DEVICE]
                    .isin( # pyright: ignore[reportUnknownMemberType]
                        devices
                    )
                )
                df = df[mask]
            if dropna:
                df = df.dropna( # pyright: ignore[reportUnknownMemberType]
                    subset=[x, y]
                )
            df = df.sort_values([*KEYS])
        if df.empty:
            return ax
        # Plot
        baseline = df[y].min()
        data_groups = df.groupby( # pyright: ignore[reportUnknownMemberType]
            Column.DEVICE
        )
        for device, g in data_groups:
            line, = ax.plot( # pyright: ignore[reportUnknownMemberType]
                g[x],
                g[y],
                label=str(device),
            )
            ax.fill_between( # pyright: ignore[reportUnknownMemberType]
                g[x],
                g[y],
                baseline,
                color=line.get_color(),
                alpha=0.15,
            )
        
        ax.set_xlabel(x) # pyright: ignore[reportUnknownMemberType]
        ax.set_ylabel(y) # pyright: ignore[reportUnknownMemberType]
        ax.set_title(f"{y} vs {x}") # pyright: ignore[reportUnknownMemberType]
        if df[Column.DEVICE].nunique() > 1:
            ax.legend() # pyright: ignore[reportUnknownMemberType]

        return ax
    
    def _plot_polars_lazyframe(
            self,
            data: pl.LazyFrame,
            x: Field,
            y: Field,
            ax: Axes,
            *,
            dropna: bool = True,
            devices: Optional[Tuple[str, ...]] = None,
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
        ) -> Axes:
        """
        Plot a metric from a Polars LazyFrame using lazy filtering and 
        pandas rendering.

        This method applies all filtering operations lazily using Polars
        (including date range filtering, device selection, and optional 
        null removal), reduces the data to the required columns, and 
        then materializes the result as a pandas DataFrame for plotting.
        The actual rendering is delegated to _plot_pandas.

        Parameters
        ----------
        data : polars.LazyFrame
            Input lazy frame containing the timestamp column, device 
            identifier, and the fields specified by x and y.
        x : Field
            Column name to use for the x-axis.
        y : Field
            Column name to use for the y-axis.
        ax : matplotlib.axes.Axes
            Axes object on which to draw the plot.
        dropna : bool, default True
            If True, drop rows with null values in x or y before
            collecting the data.
        devices : tuple[str, ...], optional
            Optional subset of device identifiers to include. If None, 
            all devices present in the data are included.
        start : datetime.datetime, optional
            Inclusive start datetime used to filter the data.
        end : datetime.datetime, optional
            Exclusive end datetime used to filter the data.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the rendered plot.
        """
        lf = data
        if dropna:
            lf = lf.drop_nulls(subset=[x, y])
        # Filter for dates if passed
        if start is not None:
            lf = lf.filter(pl.col(Column.TIMESTAMP) >= start)
        if end is not None:
            lf = lf.filter(pl.col(Column.TIMESTAMP) < end)
        # Filter for devices
        if devices is not None:
            lf = lf.filter(
                pl.col(Column.DEVICE).is_in(devices)
            )
        # Define columns to keep for dimension reduction 
        select_cols = tuple(dict.fromkeys((*KEYS, x, y)))
        # Select columns, sort, and collect to pandas
        df = lf.select(select_cols).sort(KEYS).collect().to_pandas()

        return self._plot_pandas(df, x, y, ax=ax, filtered=True)

    def plot_metric(
            self, 
            data: DataFrame, 
            x: Field,
            y: Field,
            *,
            dropna: bool = True,
            devices: Optional[Tuple[str, ...]] = None,
            ax: Optional[Axes] = None,
            format: Optional[str] = None,
            single_day: bool = False,
            start: Optional[Union[datetime, str]] = None,
            end: Optional[Union[datetime, str]] = None,
            window: Optional[Union[timedelta, str]] = None
        ) -> Axes:
        """
        Plot a metric against time (or another field) from pandas or 
        Polars data.

        This is the public plotting entry point. It validates and 
        normalizes datetime-related arguments, optionally derives an end
        time from a window, and dispatches to the appropriate backend 
        depending on whether the input data is a pandas DataFrame, a 
        Polars DataFrame, or a Polars LazyFrame.

        Filtering by date range, device selection, and missing values is
        supported. When a Polars LazyFrame is provided, all filtering is
        performed lazily prior to materialization.

        Parameters
        ----------
        data : pandas.DataFrame | polars.DataFrame | polars.LazyFrame
            Input data containing a timestamp column, device identifier,
            and the fields specified by x and y.
        x : Field
            Column name to use for the x-axis.
        y : Field
            Column name to use for the y-axis.
        dropna : bool, default True
            If True, drop rows with missing values in x or y before
            plotting.
        devices : tuple[str, ...], optional
            Optional subset of device identifiers to plot. If None, all 
            devices present in the data are plotted.
        ax : matplotlib.axes.Axes, optional
            Axes object on which to draw the plot. If None, a new figure
            and axes are created.
        format : str, optional
            Optional datetime format string used when parsing 
            string-valued start or end arguments.
        single_day : bool, default False
            If True, plot a single day of data starting at start. This 
            mode cannot be combined with end or window.
        start : datetime | str, optional
            Inclusive start datetime for filtering the data. Required if
            end, window, or single_day is specified.
        end : datetime | str, optional
            Exclusive end datetime for filtering the data. Cannot be 
            combined with window.
        window : timedelta | str, optional
            Time window duration used to derive end from start. Cannot 
            be combined with end.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the rendered plot.

        Raises
        ------
        ValueError
            If mutually exclusive arguments are provided (e.g., end and
            window together), or if required arguments are missing 
            (e.g., start when end or window is specified).
        """
        # General validation
        if end is not None and window is not None:
            raise ValueError("Specify only one of `end` or `window`.")
        if end is not None and start is None:
            raise ValueError(
                "`start` must be specified if `end` is provided."
            )
        if window is not None and start is None:
            raise ValueError(
                "`start` must be specified if `window` is provided."
            )
        # Single-day mode
        if single_day:
            if start is None:
                raise ValueError(
                    "`start` must be specified when "
                    "`single_day=True`."
                )
            if end is not None or window is not None:
                raise ValueError(
                    "`single_day=True` cannot be combined with `end` "
                    "or `window`."
                )
            
            window = timedelta(days=1)

        if ax is None:
            _, ax = plt.subplots() # pyright: ignore[reportUnknownMemberType]
        # Normalize datetime inputs
        start_ts, start_py = self.__normalize_datetime(start, format=format)
        end_ts, end_py = self.__normalize_datetime(end, format=format)
        window_td, window_py = self.__normalize_timedelta(window)
        # Derive end from window if needed
        if start_ts is not None and window_td is not None:
            # Assertions for type checker
            assert start_py is not None
            assert window_py is not None

            end_ts = start_ts + window_td
            end_py = start_py + window_py
        
        if isinstance(data, pl.LazyFrame):
            return self._plot_polars_lazyframe(
                data,
                x,
                y,
                ax,
                dropna=dropna,
                devices=devices,
                start=start_py,
                end=end_py
            )
        elif isinstance(data, pl.DataFrame):
            return self._plot_pandas(
                data.to_pandas(),
                x,
                y,
                ax,
                dropna=dropna,
                devices=devices,
                start=start_ts,
                end=end_ts
            )
        else:
            return self._plot_pandas(
                data,
                x,
                y,
                ax,
                dropna=dropna,
                devices=devices,
                start=start_ts,
                end=end_ts
            )
        

