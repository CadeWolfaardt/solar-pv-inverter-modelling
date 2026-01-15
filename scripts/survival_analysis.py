# stdlib
from typing import List
# thirdpartylib
import polars as pl
import numpy as np
# projectlib
from pv_inverter_modeling.utils.typing import Address, Field
from pv_inverter_modeling.utils.paths import validate_address
from pv_inverter_modeling.data.schemas import Column, Metric
from pv_inverter_modeling.data.loaders import load_lazyframe
from pv_inverter_modeling.config.private_map import REVERSE_ENTITY_MAP
from pv_inverter_modeling.models.survival_analysis import KaplanMeierModel
from pv_inverter_modeling.config.env import DATA_ROOT, SA_SAMPLE_DEVICE

class DataPipeline(object):
    """
    End-to-end preprocessing pipeline for inverter time-series data.

    This class is responsible for:
    - Loading a subset of relevant columns from disk as a Polars 
      LazyFrame
    - Filtering to AC-related metrics
    - Ensuring missing values are handled consistently
    - Converting sparse intraday time series into fixed-length,
      day-aligned arrays suitable for ML models (e.g., sequence models)
    """

    def __init__(self, path: Address, file: Address) -> None:
        """
        Initialize the data pipeline and eagerly apply AC filtering.

        Parameters
        ----------
        path : Address
            Base directory containing the data file.
        file : Address
            Filename (or relative path) of the dataset to load.
        """
        self.path = validate_address(path)
        self.file = file
        self.source = self.path / file
         # Columns required for downstream processing
        self.cols: List[Field] = [
            Column.TIMESTAMP,
            Column.DEVICE,
            Column.TYPE,
            Column.METRIC,
            Column.VALUE,
        ]
        # Load lazily and immediately restrict to required columns
        lf = load_lazyframe(self.source).select(self.cols)
        # Apply AC-only filtering and null handling
        self.filter_AC_data(lf)

    def filter_AC_data(self, lf: pl.LazyFrame) -> None:
        """
        Filter the dataset to AC-related metrics and normalize missing 
        values.

        This method:
        - Keeps only metrics whose names start with ``"AC_"``
        - Detects whether any nulls are present
        - Replaces all null values with zeros (explicitly encoding 
          absence)

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing raw metric data.
        """
        self.lf_filtered = lf.filter(
            pl.col(Column.METRIC).str.starts_with('AC_')
        )
        # Detect presence of any null values (short-circuited at 
        # collect)
        has_null = (
            self.lf_filtered
            .select(pl.any_horizontal(pl.all().is_null().any()))
            .collect()
            .item()
        )
        print(f"Does the DataFrame contain any NaN values? {has_null}")
        # Replace nulls with zero for downstream numerical processing
        self.lf_filtered = self.lf_filtered.fill_null(0)

    def fill_missing_times_to_padded_array_polars(
            self,
            lf: pl.LazyFrame,
            *,
            freq: str = '5m',
        ) -> np.ndarray:
        """
        Convert sparse intraday time-series data into fixed-length daily 
        arrays.

        For each calendar day:
        - Duplicate timestamps are aggregated via mean
        - A full intraday timestamp grid is generated at the specified 
          frequency
        - Missing intervals are filled with zeros
        - Only complete days (exactly 288 intervals for 5-minute data) 
          are kept

        The result is a dense NumPy array suitable for ML models:
        ``(num_days, 288)``.

        Parameters
        ----------
        lf : pl.LazyFrame
            LazyFrame containing timestamped metric values.
        freq : str, default "5m"
            Intraday sampling frequency (e.g., "5m", "10m").

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_days, steps_per_day) containing
            zero-filled daily time series.
        """
        # Ensure timestamp column is a proper datetime type
        lf = lf.with_columns(
            pl.col(Column.TIMESTAMP).cast(pl.Datetime("us"))
        )
        # Extract calendar day for grouping
        lf = lf.with_columns(
            pl.col(Column.TIMESTAMP).dt.date().alias("day")
        )
        # Aggregate duplicate timestamps within each day
        lf = (
            lf
            .group_by(["day", Column.TIMESTAMP])
            .agg(pl.col(Column.VALUE).mean().alias(Column.VALUE))
        )
        # Generate a dense intraday timestamp grid per calendar day so 
        # that sparse time-series data can be left-joined and missing 
        # intervals explicitly represented and filled.
        full_grid = (
            lf
            .select("day")
            .unique()
            .with_columns(
                pl.datetime_ranges(
                    pl.col("day").cast(pl.Datetime),
                    (pl.col("day") + pl.duration(days=1)).cast(pl.Datetime),
                    interval=freq,
                    closed="left",
                ).alias(Column.TIMESTAMP)
            )
            .explode(Column.TIMESTAMP)
        )
        # Left join sparse data onto full grid and fill missing values
        lf_full = (
            full_grid
            .join(lf, on=["day", Column.TIMESTAMP], how="left")
            .with_columns(pl.col(Column.VALUE).fill_null(0.0))
        )
        # Retain only complete days (e.g., exactly 288 intervals for 
        # 5-minute data)
        lf_full = (
            lf_full
            .with_columns(
                pl.len().over("day").alias("n_per_day")
            )
            .filter(pl.col("n_per_day") == 288)
            .drop("n_per_day")
            .sort(["day", Column.TIMESTAMP])
        )
        # Materialize once and reshape to (num_days, 288)
        padded_array = (
            lf_full
            .select(Column.VALUE)
            .collect()
            .to_numpy()
            .reshape(-1, 288)
        )

        return padded_array
        
    def select_inverter_km(self, inverter: str) -> np.ndarray:
        """
        Extract a padded daily AC power time series for a single 
        inverter.

        This method:
        - Filters the dataset to a specific inverter
        - Restricts to inverter-level AC power measurements
        - Produces a fixed-length daily array representation

        Parameters
        ----------
        inverter : str
            Inverter identifier to select.

        Returns
        -------
        np.ndarray
            Dense daily AC power time series of shape (num_days, 288).
        """
        # Filter to the selected inverter and AC power metric
        data = (
            self.lf_filtered
            .filter(
                pl.col(Column.DEVICE) == inverter,
                pl.col(Column.TYPE) == "Inverter",
                pl.col(Column.METRIC) == REVERSE_ENTITY_MAP[Metric.AC_POWER],
            )
            .sort(Column.TIMESTAMP)
        )
        # Convert to padded daily representation
        data_padded = self.fill_missing_times_to_padded_array_polars(data)
        
        return data_padded
    
def main() -> None:
    """
    Entry point for Kaplan-Meier-based inverter degradation analysis.

    This routine initializes the data pipeline, extracts padded daily
    productivity sequences for a sample inverter, fits a Kaplan-Meier
    survival model, and reports an early-warning horizon based on
    productivity degradation.
    """
    # Root directory for data
    path = DATA_ROOT
    # Input parquet file
    file = "sa_data.parquet"
    # Initialize data ingestion and preprocessing pipeline
    data_pipeline = DataPipeline(path=path, file=file)
    # Extract padded daily productivity array for a sample inverter
    data = data_pipeline.select_inverter_km(SA_SAMPLE_DEVICE)
    # Fit Kaplan–Meier survival model and issue early-warning signal
    klm = KaplanMeierModel(data)
    klm.early_warning()

if __name__ == "__main__":
    main()