# stdlib
import argparse
# thirdpartylib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import polars as pl
# projectlib
from pv_inverter_modeling.config.env import (
    DATA_ROOT, 
    ANOMALY_DATA, 
    ANOMALY_P_RATED, 
    ANOMALY_SAMPLE_DEVICE
)
from pv_inverter_modeling.config.constants import MIN_POA
from pv_inverter_modeling.data.loaders import Open
from pv_inverter_modeling.preprocessing.outliers import OutlierDetector
from pv_inverter_modeling.data.schemas import Metric, Column, KEYS
from pv_inverter_modeling.utils.runtime import running_in_ipython_kernel

def parse_args() -> argparse.Namespace:
    """Parse input arguments for Anomaly Detection."""
    parser = argparse.ArgumentParser(
        description="Run anomaly detection pipeline",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help=(
            "Verbosity level: "
            "0 = silent, "
            "1 = info, "
            "2 = debug"
        ),
    )
    parser.add_argument(
        "--memory-constrained",
        action="store_true",
        help="Run pipeline without writing outputs",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=1.3,
        help="Robust Z-score threshold for outlier detection",
    )

    return parser.parse_args()

def main() -> None:
    # Parse args
    args = parse_args()
    # Define data path
    data_path = DATA_ROOT / ANOMALY_DATA
    # Instantiate lf as a lazyframe
    lf: pl.LazyFrame | None = None
    # Read in data
    with Open(data_path, verbose=args.verbosity) as file:
        lf = file.read(low_mem=args.memory_constrained)
    assert lf is not None  # type narrowing for Pylance
    od = OutlierDetector()
    # Filter for Metric.AC_POWER > 0 before calling 
    # get_robust_z_outliers
    lf_filtered = lf.filter(pl.col(Metric.AC_POWER) > 0)
    # Adjust z_thresh to 1.3 for sensitive outlier detection to ensure 
    # some anomalies are present
    _, anom = od.get_robust_z_outliers(
        lf_filtered, 
        Metric.AC_POWER, 
        Metric.AC_POWER,
        z_thresh=args.z_threshold
    )
    # Join the original LazyFrame 'lf' with the anomaly flags 'anom'
    # This time, 'anom' should only contain keys, robust_z, and
    #  is_outlier, avoiding duplication.
    final_df_with_anom = lf.join(
        anom,
        on=KEYS,
        how="left" # Use a left join to keep all original rows
    )

    # Creating a DC > AC flag to make sure DC and AC measures are 
    # matching
    final_df_with_anom = final_df_with_anom.with_columns(
        (pl.col(Metric.DC_POWER) > pl.col(Metric.AC_POWER))
        .alias('dc_gt_ac_power_flag')
    )

    # Adding the PR filter to detect the actual annomalies
    # The Performance Ratio (PR) is a crucial metric used in solar 
    # photovoltaic (PV) systems to evaluate their overall quality and 
    # efficiency, taking into account all losses from the panel to the 
    # point of measurement. Unlike simply looking at AC power output, 
    # PR normalizes the output by the available solar irradiance 
    # (Plane of Array - POA), providing a more accurate picture of how 
    # well a system is performing relative to its potential.
    P_rated = ANOMALY_P_RATED  # Nominal power of a 35kW system in Watts
    G_ref = 1000   # Reference irradiance in W/m^2

    final_df_with_anom = final_df_with_anom.with_columns(
        pl.when(pl.col(Metric.POA_MEDIAN) > MIN_POA)
        .then(
            (pl.col(Metric.AC_POWER) * G_ref) 
            / (pl.col(Metric.POA_MEDIAN) * P_rated)
        )
        .otherwise(pl.lit(None))
        .alias('performance_ratio')
    )

    # Apply outlier detection to 'performance_ratio'
    _, pr_anom = od.get_robust_z_outliers(
        final_df_with_anom,
        test_col='performance_ratio',
        target='performance_ratio',
        z_thresh=1.5
    )

    # Combining the DataFrame with all the columns
    pr_outlier_status = pr_anom.select(
        pl.col(Column.DEVICE),
        pl.col(Column.TIMESTAMP),
        pl.col('is_outlier').alias('pr_is_outlier')
    )

    final_df_with_anom = final_df_with_anom.join(
        pr_outlier_status,
        on=KEYS,
        how='left'
    )

    # Making a overall flag to track all the detection meathods created 
    # above.
    final_df_with_anom = final_df_with_anom.with_columns(
        (
            pl.col("is_outlier").fill_null(False) &
            pl.col("dc_gt_ac_power_flag").fill_null(False) &
            pl.col("pr_is_outlier").fill_null(False)
        ).alias("overall_anomaly")
    ).collect()

    # Re-using the same sample device name
    sample_device_name = ANOMALY_SAMPLE_DEVICE 

    # Filter the main DataFrame for the specific device
    device_data_for_plot = final_df_with_anom.filter(
        pl.col(Column.DEVICE) == sample_device_name
    )

    # Filter for overall anomalies for the specific device
    overall_anomalies_for_plot = device_data_for_plot.filter(
        pl.col("overall_anomaly") == True
    ).select([Column.TIMESTAMP, Metric.AC_POWER])

    # Convert 'event_local_time' to datetime for proper plotting
    device_data_for_plot = device_data_for_plot.with_columns(
        pl.col(Column.TIMESTAMP).cast(pl.Datetime)
    )
    overall_anomalies_for_plot = (
        overall_anomalies_for_plot
        .with_columns(
            pl.col(Column.TIMESTAMP).cast(pl.Datetime)
        )
    )
    print("\nOverall Anomalies for Plotting (Head):")
    if running_in_ipython_kernel():
        display(overall_anomalies_for_plot.head())
    else:
        print(overall_anomalies_for_plot.head())
    print(
        "Overall Anomalies for Plotting (Shape): "
        f"{overall_anomalies_for_plot.shape}\n"
    )

    # Create the plot
    plt.figure(figsize=(18, 8))
    sns.lineplot(
        data=device_data_for_plot,
        x=Column.TIMESTAMP,
        y=Metric.AC_POWER,
        label="AC Power",
        color='blue',
        linewidth=0.8
    )

    # Plot overall anomalies
    sns.scatterplot(
        data=overall_anomalies_for_plot,
        x=Column.TIMESTAMP,
        y=Metric.AC_POWER,
        color='red',
        marker='X',
        s=150,
        label="Overall Anomaly",
        zorder=5
    )

    plt.title(
        f"AC Power Over Time for Device: {sample_device_name} "
        "with Overall Anomalies"
    )
    plt.xlabel("Timestamp")
    plt.ylabel("AC Power")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()