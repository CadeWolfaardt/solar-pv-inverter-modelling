# std lib
from typing import (
    Tuple,
    Optional,
    Union,
)
# thirdpartylib
import polars as pl
# projectlib
from pv_inverter_modeling.data.schemas import Column, Metric
from pv_inverter_modeling.config.constants import MAD_CONSISTENCY_CONSTANT
from pv_inverter_modeling.utils.typing import InferedFloat

class OutlierDetector(object):

    """
    Class for detecting outliers and masking them as None in the data 
    passed.
    
    Responsibilities:
    - Broad outlier masking using robust-Z across many float columns 
      (daylight-only guard).
    - Deriving temperature-dependent apparent power rating S_R(T) from 
      data (binning + quantiles).
    - Converting S_R(T) to a piecewise-linear curve for arbitrary 
      temperatures.
    - Computing a temperature-aware active power rating P_R(T) that 
      respects VAR and curtailment.
    - Computing per-regime efficiencies and masking outliers via 
      robust-Z.
    
    General Notes:
    - All operations are expressed as Polars LazyFrame transformations 
      (no collection unless needed).
    - Masking strategy: flagged values are joined back and replaced with
      nulls to preserve schema.
    """
    
    def mask_outliers(
            self, 
            lf: pl.LazyFrame, 
            keys: Tuple[Column, ...] = (Column.DEVICE, Column.TIMESTAMP)
        ) -> pl.LazyFrame:
        """
        Apply both broad outlier masking and efficiency-based outlier 
        masking.
        
        Parameters
        ----------
        lf : pl.LazyFrame
            Input lazy frame containing required columns (see other 
            methods for specifics).
        keys : Tuple[Column, ...],
            Join keys used to realign masks back to the original rows.
        
        Returns
        -------
        pl.LazyFrame
            LazyFrame with outliers masked as nulls, sorted by `keys`.
        """
        lf = lf.clone()
        # Mask broad outliers
        lf = self.get_broad_outliers(lf)
        # Get efficiency outliers
        lf, _ = self.get_eff_outliers(lf)
        # return masked lf
        return lf.sort(keys)
    
    def __validate_cols_found(
            self, required: Tuple[Union[Column, Metric], ...], 
            lf: pl.LazyFrame, bool_return: bool = False) -> bool:
        """
        Ensure that all `required` column names are present in the 
        LazyFrame.

        Parameters
        ----------
        required : Tuple[Column, ...]
            Columns that must exist.
        lf : pl.LazyFrame
            LazyFrame to check.
        bool_return : bool, default False
            If True, return False on missing columns instead of raising.

        Returns
        -------
        bool
            True if columns are present; otherwise False when 
            `bool_return=True`.
        
        Raises
        ------
        pl.exceptions.ColumnNotFoundError
            If required columns are missing and `bool_return=False`.
        """
        req = set(required)
        cols = set(lf.collect_schema().names())
        if not req.issubset(cols):
            if not bool_return:
                msg = f"{req} not in LazyFrame"
                raise pl.exceptions.ColumnNotFoundError(msg)
            return False
        return True

    def get_broad_outliers(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Mask broad outliers using a robust-Z approach on float32 columns
        (excluding specified metrics).
        
        Strategy
        --------
        - Consider only finite values and daylight-producing rows 
          (Metric.AC_POWER > 0).
        - For each candidate float column:
            median -> absolute deviations -> MAD -> robust sigma -> 
            robust-Z.
        - Flag |robust-Z| > 3.5 as outliers (Iglewicz & Hoaglin, 1993).
        - Join masks back and set outliers to null.
        
        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing at least the AC power metric and 
            float32 features.
        
        Returns
        -------
        pl.LazyFrame
            LazyFrame with outliers masked as nulls for considered 
            columns.
        """
        keys = (Column.DEVICE, Column.TIMESTAMP)
        self.__validate_cols_found(keys, lf)
        cols_to_ignore = {
            # Ignore grid side fluctuations as they are more volatile
            Metric.POWER_FACTOR,
            Metric.FREQUENCY,
            Metric.AC_LINE_AB,
            Metric.AC_LINE_BC,
            Metric.AC_LINE_CA,
            Metric.AC_PHASE_A,
            Metric.AC_PHASE_B,
            Metric.AC_PHASE_C,
            Metric.VAR,
            # Ignore curtailment
            Metric.ACTIVE_LIMIT,
            Metric.SVA_LIMIT,
            Metric.VAR_LIMIT,
        }
        fit: set[Metric] = {
            Metric(col)
            for col, dtype in lf.collect_schema().items()
            if dtype.is_float()
        } - cols_to_ignore
        outliers = (
            lf
            .filter(
                # Use only real values
                pl.all_horizontal(*(pl.col(c).is_finite() for c in fit))
                # As values with no daylight dominate the metrics are 
                # sskewed so we filter for around daylight
                # i.e., when AC power is being generated 
                & (pl.col(Metric.AC_POWER) > 0)
            )
            # Obtain per col median
            .with_columns(
                *(pl.col(col).median().alias(f"median_{col}")
                for col in fit)
            )
            # Obtain deviation of per col values from per col median
            .with_columns(
                *(
                    (pl.col(col) - pl.col(f"median_{col}"))
                    .abs().alias(f"abs_dev_{col}")
                    for col in fit
                )
            )
            # Obtain median of deviations
            .with_columns(
                pl.col(f"abs_dev_{col}").median().alias(f"mad_{col}")
                for col in fit
            )
            # Compute robust sigma
            .with_columns(
                *(
                    (MAD_CONSISTENCY_CONSTANT * pl.col(f"mad_{col}"))
                    .alias(f"robust_sigma_{col}") for col in fit
                )
            )
            # Safe robust z: (x - median) / (1.4826 * MAD)
            .with_columns(
                *(
                    pl.when(pl.col(f"robust_sigma_{col}") > 0)
                    .then(
                        (pl.col(col) - pl.col(f"median_{col}")) 
                        / pl.col(f"robust_sigma_{col}")
                    )
                    .otherwise(pl.lit(None))
                    .alias(f"robust_z_{col}")
                    for col in fit
                )
            )
            # Binary flag
            .with_columns(
                *(
                    pl.col(f"robust_z_{col}")
                    .abs().gt(3.5).alias(f"is_outlier_{col}")
                    for col in fit
                )
            )
            # Aggregate masks, as Kleene logic is used by any_horizontal 
            # (if the column contains any null values and no True 
            # values, the output is null) be replace null with False
            .with_columns(
                pl.any_horizontal(
                    *[pl.col(f"is_outlier_{col}") for col in fit]
                ).replace(None, False).alias("is_broad_outlier")
            )
        )
        # Mask outliers as null in lazyframe
        mask = (
            outliers
            .select([
                *keys, 
                *(col for col in fit), 
                *(f"is_outlier_{col}" for col in fit)
            ])
            .with_columns(
                *(pl.when(pl.col(f"is_outlier_{col}"))
                .then(pl.lit(None, dtype=pl.Float32))
                .otherwise(pl.col(col))
                .alias(col)
                for col in fit)
            )
            .select([*keys, *(col for col in fit)])
            # Create col to highlight observations in this lazyframe
            # when joined to original lazyframelater
            .with_columns(pl.lit(True).alias("__matched__"))
        )
        suffix = "_mask"
        # Join LazyFrames and mask outliers as null
        masked = (
            lf
            .join(mask, on=keys, how="left", suffix=suffix)
            # When there is a null from mask overwrite lf 
            # ignoring nulls introduced from join
            .with_columns([
                pl.when(pl.col("__matched__"))
                .then(pl.col(f"{col}{suffix}"))
                .otherwise(pl.col(col))
                .alias(col)
                for col in fit
            ])
            .drop([*(f"{col}{suffix}" for col in fit), "__matched__"])
        )

        return masked

    def get_S_of_T(
            self, lf: pl.LazyFrame, 
            temp_col: Metric = Metric.ADMISSION_TEMP, 
            grouping: Tuple[str, ...] = ("bin_left",),
            min_temp: float = 0, max_temp: InferedFloat = "infer", 
            temp_quant: float = 0.99, temp_tolerance: float = 5, 
            derate_thresh: float = 100, pf_epsilon: float = 0.02, 
            poa_thresh: float = 900, eta_median: float = 0.98, 
            eta_tolerance: float = 0.01, temp_bin_width: float = 2, 
            count_thresh: int = 20) -> pl.LazyFrame:
        """
        Find the curve of temperature apparent rating curve from the 
        data given.
        
        Purpose
        -------
        Estimate the apparent power rating S_R(T) as a function of 
        temperature by:
        - filtering to near-ideal operating conditions (no curtailment, 
          PF≈1, high POA, high efficiency);
        - binning temperature (width = `temp_bin_width`);
        - taking a high-quantile (0.95) of SVA as the empirical rating 
          per bin;
        - returning a binned curve (`bin_left`, `empirical_rated_SVA`) 
          suitable for interpolation.
        
        Parameters
        ----------
        lf : pl.LazyFrame
            Input data.
        temp_col : Metric, Temperature column to bin.
        grouping : Tuple[str, ...], default ("bin_left",)
            Group-by key(s), typically the left edge of the temp bin.
        min_temp : float, default 0
            Minimum temperature allowed.
        max_temp : InferedFloat, default "infer"
            Maximum temperature; if "infer", use quantile(`temp_quant`) 
            + `temp_tolerance`.
        temp_quant : float, default 0.99
            Quantile used to infer `max_temp` when `max_temp="infer"`.
        temp_tolerance : float, default 5
            Additive margin above the inferred quantile for `max_temp`.
        derate_thresh : float, default 100
            Setpoint threshold (percent) indicating no curtailment 
            (== 100).
        pf_epsilon : float, default 0.02
            |PF - 1| tolerance for "PF≈1".
        poa_thresh : float, default 900
            POA irradiance threshold for "high resource" filtering.
        eta_median : float, default 0.98
            Nominal efficiency center for filtering.
        eta_tolerance : float, default 0.01
            Acceptable deviation around `eta_median` for filtering.
        temp_bin_width : float, default 2
            Degrees per bin.
        count_thresh : int, default 20
            Minimum samples per bin to retain.
        
        Returns
        -------
        pl.LazyFrame
            Binned table with columns: `bin_left`, `n`, 
            `empirical_rated_SVA`.
        """
        # Ensure required cols are present in dataframe
        required = (
            temp_col,
            Metric.SVA_LIMIT,
            Metric.VAR_LIMIT,
            Metric.ACTIVE_LIMIT,
            Metric.POWER_FACTOR,
            Metric.POA_MEDIAN,
            Metric.AC_POWER,
            Metric.DC_POWER,
        )
        self.__validate_cols_found(required, lf)
        if isinstance(max_temp, str):
            max_temp = (
                lf.select(temp_col).quantile(temp_quant).collect().item() 
                + temp_tolerance
            )
        
        temp_filter = pl.col(temp_col).is_between(min_temp, max_temp)
        derate_filter = (
            (pl.col(Metric.SVA_LIMIT) == derate_thresh)
            & (pl.col(Metric.VAR_LIMIT) == derate_thresh)
            & (pl.col(Metric.ACTIVE_LIMIT) == derate_thresh)
        )
        pf_filter = ((pl.col(Metric.POWER_FACTOR) - 1).abs() < pf_epsilon)
        poa_filter = (pl.col(Metric.POA_MEDIAN) > poa_thresh)
        # Filter for nominal efficiencies given other filter conditions
        eta_filter =  (
            (pl.col(Metric.AC_POWER) / pl.col(Metric.DC_POWER)) 
            > (eta_median - eta_tolerance)
        )
        # Filter null values
        lf = (
            lf
            .filter(
                temp_filter 
                & derate_filter 
                & pf_filter 
                & poa_filter 
                & eta_filter
            )
            .drop_nulls()
            .with_columns(
                ((pl.col(temp_col) / temp_bin_width).floor() * temp_bin_width)
                .alias("bin_left")
            )
            .group_by(grouping)
            .agg([
                pl.len().alias("n"),
                pl.col(Metric.SVA).quantile(0.95).alias("empirical_rated_SVA")
            ])
            .filter(pl.col("n") >= count_thresh)
            .sort(grouping)
        )

        return lf
    
    def piece_wise_linear_S_of_T(
            self, 
            lf: pl.LazyFrame, 
            rated_sva_lf: pl.LazyFrame, 
            temp_col: Metric = Metric.ADMISSION_TEMP
        ) -> pl.LazyFrame:
        """
        Interpolate a piecewise-linear S_R(T) curve from binned ratings.
        
        Parameters
        ----------
        lf : pl.LazyFrame
            Original data providing temperature samples for 
            interpolation.
        rated_sva_lf : pl.LazyFrame
            Output of `get_S_of_T` with `bin_left` and 
            `empirical_rated_SVA`.
        temp_col : Metric,
            Temperature column to interpolate over.
        
        Returns
        -------
        pl.LazyFrame
            LazyFrame with an added `SVA_R_T` column representing 
            S_R(T) at each row's temperature.
        """
        curve = (
            rated_sva_lf
            .select(["bin_left", "empirical_rated_SVA"])
            .unique(["bin_left"])
            .sort("bin_left")
        )
        # Backward neighbor (T0,S0)
        lo = (
            lf
            .sort(temp_col)  # only by temperature since there are no keys
            .join_asof(
                curve,
                left_on=temp_col, right_on="bin_left",
                strategy="backward",
                suffix="_lo",
            )
        )
        # Forward neighbor (T1,S1)
        both = (
            lo.join_asof(
                curve,
                left_on=temp_col, right_on="bin_left",
                strategy="forward",
                suffix="_hi",
            )
        )
        # Linear blend with safe fallbacks
        SVA_R_T = (
            both
            .with_columns([
                pl.col(temp_col).alias("T"),
                pl.col("bin_left").alias("T0"),
                pl.col("empirical_rated_SVA").alias("S0"),
                pl.col("bin_left_hi").alias("T1"),
                pl.col("empirical_rated_SVA_hi").alias("S1"),
            ])
            .with_columns(
                pl.when(
                    pl.all_horizontal(
                        pl.col("T0").is_not_null(),
                        pl.col("T1").is_not_null(),
                        pl.col("T1") > pl.col("T0")
                    )
                )
                .then(
                    (pl.col("T") - pl.col("T0")) 
                    / (pl.col("T1") - pl.col("T0"))
                )
                .otherwise(None)
                .alias("w")
            )
            .with_columns(
                pl.when(pl.col("w").is_not_null())
                .then(
                    (1 - pl.col("w")) * pl.col("S0") 
                    + pl.col("w") * pl.col("S1")
                )
                 # Edge: only one neighbor exists
                .otherwise(pl.coalesce([pl.col("S0"), pl.col("S1")]))
                .cast(pl.Float64)
                .alias("SVA_R_T")
            )
            .drop([
                "T", 
                "T0", 
                "T1", 
                "S0", 
                "S1",
                "w", 
                "bin_left", 
                "empirical_rated_SVA",
                "bin_left_hi",
                "empirical_rated_SVA_hi"
            ])
        )

        return SVA_R_T
  
    def get_P_of_T(self, lf: pl.LazyFrame, 
                   temp_col: Metric = Metric.ADMISSION_TEMP) -> pl.LazyFrame:
        """
        Compute the active power rating curve P_R(T) by combining:
        - S_R(T) (apparent rating) from `get_S_of_T` + interpolation,
        - Reactive power Q from data (Metric.VAR),
        - Curtailment alpha from setpoints (min of relevant limit 
          setpoints).
        
        Formulas
        --------
        alpha = min(Metric.ACTIVE_LIMIT, Metric.SVA_LIMIT) / 100
        P_R_1 = sqrt(S_R(T)^2 - Q^2)
        P_R_2 = alpha * S_R(T)
        P_R(T) = min(P_R_1, P_R_2)
        
        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame with necessary columns.
        temp_col : Metric,
            Temperature column for S_R(T).
        
        Returns
        -------
        pl.LazyFrame
            LazyFrame with `P_R_T` appended.
        """
        rated_apparent_power = self.get_S_of_T(lf.clone(), temp_col)
        SVA_R_T = self.piece_wise_linear_S_of_T(lf, rated_apparent_power)
        return (
            SVA_R_T
            # Get alpha = min(Metric.ACTIVE_LIMIT, Metric.SVA_LIMIT) / 100
            .with_columns(
                (
                    (pl.min_horizontal(
                        pl.col(Metric.ACTIVE_LIMIT),
                        pl.col(Metric.SVA_LIMIT)
                    ).clip(0,100)) / 100.0
                ).alias("alpha")
            )
            # Get P_R_1 = sqrt(SVA_R_T**2 - Q**2)
            .with_columns(
                pl.max_horizontal(
                    pl.lit(0), 
                    (pl.col('SVA_R_T').pow(2) - pl.col(Metric.VAR).pow(2))
                ).sqrt().alias("P_R_1")
            )
            # Get P_R_2 = alpha * SVA_R_T
            .with_columns(
                (pl.col("alpha") * pl.col("SVA_R_T")).alias("P_R_2")
            )
            # Get P_R_T = min(P_R_1, P_R_2)
            .with_columns(
                pl.min_horizontal(pl.col("P_R_1"), pl.col("P_R_2"))
                .alias("P_R_T")
            )
        )

    def get_eff_outliers(
            self, 
            lf: pl.LazyFrame, 
            temp_col: Metric = Metric.ADMISSION_TEMP, 
            tau_rel: float = 0.015, 
            tau_abs: float = 10, 
            keys: Tuple[Column,...] = (Column.DEVICE, Column.TIMESTAMP)
        ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Compute regime-aware efficiency and mask outliers via robust-Z 
        per regime.
        
        Regimes
        -------
        - constrained_regime: 
            |P_R(T) - P_AC| < max(tau_abs, P_R(T) * tau_rel)
            (AC constrained near rating; PF/Q affects P_AC)
        - normal_regime: not constrained AND POA > 20
            (standard operation; efficiency = P_AC / P_DC)
        - other_regime: remaining cases
        
        Workflow
        --------
        1) Derive P_R(T).
        2) Flag regimes and compute `efficiency`:
           - constrained: P_R(T) / P_DC
           - normal:      P_AC / P_DC
        3) Run robust-Z masking of `efficiency` within each regime.
        
        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame with required metrics.
        temp_col : Metric,
            Temperature column for rating curves.
        tau_rel : float, default 0.015
            Relative tolerance (%) for constrained regime band.
        tau_abs : float, default 10
            Absolute tolerance (same units as power) for constrained 
            regime band.
        keys : Tuple[Column, ...],
            Join keys for masking merges.
        
        Returns
        -------
        Tuple[pl.LazyFrame, pl.LazyFrame]
            (masked LazyFrame, efficiency LazyFrame with regime flags)
        """
        # Get P_R(T)
        P_R_T = self.get_P_of_T(lf, temp_col)
        orig_cols = sorted(set(lf.collect_schema().names()).difference(keys))

        efficiency = (
            P_R_T
            # Get constrained regime mask: 
            #     |P_R_(T) - P_AC| < max(tau_abs, P_R_(T) * tau_rel)
            # this defines tau_rel as a percentage bound within which 
            # P_AC is assumed to be close enough to P_R_(T) that P_AC is 
            # impacted by Q generation
            .with_columns(
                pl.when(
                    (pl.col("P_R_T") - pl.col(Metric.AC_POWER)).abs() 
                    < pl.max_horizontal(
                        pl.lit(tau_abs), pl.col("P_R_T") * tau_rel
                    )
                )
                .then(True)
                .otherwise(False)
                .alias("constrained_regime")
            )
            # Normal regime
            .with_columns(
                pl.when(
                    (~pl.col("constrained_regime"))
                    # Daylight threshold for defining presence of > 0 
                    # output AC power
                    & (pl.col(Metric.POA_MEDIAN) > 20.0)
                )
                .then(True)
                .otherwise(False)
                .alias("normal_regime")
            )
            # Other regime
            .with_columns(
                pl.when(
                    (~pl.col("constrained_regime"))
                    & (~pl.col("normal_regime"))
                )
                .then(True)
                .otherwise(False)
                .alias("other_regime")
            )
            # Get efficiency for the different regimes
            .with_columns(
                pl.when(
                    pl.col("constrained_regime") 
                    & (pl.col(Metric.DC_POWER) > 0)
                )
                .then(
                    (pl.col("P_R_T") / pl.col(Metric.DC_POWER))
                )
                .when(
                    (~pl.col("constrained_regime")) 
                    & (pl.col(Metric.DC_POWER) > 0)
                )
                .then(
                    pl.col(Metric.AC_POWER) / pl.col(Metric.DC_POWER)
                )
                .otherwise(None)
                .alias("efficiency")
            )
            # Keep needed columns
            .select([
                *keys, 
                *orig_cols, 
                'efficiency', 
                'constrained_regime', 
                'normal_regime', 
                'other_regime'
            ])
        )
        
        # Define filters for different regimes
        outliers_1, _ = self.get_robust_z_outliers(
            efficiency, 
            "efficiency", 
            Metric.AC_POWER, 
            mask=("constrained_regime", True)
        )
        outliers_2, _ = self.get_robust_z_outliers(
            outliers_1, 
            "efficiency", 
            Metric.AC_POWER, 
            mask=("normal_regime", True)
            )
        outliers_3, _ = self.get_robust_z_outliers(
            outliers_2, 
            "efficiency", 
            Metric.AC_POWER, 
            mask=("other_regime", True)
        )

        return outliers_3, efficiency

    def get_robust_z_outliers(
            self, 
            lf: pl.LazyFrame, 
            test_col: str, 
            target: str, 
            keys: Tuple[Column, ...] = (Column.DEVICE, Column.TIMESTAMP), 
            mask: Optional[Tuple[str, bool]] = None, 
            z_thresh: InferedFloat = 3.5
        ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Mask outliers in `target` using the robust-Z score computed on 
        `test_col`.
        
        Steps
        -----
        - Optionally filter to rows matching `mask` (col == value).
        - Compute median, MAD, robust sigma, robust-Z on `test_col`.
        - If `z_thresh == "infer"`, use the 99.9th 
          percentile(|robust-Z|) as threshold.
        - Flag |robust-Z| > z_thresh as outliers and set `target` to 
          null for those rows.
        - Join mask back to original `lf` on `keys` and replace values.
        
        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame.
        test_col : str
            Column used to compute robust-Z.
        target : str
            Column to mask when an outlier is detected.
        keys : Tuple[Column, ...],
            Join keys used when merging masks back.
        mask : Optional[Tuple[str, bool]], default None
            A (column, value) pair to filter rows before computing 
            robust-Z.
        z_thresh : InferedFloat, default 3.5
            Threshold for |robust-Z|; if "infer", the threshold is 
            data-driven at 99.9%.
        
        Returns
        -------
        pl.LazyFrame
            LazyFrame with `target` masked to null where robust-Z 
            exceeds the threshold.
        
        References
        ----------
        Iglewicz, B., & Hoaglin, D. (1993). *How to Detect and Handle 
        Outliers*.
        """
        outliers = lf.clone()
        if mask:
            outliers = outliers.filter(pl.col(mask[0]) == mask[1])

        outliers = (
            outliers
            .select([*keys, test_col])
            # Filter for real values
            .filter(pl.col(test_col).is_finite())
            # Get median of test_col
            .with_columns(pl.col(test_col).median().alias(f"median"))
            # Get abs deviation of test_col from median
            .with_columns(
                (pl.col(test_col) - pl.col(f"median")).abs().alias("abs_dev")
            )
            # Get median of absolute deviations (MAD)
            .with_columns(pl.col("abs_dev").median().alias("MAD"))
            # Get robust sigma, 1.4826 scales MAD to match standard 
            # deviation of a normal distribution 
            .with_columns(
                (MAD_CONSISTENCY_CONSTANT * pl.col("MAD"))
                .alias("robust_sigma")
            )
            # Compute safe robust-z
            .with_columns(
                pl.when(pl.col(f"robust_sigma") > 0)
                .then(
                    (pl.col(test_col) - pl.col(f"median")) 
                    / pl.col(f"robust_sigma")
                )
                .otherwise(pl.lit(None))
                .alias(f"robust_z")
            )
            
        )
        # If z_thresh is "infer" then obtain 99.9th percentile value of 
        # robust_z and use as threshold. The 3.5 preset comes from 
        # Iglewicz & Hoaglin (1993), How to Detect and Handle Outliers.
        if z_thresh == "infer":
            z_thresh = (
                outliers
                .select(pl.col("robust_z").quantile(.999))
                .collect()
                .item()
            )

        # Get target dtype
        dtype = lf.collect_schema().get(target)
        outliers_flg = (
            outliers
            # Compute outlier boolean flag
            .with_columns(
                pl.col(f"robust_z").abs().gt(z_thresh).alias("is_outlier")
            )
            # Mask outliers with null
            .select([*keys, "robust_z", "is_outlier"])
        )
        suffix = "_mask"
        masked = (
            lf
            .join(
                outliers_flg.select([*keys, "is_outlier"]), 
                on=keys, 
                how="left", 
                suffix=suffix
            )
            .with_columns(
                pl.when(
                    pl.col(f"is_outlier")
                )
                .then(pl.lit(None, dtype=dtype))
                .otherwise(pl.col(target))
                .alias(target)
            )
            .drop([f"is_outlier"])
        )

        return masked, outliers_flg
