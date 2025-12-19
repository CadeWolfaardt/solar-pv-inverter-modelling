# stdlib
from typing import Tuple, List, Optional, Unpack, Dict, Union, Iterable, cast
from datetime import datetime
# thirdpartylib
import numpy as np
import polars as pl
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error
)
# projectlib
from pv_inverter_modeling.utils.typing import (
    LossMode, 
    LossKwargs, 
    InjectKwargs,
    InterpMethod,
    InterpMethods,
    Field,
    DataMode,
    Address
)
from pv_inverter_modeling.data.schemas import Column, Metric, KEYS
from pv_inverter_modeling.config.env import (
    DAYLIGHT_MAP,
    FULL_INTERPOLATION_MAP,
    BASE_INTERPOLATION_MAP,
    DATA_ROOT
)
from pv_inverter_modeling.utils.util import validate_address

INTERPOLATION_METHODS: InterpMethods = (
    "linear", 
    "time", 
    "index", 
    "values", 
    "nearest",
    "zero", 
    "slinear", 
    "quadratic", 
    "cubic", 
    "polynomial", 
    "spline",
    "pchip", 
    "akima", 
    "cubicspline", 
    "from_derivatives",
)
# Hyperparameters for interpolation
VALID_FRAC = 0.8
MAX_DAYLIGHT_GAP = 6
MAX_RAMP_GAP = 3

class NullInjector(object):
    """
    Class for injecting null or missing values.
    
    Responsibilities:
    - Determining patterns of missingness in data and replicating them
      in valid, intact series with no missing values. 
    """
    def rle(
            self, 
            mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run-length encode a 0/1 mask.
        Returns (run_starts, run_lengths, run_values) where run_values 
        are 0 or 1
        """
        if mask.size == 0:
            z = np.array([], dtype=np.int_)
            return z, z, z
        # Boundary whenever value changes; prepend flipped first value 
        # to force boundary at 0
        change_idx = np.flatnonzero(np.diff(mask, prepend=mask[0] ^ 1))
        lengths = np.diff(np.r_[change_idx, mask.size]).astype(np.int_)
        values = mask[change_idx].astype(np.int_)

        return change_idx.astype(np.int_), lengths, values

    def __get_run_lengths(
            self, 
            mask: np.ndarray
        ) -> Tuple[List[int], List[int]]:
        """Return run lengths of null (1) and observed (0) records"""
        _, lengths, values = self.rle(mask)
        miss_lengths = lengths[values == 1].tolist()
        obs_lengths = lengths[values == 0].tolist()

        return miss_lengths, obs_lengths
    
    def __get_hourly_profile(self, series: pd.Series) -> List[float]:
        """Obtain means of data aggregated by hour."""
        if not isinstance(series.index, pd.DatetimeIndex):
            series = series.copy()
            series.index = pd.to_datetime(series.index)

        by_hour = (
            series
            .isna()
            .groupby( # pyright: ignore[reportUnknownMemberType]
                series.index.hour
            )
            .mean()
        )
        return by_hour.reindex(range(24), fill_value=0.0).values.tolist()
    
    def __get_periodicity_fft(self, mask: np.ndarray) -> Optional[int]:
        """
        Return approximate period (in samples) of missingness via FFT
        of zero-mean indicator.
        """
        m = mask - mask.mean()
        spec = np.abs(np.fft.rfft(m))
        freqs = np.fft.rfftfreq(len(m), d=1) # sampling space (d) of 1
        # Ignore zero frequency
        if len(spec) < 3: return None
        k = np.argmax(spec[1:]) + 1
        f = freqs[k]
        if f == 0: return None

        return int(round(1/f))
    
    def get_data_stats(self, series: pd.Series) -> LossKwargs:
        """
        Get data statistics for replicating missingness patterns found
        in data.
        """
        s = series.copy().astype(float)
        mask = s.isna().to_numpy().astype(int)
        # Fraction missing
        f_miss = mask.mean()
        # Lengths (missing & obs) and there means
        miss_lengths, _ = self.__get_run_lengths(mask)
        mean_ml = np.mean(miss_lengths) if miss_lengths else 0.0
        # Markov params
        p_recover = 1.0 / mean_ml if mean_ml > 0 else 0
        # Ensure no division by zero error using max func in denominator
        p_miss = (f_miss * p_recover) / max(1e-9, (1 - f_miss))
        # Time of Day (ToD) params
        p_by_hour = self.__get_hourly_profile(s)
        # Periodic params
        k_period = self.__get_periodicity_fft(mask)
        
        out: LossKwargs = {
            "p": f_miss,
            "p_miss": p_miss,
            "p_recover": float(p_recover),
            "p_by_hour": p_by_hour,
        }
        if k_period:
            out['k'] = k_period
 
        return out

    def __apply_mask(self, series: pd.Series, 
                     mask: np.ndarray, fill: float = np.nan) -> pd.Series:
        """Apply a mask of specified fill to a series."""
        out = series.copy()
        out.iloc[mask] = fill
        return out
    
    def __segment_bounds(self, n: int, mean_len: float) -> Tuple[int, int]:
        """Return start and end indices for runs of data."""
        start = np.random.randint(0, max(1, n-1))
        length = max(1, np.random.poisson(mean_len))
        end = min(n, start+length)
        return start, end

    def __random_mask(self, n: int, p: float) -> np.ndarray:
        """MCAR: independently drop each point with prob p"""
        return np.where(np.random.rand(n) < p)[0]
    
    def __blocks_mask(self, n: int, 
                      n_blocks: int, mean_len: float) -> np.ndarray:
        """
        Contiguous outages: Poisson/geom-like block lengths around 
        mean_len.
        """
        idx: List[int] = []
        for _ in range(n_blocks):
            start, end = self.__segment_bounds(n, mean_len)
            idx.extend(range(start, end))
        
        return np.unique(idx)

    def __markov_bursty_mask(
            self, 
            n: int, 
            p_miss: float, 
            p_recover: float, 
            start_state: str
        ) -> np.ndarray:
        """
        Two-state chain:
        state=0 observe, state=1 assign null
        p_null:     P(0->1)  (start assigning null)
        p_recover:  P(1->0)  (end null assignment)
        Avg null run ~ 1/p_recover; 
        overall null frac ~ p_miss/(p_miss+p_recover)
        """
        state = 0 if start_state == "obs" else 1
        mask_idx: List[int] = []
        for i in range(n):
            if state == 0 and np.random.rand() < p_miss:
                state = 1
            elif state == 1 and np.random.rand() < p_recover:
                state = 0
            if state == 1:
                mask_idx.append(i)
        return np.array(mask_idx)

    def __tod_profile_mask(self, ts: pd.DatetimeIndex, 
                           p_by_hour: List[float]) -> np.ndarray:
        """
        Time-of-day bias: p_by_hour is length-24 list/array of drop 
        probs per hour.
        """
        probs = np.array([p_by_hour[h] for h in ts.hour])
        return np.where(np.random.rand(len(ts)) < probs)[0]
    
    def __periodic_mask(self, n: int, k: int, jitter: int) -> np.ndarray:
        """Drop every k-th sample (with optional random jitter)"""
        base = np.arange(0, n, k)
        if jitter > 0:
            base = np.clip(
                base + np.random.randint(-jitter, jitter+1, size=len(base)), 
                0, 
                n-1
            )
        return np.unique(base)

    def __stuck_segments(self, series: pd.Series, 
                         n_segments: int, mean_len: float) -> pd.Series:
        """Replicate stuck sensor or same value repition error."""
        s = series.copy()
        n = len(series)
        for _ in range(n_segments):
            start, end = self.__segment_bounds(n, mean_len)
            if start > 0:
                const_val = float(series.iloc[max(0, start-1)])  
            else:
                const_val = float(series.iloc[0])
            s.iloc[start:end] = const_val
        
        return s

    def inject_loss(
            self, 
            series: pd.Series, 
            mode: LossMode = 'markov',
            **kwargs: Unpack[LossKwargs]
        ) -> pd.Series:
        """
        Returns a 'corrupted' series with realistic losses.
        modes: 'random', 'blocks', 'markov', 'tod', 'periodic', 'stuck'

        `series` should be an intact series with no missing values.
        """
        series = series.copy()
        n = len(series)

        match mode:
            case "random":
                idx = self.__random_mask(n, kwargs.get('p', 0.02))
                return self.__apply_mask(series, idx)
            
            case "blocks":
                idx = self.__blocks_mask(
                    n,
                    int(kwargs.get('n_blocks', 5)),
                    kwargs.get('mean_len', 12.0)
                )
                return self.__apply_mask(series, idx)
            
            case "markov":
                idx = self.__markov_bursty_mask(
                    n,
                    p_miss=kwargs.get('p_miss', 0.02),
                    p_recover=kwargs.get('p_recover', 0.2),
                    start_state=kwargs.get('start_state', "obs")
                )
                return self.__apply_mask(series, idx)
            case "tod":
                if not isinstance(series.index, pd.DatetimeIndex):
                    try:
                        ts = pd.to_datetime(series.index)
                    except Exception:
                        msg = (
                            "Unknown index type and cannot convert to "
                            "DatetimeIndex"
                        )
                        raise ValueError(msg)
                else:
                    ts = series.index
                p_by_hour = kwargs.get('p_by_hour', [0.0] * 24)
                idx = self.__tod_profile_mask(ts, p_by_hour)
                return self.__apply_mask(series, idx)
            
            case "periodic":
                idx = self.__periodic_mask(
                    n, kwargs.get('k', 6), kwargs.get('jitter', 0)
                )
                return self.__apply_mask(series, idx)
            
            case "stuck":
                return self.__stuck_segments(
                    series, 
                    kwargs.get('n_segments', 3), 
                    kwargs.get('mean_len', 12.0)
                )
            
            case _:
                raise ValueError("Unkown mode")

class InterpolationTester(object):
    """
    Class for testing interpolation methods across methods and devices
    to determine the best interpolation method for each metric across 
    all devices.
    """
    def __init__(self) -> None:
        self.rle = NullInjector().rle
        self.inject_loss = NullInjector().inject_loss
        self.get_data_stats = NullInjector().get_data_stats
    
    def __normalize_key(self, name: Union[Tuple[str, ...], str]) -> str:
        """Return a key normalized as a string."""
        if isinstance(name, tuple) and len(name) == 1:
            return str(name[0])
        return str(name)

    def get_device_sep_dfs(
            self, 
            lf: pl.LazyFrame, 
            cols_to_drop: Optional[Iterable[Field]] = None, 
            pivot_df: bool = False
        ) -> Dict[str, pd.DataFrame]:
        """
        Return inverter seperated dataframes indexed by 
        Column.TIMESTAMP.
        """
        # Drop cols if if not in required
        required = set(KEYS)
        if cols_to_drop:
            cols = lf.collect_schema().names()
            drop = set(cols_to_drop) - required
            cols_to_keep = [c for c in cols if c not in drop]
            lf = lf.select(cols_to_keep)
        # Ensure timestamp is datetime
        lf = lf.with_columns(
            pl.col(Column.TIMESTAMP).cast(pl.Datetime("us"))
        )
        # Collect dataframe
        df = lf.sort(KEYS).collect()
        # Pivot the table
        if pivot_df:
            df = (
                df
                .pivot(
                    Column.METRIC,
                    index=KEYS,
                    values=Column.VALUE,
                    aggregate_function="first"
                )
            )
        # Split into groups by device_name
        try:
            groups = df.partition_by(
                Column.DEVICE, 
                as_dict=True, 
                maintain_order=True
            )  # {name: DataFrame}
        except TypeError:
            parts = df.partition_by(Column.DEVICE, maintain_order=True)
            groups = {g[Column.DEVICE][0]: g for g in parts}
        
        inv_dfs: Dict[str, pd.DataFrame] = {
            self.__normalize_key(name): ( # pyright: ignore[reportArgumentType]
                g.drop(Column.DEVICE)
                .to_pandas()
                .set_index(Column.TIMESTAMP)
                .sort_index()
            )
            for name, g in groups.items()
        }
        
        return inv_dfs

    def __get_longest(self, series: pd.Series) -> pd.Series:
        """
        Return the longest contiguous non-missing slice of the series.
        """
        missing = series.isna().to_numpy(dtype=np.int_)
        starts, lengths, values = self.rle(missing)
        # Observed runs are where values == 0
        obs_mask = (values == 0)
        if not np.any(obs_mask):
            return series.iloc[0:0] # empty
        # Get longest run
        obs_starts = starts[obs_mask]
        obs_lens   = lengths[obs_mask]
        best_i     = int(np.argmax(obs_lens))
        best_start = int(obs_starts[best_i])
        best_len   = int(obs_lens[best_i])
        best_end   = best_start + best_len - 1

        return series.iloc[best_start: best_end + 1] 

    def _eval_interpolators(
            self, 
            series: pd.Series,
            metric: Metric, 
            inject_kwargs: InjectKwargs = {},
            methods: Optional[InterpMethods] = None,
            order: Optional[int] = 3
        ) -> pd.DataFrame:
        """
        Returns a DataFrame with error metrics for each interpolation 
        method. Assumes series index is a regular DatetimeIndex if using 
        method='time'.
        """
        if methods is None:
            methods = INTERPOLATION_METHODS
        elif not set(methods).issubset(INTERPOLATION_METHODS):
            msg = (
                "Unknown interpolation method passed, acceptable methods are: " 
                f"{INTERPOLATION_METHODS}"
            )
            raise ValueError(msg)
        
        truth = series.astype(float).copy()
        # Inject nulls/nan values
        corrupted = self.inject_loss(truth, **inject_kwargs)

        limit_direction = 'both'
        limit_area = "inside"
        results: List[Dict[str, Optional[Union[str, float, bool]]]] = []
        for m in tqdm(methods, desc=f"{metric}", leave=False):
            # Safe guards
            if m == "time" and not isinstance(series.index, pd.DatetimeIndex):
                continue
            if m in ("polynomial", "spline") and order is None:
                continue

            y = corrupted.copy()
            try:
                y = y.interpolate(
                    method=m, 
                    order=order, 
                    limit_direction=limit_direction, 
                    limit_area=limit_area
                )
            except Exception:
                print(f"Interpolation using {m} failed; fallback linear")
                results.append({
                    "method": None,
                    "MAE": None, 
                    "RMSE": None,
                    "Complete": None,
                    "Ran": False
                })
                continue

            # Mask injected missing vals to prevent false test results
            test_mask = corrupted.isna().to_numpy()
            if test_mask.sum() == 0:
                continue
            # Convert to numpy to deal with typing and stability
            y_np = y.astype(float).to_numpy()
            truth_np = truth.astype(float).to_numpy()
            # Keep only injected-NaN positions that ended up filled 
            # (finite) by interpolation
            valid = test_mask & np.isfinite(y_np)
            if not valid.any():
                continue
            yhat  = y_np[valid]
            ytrue = truth_np[valid]
            mae = mean_absolute_error(ytrue, yhat)
            rmse = root_mean_squared_error(ytrue, yhat)

            results.append({
                "method": m,
                "MAE": mae, 
                "RMSE": rmse,
                "Complete": not y.hasnans,
                "Ran": True
            })

        return (
            pd.DataFrame(results)
            .set_index("method")
            .sort_index()
            .sort_values('RMSE')
        )

    def eval_on_lf(
            self, 
            lf: pl.LazyFrame, 
            cols_to_drop: Optional[Iterable[Field]] = None,
            pivot: bool = False
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate inerpolation methods on passed polars lazyframe."""
        inv_dfs = self.get_device_sep_dfs(lf, cols_to_drop, pivot_df=pivot)
        frames: List[pd.DataFrame] = []

        for device, df in tqdm(inv_dfs.items(), desc="Inverters"):
            # Consider only numeric cols
            num_cols: List[Metric] = [
                cast(Metric, c) for c in df.columns 
                if pd.api.types.is_numeric_dtype(df[c])
            ]
            if not num_cols:
                continue

            for metric in tqdm(num_cols, desc=f"{device}", leave=False):
                s = df[metric]
                # Get kwargs for loss injector
                dkwargs: InjectKwargs = cast(
                    InjectKwargs, 
                    self.get_data_stats(s)
                )
                # Get longest contiguous valid run
                clean = self.__get_longest(s)
                if clean.empty:
                    continue
                # Run eval
                results: pd.DataFrame = self._eval_interpolators(
                    clean, metric, inject_kwargs=dkwargs
                )
                results = results.copy()
                results = results.assign(device=device)
                results = results.assign(metric=metric)
                results = (
                    results
                    .reset_index()
                    .rename(columns={"index": "method"})
                )

                frames.append(results)

        final_df = (
            pd.concat(frames, ignore_index=True) if frames 
            else pd.DataFrame(
                columns=["device", "metric", "method", "MAE", "RMSE"]
            )
        )
        local_keys = ("device", "metric")
        # Ranking by RMSE
        if not final_df.empty:
            best_idx = (
                final_df
                .groupby( # pyright: ignore[reportUnknownMemberType]
                    local_keys
                )["RMSE"]
                .idxmin()
            )
            best_df = (
                final_df
                .loc[best_idx]
                .reset_index(drop=True)
                .sort_values(local_keys)
            )
        else:
            best_df = final_df.copy()

        return final_df, best_df
        
    def count_votes(
            self, 
            best_result: pd.DataFrame
        ) -> Tuple[
            pd.DataFrame, 
            pd.DataFrame, 
            pd.DataFrame, 
            Dict[Metric, InterpMethod]
        ]:
        # Count votes per metric
        votes = (
            best_result
            .groupby( # pyright: ignore[reportUnknownMemberType]
                ['metric', 'method']
            )
            .size()
            .rename('votes')
            .reset_index()
        )
        # Rank methods within each metric
        votes['rank'] = (
            votes
            .groupby( # pyright: ignore[reportUnknownMemberType]
                'metric'
            )['votes']
            .rank(
                method='dense', ascending=False
            )
        )
        votes['share'] = (
            votes['votes'] 
            / votes.groupby( # pyright: ignore[reportUnknownMemberType]
                'metric'
            )['votes']
            .transform('sum')
        )
        # Pick the winner per metric
        winners = (
            votes
            .sort_values(
                ['metric', 'votes', 'method'],
                ascending=[True, False, True]
            )
            .groupby( # pyright: ignore[reportUnknownMemberType]
                'metric', as_index=False
            ) 
            .first()         
        )
        # Pivot to table counts
        table = (
            votes
            .pivot(index='metric', columns='method', values='votes')
            .fillna( # pyright: ignore[reportUnknownMemberType]
                0
            )
            .astype(int)
        )
        # Dict of recommended interpolation methods per metric based on 
        # vote
        recommended_by_metric = dict(zip(winners['metric'], winners['method']))

        return votes, winners, table, recommended_by_metric

class Interpolator(object):

    def __init__(self):
        self.sep_dfs = InterpolationTester().get_device_sep_dfs

    def __get_ramp_mask_for_day(
            self, 
            series: pd.Series, 
            day: datetime, 
            first_light: pd.Series, 
            last_light: pd.Series
        ) -> pd.Series:
        """
        Construct a boolean mask identifying ramp-up and ramp-down 
        periods for a single day.

        The mask is indexed to the timestamps of ``series`` for the 
        specified day and is ``True`` during the first two hours after 
        first light and the last two hours before last light. If first 
        or last light is unavailable for the day, a mask of all `False` 
        values is returned.
        """
        ts = series.loc[day: day].index
        if pd.isna(first_light.loc[day]) or pd.isna(last_light.loc[day]):
            return pd.Series(False, index=ts)
        start = first_light.loc[day]
        end   = last_light.loc[day]
        ramp1_end   = start + pd.Timedelta(hours=2)
        ramp2_start = end   - pd.Timedelta(hours=2)
        mask = (
            ((ts >= start) & (ts < ramp1_end)) |
            ((ts > ramp2_start) & (ts <= end))
        )

        return pd.Series(mask, index=ts)
    
    def __get_daylight_mask(self, idx: pd.DatetimeIndex) -> pd.Series:
        """
        Generate a boolean daylight mask for a given datetime index.

        The mask flags timestamps that fall within predefined daylight
        hours for each month of the year. Daylight start and end hours
        are defined using generic, non-site-specific constants to
        preserve confidentiality.
        """
        months = idx.month
        hours = idx.hour
        # Define daylight start and end hours by month of year
        daylight_map: Dict[int, Tuple[int, int]] = DAYLIGHT_MAP
        # Lookup arrays
        starts = np.array([0] + [daylight_map[m][0] for m in range(1, 13)])
        ends = np.array([0] + [daylight_map[m][1] for m in range(1, 13)])

        start_h = starts[months]
        end_h = ends[months]
        mask = (hours >= start_h) & (hours <= end_h)
        return pd.Series(mask, index=idx, name="is_daylight")

    def __first_true_timestamp(self, s: pd.Series) -> Optional[pd.Timestamp]:
        return s.index[s].min() if s.any() else None
    
    def __last_true_timestamp(self, s: pd.Series) -> Optional[pd.Timestamp]:
        return s.index[s].max() if s.any() else None

    def __get_day_rules(
            self, 
            series: pd.Series, 
            poa: Optional[pd.Series] = None, 
            daylight_thr: float = 20.0
        ) -> pd.DataFrame:
        """
        Evaluate per-day data quality rules based on daylight coverage 
        and missingness patterns.

        For each day in the input time series, this method determines 
        whether the day should be retained for interpolation or 
        downstream analysis. Daylight periods are identified either 
        using a fixed, month-dependent daylight mask or via a 
        plane-of-array (POA) irradiance threshold when provided.

        The decision for each day is based on:
        - The fraction of non-missing observations during daylight hours
        - The maximum contiguous gap of missing values during daylight
        - The maximum contiguous gap of missing values during ramp 
          periods around first and last light

        Days failing any of these criteria are marked for exclusion and
        accompanied by diagnostic statistics.
        """
        # Safety checks
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"Index is wrong type: {type(series.index)}")
        if not series.index.is_monotonic_increasing:
            raise ValueError("Series index must be sorted")
        
        idx: pd.DatetimeIndex = series.index
        # Define mask for daylight
        if poa is None:
            daylight_mask = self.__get_daylight_mask(idx)
        else:
            daylight_mask = (poa > daylight_thr).reindex(idx, fill_value=False)
        
        # Find first and last daylight timestamps per day
        day = idx.normalize()
        by_day = (
            daylight_mask
            .groupby( # pyright: ignore[reportUnknownMemberType]
                day
            )
        )

        first_light = by_day.apply( # pyright: ignore[reportUnknownMemberType]
            self.__first_true_timestamp
        ).astype("datetime64[ns]")
        last_light = by_day.apply( # pyright: ignore[reportUnknownMemberType]
            self.__last_true_timestamp
        ).astype("datetime64[ns]")

        decisions: List[
            Tuple[pd.Timestamp, bool, Dict[str, str | float | int]]
        ] = []
        normalized_series = (
            series
            .groupby( # pyright: ignore[reportUnknownMemberType]
                series.index.normalize()
            )
        )
        for d, series_day in normalized_series:
            if series_day.empty:
                continue
            dm = daylight_mask.loc[series_day.index]
            if not dm.any():
                # Drop day if no daylight observations
                decisions.append((d, False, {"reason": "no_daylight"}))
                continue
            # Daylight observations
            series_daylight = series_day[dm]
            if series_daylight.empty:
                decisions.append((d, False, {"reason": "no_daylight_samples"}))
                continue
            # Valid fraction in daylight
            valid_frac = series_daylight.notna().mean()
            # Max contiguous NaN run in daylight
            m = series_daylight.isna()
            if m.any():
                # run lengths via group changes
                grp = m.ne(m.shift()).cumsum()
                runs = m.groupby( # pyright: ignore[reportUnknownMemberType]
                    grp
                ).sum()
                is_nan_grp = (
                    m.groupby( # pyright: ignore[reportUnknownMemberType]
                        grp
                    ).first()
                )
                max_gap_daylight = int(runs[is_nan_grp].max())
            else:
                max_gap_daylight = 0

            # Ramp gap check
            rmask = self.__get_ramp_mask_for_day(
                series, 
                d, 
                first_light, 
                last_light
            )
            ramp_dm = daylight_mask.loc[rmask.index] & rmask
            s_ramp = series.loc[d: d][ramp_dm]
            if s_ramp.empty:
                max_gap_ramp = 0
            else:
                mr = s_ramp.isna()
                if mr.any():
                    grp_r = mr.ne(mr.shift()).cumsum()
                    runs_r = (
                        mr.groupby( # pyright: ignore[reportUnknownMemberType]
                            grp_r
                        ).sum()
                    )
                    is_nan_grp_r = (
                        mr.groupby( # pyright: ignore[reportUnknownMemberType]
                            grp_r
                        ).first()
                    )
                    max_gap_ramp = int(runs_r[is_nan_grp_r].max())
                else:
                    max_gap_ramp = 0
            # Hyperparameters for allowable missingness in samples to 
            # be interpolated
            keep = (
                (valid_frac >= VALID_FRAC) 
                and (max_gap_daylight <= MAX_DAYLIGHT_GAP) 
                and (max_gap_ramp <= MAX_RAMP_GAP)
            )
            decisions.append((d, keep, {
                "valid_frac_daylight": float(valid_frac),
                "max_gap_daylight": max_gap_daylight,
                "max_gap_ramp": max_gap_ramp
            }))
        out = pd.DataFrame(
            decisions, 
            columns=["day", "keep", "stats"]
        ).set_index("day")
        return out

    def __clean_lf(
            self, 
            lf: pl.LazyFrame, 
            poa: str = Metric.POA_MEDIAN, 
            pivot: bool = False, 
            cols_to_drop: Optional[Tuple[Field, ...]] = None
        ) -> Tuple[set[pd.Timestamp], Dict[str, pd.DataFrame]]:
        """
        Apply per-day data quality rules across devices and return the 
        set of valid days.

        This method separates the input lazy frame into per-device 
        pandas DataFrames, evaluates daily data-quality rules for each 
        numeric metric, and aggregates the results to determine which 
        days satisfy all quality criteria across all devices.

        Quality rules are evaluated at the metric level and include 
        checks on daylight coverage and missingness patterns. A day is 
        retained only if it is marked as valid for every metric of every
        device. The original per-device DataFrames are returned 
        unchanged alongside the set of retained days.

        Returns
        -------
        set[pd.Timestamp]
            Set of days that satisfy all quality rules across all 
            devices.
        Dict[str, pd.DataFrame]
            Mapping from device identifier to the corresponding pandas
            DataFrame derived from the input lazy frame.
        """
        inv_dfs = self.sep_dfs(lf, cols_to_drop, pivot_df=pivot)
        masks: Dict[str, pd.Series] = {}
        for device, df in tqdm(inv_dfs.items(), desc="Inverters"):
            # Consider only numeric cols
            num_cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            if not num_cols:
                continue
            # Get daily quality summaries
            quality: Dict[str, pd.DataFrame] = {}
            for metric in tqdm(num_cols, desc=f"{device}", leave=False):
                s = df[metric]
                if poa:
                    quality[metric] = self.__get_day_rules(s, poa=df[poa])
                else:
                    quality[metric] = self.__get_day_rules(s)
            # Merge daily quality summaries
            qual_df = (
                pd.concat({k: v for k, v in quality.items()}, axis=1)
                .swaplevel(axis=1)
                .sort_index(axis=1)
            )
            # Cast pd.DataFrame.xs output to pd.DataFrame as it is 
            # weakly typed
            temp_mask: pd.DataFrame = cast(
                pd.DataFrame, 
                qual_df.xs("keep", level=0, axis=1)
            )
            masks[device] = temp_mask.all(axis="columns")

        combined = pd.concat(masks, axis=1)
        final_mask  = combined.all(axis=1)
        days_to_keep = set(final_mask.index[final_mask])
        
        return days_to_keep, inv_dfs
        
    def interpolate(
            self, 
            lf: pl.LazyFrame, 
            mode: DataMode = "full",
            order: Optional[int] = 3, 
            pivot: bool = False
        ) -> pd.DataFrame:
        """
        Interpolate time-series metrics for each device after applying
        per-day data quality filtering.

        This method first evaluates daily data-quality rules to 
        determine which days should be retained, then applies 
        metric-specific interpolation methods to the remaining 
        observations for each device. Interpolation is performed 
        independently per device and metric, using predefined method 
        mappings determined by the selected mode.

        Only days that satisfy all quality criteria are included in the
        interpolation step. Interpolation is restricted to gaps fully
        enclosed by valid data (i.e., no extrapolation beyond observed
        ranges).

        Returns
        -------
        pd.DataFrame
            Interpolated time-series data across all devices, indexed 
            by the standard key set and sorted chronologically.
        """
        days_to_keep, inv_dfs = self.__clean_lf(lf, pivot=pivot)
        limit_direction = 'both'
        limit_area = "inside"
        # Dictp[str, str] mapping metric to interpolation method
        if mode == "full":
            interp_map = FULL_INTERPOLATION_MAP
        else:
            interp_map = BASE_INTERPOLATION_MAP
    
        for device, df in tqdm(inv_dfs.items(), desc="Inverters"):
                df = df.copy()
                # if the datetime is a column
                if Column.TIMESTAMP in df.columns:
                    ts = pd.to_datetime(df[Column.TIMESTAMP])
                    day = ts.dt.normalize()
                    df = df[
                        day.isin( # pyright: ignore[reportUnknownMemberType]
                            days_to_keep
                        )
                    ].copy()
                # if it's the index
                elif isinstance(df.index, pd.DatetimeIndex):
                    day = df.index.normalize()
                    df = df[
                        day.isin( # pyright: ignore[reportUnknownMemberType]
                            days_to_keep
                        )
                    ].copy()
                    
                for metric, method in tqdm(interp_map.items(), desc="Metrics"):
                    s = df[metric]
                    df[metric] = s.interpolate(
                        method=method, 
                        order=order, 
                        limit_direction=limit_direction, 
                        limit_area=limit_area
                    )

                df = df.assign(device_name=device)
                df = df.assign(event_local_time=df.index)
                inv_dfs[device] = df
    
        self.final = pd.concat(inv_dfs.values()).set_index(KEYS).sort_index()
        return self.final

    def write_parquet(self, dest: Address = DATA_ROOT, 
                      mode: DataMode = 'full') -> None:
        """Write final interpolated data to parquet."""
        dest = validate_address(dest, mode="w")
        self.final.to_parquet(dest / f"{mode}_train_cleaned.parquet")
