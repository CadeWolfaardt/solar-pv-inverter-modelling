# stdlib
from typing import Sequence, Optional, Union, cast
# thirdpartylib
import polars as pl
import pandas as pd
# projectlib
from pv_inverter_modeling.utils.typing import (
    DataFrame, 
    Aggregation,
    PdAggregation,
    PlAggregation,
    Field,
    GenericAggregation,
    TypeAliasType,
    custom_get_args
)
from pv_inverter_modeling.data.schemas import KEYS, Column

def _normalize_agg(
        agg: Aggregation,
        *,
        allowed: TypeAliasType,
        fallback: GenericAggregation,
    ) -> Aggregation:
    """
    Normalize an aggregation identifier to a backend-supported value.

    If the provided aggregation is not included in the set of values
    allowed by the given type alias, a fallback aggregation is returned
    instead. This helper is intended to reconcile a generic aggregation
    choice with backend-specific aggregation constraints (e.g. pandas
    vs. polars).

    Parameters
    ----------
    agg : Aggregation
        Requested aggregation identifier.
    allowed : TypeAliasType
        Type alias defining the set of aggregation values supported by
        the target backend.
    fallback : GenericAggregation
        Aggregation to use when the requested value is not supported.

    Returns
    -------
    Aggregation
        A valid aggregation identifier guaranteed to be supported by the
        backend.
    """
    # Validate the aggregation against the allowed backend-specific 
    # values
    return agg if agg in custom_get_args(allowed) else fallback

def pivot_polars(
        data: Union[pl.LazyFrame, pl.DataFrame],
        *,
        on: Field = Column.METRIC,
        on_columns: Optional[Sequence[str]] = None,
        index: Sequence[str] = [*KEYS],
        values: Field = Column.VALUE,
        agg_func: PlAggregation = "median",
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Pivot a Polars DataFrame or LazyFrame from long to wide format.

    For eager ``DataFrame`` inputs, Polars automatically discovers the
    set of pivot columns. For ``LazyFrame`` inputs, the pivot column
    domain must be materialized explicitly to produce a stable schema,
    which is handled internally when ``on_columns`` is not provided.

    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data in long format.
    on : Field, default=Column.METRIC
        Column whose distinct values become the output columns.
    on_columns : Optional[Sequence[str]], default=None
        Explicit list of pivot column names. Required for lazy execution
        if the column domain cannot be inferred ahead of time.
    index : Sequence[str], default=KEYS
        Columns used to form the row index of the pivoted output.
    values : Field, default=Column.VALUE
        Column containing the values to aggregate.
    agg_func : PlAggregation, default="median"
        Aggregation function applied when multiple values exist per
        (index, column) pair.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Pivoted data in wide format.
    """
    if isinstance(data, pl.LazyFrame):
        # LazyFrame requires an explicit column domain for pivoting
        if on_columns is None:
            on_columns = (
                data
                .select(pl.col(Column.METRIC).unique().sort())
                .collect()
                .to_series()
                .to_list()
            )
        if not on_columns:
            raise ValueError("No metric columns found for pivot")
    data = (
        data
        .pivot(
            on=on,
            on_columns=on_columns, # pyright: ignore[reportArgumentType]
            index=index,
            values=values,
            aggregate_function=agg_func
        )
    )
    return data

def pivot_long_to_wide(
        data: DataFrame,
        *,
        on: Field = Column.METRIC,
        on_columns: Optional[Sequence[str]] = None,
        index: Sequence[str] = tuple(KEYS),
        values: Field = Column.VALUE,
        agg_func: Aggregation = "median",
        fallback: GenericAggregation = "median",
    ) -> DataFrame:
    """
    Pivot long-format metric data into wide format for pandas or Polars.

    This helper normalizes a generic aggregation identifier to a 
    backend-specific aggregation supported by the input DataFrame type, 
    then performs a pivot operation using the appropriate backend API. 
    For Polars ``LazyFrame`` inputs, the pivot column domain may be
    materialized explicitly to ensure a stable schema.

    Parameters
    ----------
    data : DataFrame
        Input data in long format. May be a pandas DataFrame or a Polars
        DataFrame / LazyFrame.
    on : Field, default=Column.METRIC
        Column whose distinct values become the output columns.
    on_columns : Optional[Sequence[str]], default=None
        Explicit list of pivot column names. Required for Polars
        ``LazyFrame`` execution when the column domain cannot be 
        inferred lazily.
    index : Sequence[str], default=KEYS
        Columns used to form the row index of the pivoted output.
    values : Field, default=Column.VALUE
        Column containing the values to aggregate.
    agg_func : Aggregation, default="median"
        Generic aggregation identifier to apply.
    fallback : GenericAggregation, default="median"
        Aggregation to use when the requested aggregation is not 
        supported by the backend.

    Returns
    -------
    DataFrame
        Pivoted data in wide format using the appropriate backend.
    """
    if isinstance(data, pd.DataFrame):
        # Normalize aggregation to a pandas-supported value
        aggfunc_pd = cast(
            PdAggregation,
            _normalize_agg(
                agg_func,
                allowed=PdAggregation,
                fallback=fallback,
            ),
        )
        return pd.pivot_table( # pyright: ignore[reportUnknownMemberType]
            data,
            values=values,
            index=index,
            columns=on,
            aggfunc=aggfunc_pd,
            sort=True,
            dropna=False,
        )
    # Normalize aggregation to a Polars-supported value
    aggfunc_pl = cast(
        PlAggregation,
        _normalize_agg(
            agg_func,
            allowed=PlAggregation,
            fallback=fallback,
        ),
    )
    return pivot_polars(
        data,
        on=on,
        on_columns=on_columns,
        index=index,
        values=values,
        agg_func=aggfunc_pl,
    )