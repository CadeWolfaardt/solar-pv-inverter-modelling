# stdlib
from typing import Sequence, Optional, Union, cast, overload
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
from pv_inverter_modeling.config.private_map import (
    ENTITY_MAP, 
    REVERSE_ENTITY_MAP
)

@overload
def apply_entity_mapping(
        data: pl.LazyFrame,
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> pl.LazyFrame: ...

@overload
def apply_entity_mapping(
        data: pl.DataFrame,
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> pl.DataFrame: ...

@overload
def apply_entity_mapping(
        data: pd.DataFrame,
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> pd.DataFrame: ...

def apply_entity_mapping(
        data: DataFrame,
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> DataFrame:
    """
    Apply a standardized entity-based column name mapping to tabular 
    data.

    This function adapts column names using ``ENTITY_MAP`` while 
    optionally enforcing strict schema compliance. For Polars inputs, 
    the operation includes safe column selection and renaming based on 
    the observed schema. For pandas inputs, a direct rename is applied.

    When ``strict`` is enabled, only columns explicitly defined in
    ``ENTITY_MAP`` are retained. When disabled, unmapped columns are 
    preserved by mapping them to themselves. Column renaming may be 
    applied in either direction depending on the ``reverse`` flag.

    Lazy Polars inputs are handled without materializing data, ensuring
    compatibility with lazy execution pipelines.

    Parameters
    ----------
    data : pl.DataFrame, pl.LazyFrame, or pd.DataFrame
        Input tabular data whose column names will be adapted.
    reverse : bool, optional
        If ``False`` (default), apply the forward mapping defined in
        ``ENTITY_MAP``. If ``True``, apply the reverse mapping.
    strict : bool, optional
        If ``True``, drop columns not explicitly present in 
        ``ENTITY_MAP``. If ``False`` (default), preserve unmapped 
        columns unchanged.

    Returns
    -------
    pl.DataFrame, pl.LazyFrame, or pd.DataFrame
        Data of the same type as the input with column names mapped and
        optional schema filtering applied.
    """
    # Pandas supports direct renaming without schema inspection
    if isinstance(data, pd.DataFrame):
        mapping = REVERSE_ENTITY_MAP if reverse else ENTITY_MAP
        return data.rename(mapping)
    # Obtain the column schema without triggering execution for lazy 
    # inputs
    schema = (
        data.collect_schema()
        if isinstance(data, pl.LazyFrame)
        else data.schema
    )
    # Cache the global mapping locally for clarity and lookup efficiency
    base_map = ENTITY_MAP
    # Build a column-level mapping consistent with strictness semantics
    col_map = (
        {col: base_map[col] for col in schema if col in base_map}
        if strict
        else {col: base_map.get(col, col) for col in schema}
    )
    # Select only columns that are safe to propagate forward
    data = data.select([col for col in schema if col in col_map])
    # Construct a rename mapping in the requested direction
    rename_map = (
        {v: k for k, v in col_map.items() if v in schema}
        if reverse
        else {k: v for k, v in col_map.items() if k in schema}
    )

    return data.rename(rename_map)

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
        map_cols: bool = True,
        reverse_map_cols: bool = False,
        strict_map_cols: bool = False,
    ) -> DataFrame:
    """
    Pivot long-format metric data into wide format for pandas or Polars.

    This helper converts long-format tabular data into a wide
    representation by pivoting metric values into columns. The pivot
    operation is performed using the appropriate backend API depending
    on the input data type, with aggregation semantics normalized from a
    generic identifier to a backend-supported aggregation.

    Optional schema adaptation (column selection and entity-name
    mapping) may be applied to the pivoted output via
    ``apply_entity_mapping``.

    Parameters
    ----------
    data : DataFrame
        Input data in long format. May be a pandas DataFrame or a Polars
        DataFrame or LazyFrame.
    on : Field, optional
        Column whose distinct values become the output columns.
        Defaults to ``Column.METRIC``.
    on_columns : Sequence[str] or None, optional
        Explicit list of pivot column names. Required for Polars
        ``LazyFrame`` execution when the column domain cannot be 
        inferred lazily.
    index : Sequence[str], optional
        Columns used to form the row index of the pivoted output.
        Defaults to ``KEYS``.
    values : Field, optional
        Column containing the values to aggregate. Defaults to
        ``Column.VALUE``.
    agg_func : Aggregation, optional
        Generic aggregation identifier to apply. Defaults to
        ``"median"``.
    fallback : GenericAggregation, optional
        Aggregation to use when the requested aggregation is not
        supported by the backend. Defaults to ``"median"``.
    map_cols : bool, optional
        If ``True`` (default), apply schema adaptation to the pivoted
        output using ``apply_entity_mapping``.
    reverse_map_cols : bool, optional
        If ``True``, apply the reverse entity-name mapping during schema
        adaptation. Defaults to ``False``.
    strict_map_cols : bool, optional
        If ``True``, drop columns not explicitly defined in
        ``ENTITY_MAP`` during schema adaptation. Defaults to ``False``.

    Returns
    -------
    DataFrame
        Pivoted data in wide format using the appropriate backend, with
        optional schema adaptation applied.
    """
    if isinstance(data, pd.DataFrame):
        agg = cast(
            PdAggregation,
            _normalize_agg(
                agg_func,
                allowed=PdAggregation,
                fallback=fallback,
            ),
        )
        frame = pd.pivot_table(  # pyright: ignore[reportUnknownMemberType]
            data,
            values=values,
            index=index,
            columns=on,
            aggfunc=agg,
            sort=True,
            dropna=False,
        )
    else:
        agg = cast(
            PlAggregation,
            _normalize_agg(
                agg_func,
                allowed=PlAggregation,
                fallback=fallback,
            ),
        )
        frame = pivot_polars(
            data,
            on=on,
            on_columns=on_columns,
            index=index,
            values=values,
            agg_func=agg,
        )
    if map_cols:
        frame = apply_entity_mapping(
            frame,
            reverse=reverse_map_cols,
            strict=strict_map_cols,
        )
    
    return frame