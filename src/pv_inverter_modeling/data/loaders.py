# stdlib
from typing import Optional, Iterable
# thirdpartylib
import polars as pl
import pandas as pd
# projectlib
from pv_inverter_modeling.utils.typing import (
    Address, 
    OpenMode,
    Verbosity,
    DataFrame,
    CollectEngine
)
from pv_inverter_modeling.utils.memory import MemoryAwareProcess
from pv_inverter_modeling.utils.paths import validate_address
from pv_inverter_modeling.preprocessing.reshape import apply_entity_mapping


class Open(MemoryAwareProcess):
    """
    Context-managed interface for reading and writing Parquet datasets.

    This class provides a unified IO wrapper for working with Parquet
    files using Polars (lazy or eager) and pandas, with optional
    memory-safety checks inherited from ``MemoryAwareProcess``. It is
    designed to act as a lightweight context manager around dataset
    access, handling path validation, read/write mode enforcement, and
    verbosity-controlled logging.

    Schema adaptation (column selection and entity-name mapping) is
    delegated to the shared ``apply_entity_mapping`` preprocessing
    utility. This class does not own schema semantics and should not be
    considered the authoritative source of column definitions or
    mappings.

    The class does not eagerly load data by default; read operations
    return Polars ``LazyFrame`` objects unless explicitly materialized
    downstream.

    Notes
    -----
    - This class is IO-focused; schema normalization and reshaping are
      intentionally delegated to preprocessing utilities.
    - Multiple methods expose column-mapping flags, which may indicate
      an opportunity to consolidate mapping policy at a higher level
      (e.g., via a configuration object).

    Parameters
    ----------
    source : Address or None, optional
        Path to the Parquet file or directory to read from or write to.
        If provided, the path is validated according to the specified
        mode.
    mode : OpenMode, optional
        File access mode (e.g., read or write). Defaults to ``"r"``.
    verbose : Verbosity, optional
        Verbosity level controlling diagnostic output. Defaults to 
        ``0``.
    """

    def __init__(
        self,
        source: Optional[Address] = None,
        mode: OpenMode = "r",
        verbose: Verbosity = 0
    ) -> None:
        """
        Initialize an ``Open`` instance with an optional data source and
        access mode.

        This constructor configures the I/O context used for subsequent
        read and write operations. When a source address is provided, it
        is validated according to the specified access mode. No data is
        read or written during initialization.

        Parameters
        ----------
        source : Address or None, optional
            Path or address of the dataset to be accessed. If provided,
            the address is validated immediately according to ``mode``.
        mode : OpenMode, optional
            File access mode indicating the intended operation (e.g.,
            read or write). Defaults to ``"r"``.
        verbose : Verbosity, optional
            Verbosity level controlling diagnostic output and inherited
            by the underlying ``MemoryAwareProcess``. Defaults to ``0``.

        Notes
        -----
        - Validation is performed eagerly for the provided source, but
            no I/O operations are executed during initialization.
        - The source may be omitted and supplied later by methods that
            accept explicit paths.
        """
        super().__init__(verbose=verbose)
        self.mode = mode
        if source:
            self.source = validate_address(source, mode=mode)
    
    def low_mem_state(self, lf: pl.LazyFrame, low_mem: bool) -> None:
        """
        Enforce memory-safety checks when operating in low-memory mode.

        This function conditionally evaluates whether a given lazy 
        Polars query can be executed safely under current memory 
        constraints. When ``low_mem`` is enabled, a memory usage 
        estimate is computed and compared against the available memory 
        budget. If the estimated cost exceeds the available memory, 
        execution is aborted with a ``MemoryError``.

        No action is taken when ``low_mem`` is disabled.

        Parameters
        ----------
        lf : pl.LazyFrame
            Lazy Polars query representing the dataset to be evaluated.
        low_mem : bool
            Flag indicating whether low-memory safeguards should be 
            enforced.

        Raises
        ------
        MemoryError
            If ``low_mem`` is ``True`` and the estimated memory required 
            to materialize the dataset exceeds the available memory 
            budget.

        Returns
        -------
        None
            This function performs validation only and does not return 
            a value.
        """
        if low_mem:
            check, mem, cost = self.memory_check(lf, "gb")
            if not check:
                msg = (
                    "Insufficient available memory to safely read dataset "
                    "in low-memory mode. "
                    f"Estimated memory required: {cost:.2f} GB, "
                    f"available budget: {mem:.2f} GB. "
                    "Consider disabling `low_mem`, reading a subset of the "
                    "data, or increasing available system memory."
                )
                raise MemoryError(msg)
    
    def read(
            self,
            *,
            multi_file: bool = False,  
            low_mem: bool = False,
            columns: Optional[Iterable[str]] = None,
            reverse_col_map: bool = False,
            strict_col_map: bool = False,
        ) -> pl.LazyFrame:
        """
        Lazily read Parquet data from the configured source path.

        This method scans one or more Parquet files from the source path
        defined during class construction and returns a Polars
        ``LazyFrame`` representing the dataset. Schema adaptation
        (column selection and entity-name mapping) is delegated to
        ``apply_entity_mapping`` and may be applied in either direction
        depending on the provided flags.

        The dataset is always read lazily using ``pl.scan_parquet``; no
        materialization occurs within this method.

        Parameters
        ----------
        multi_file : bool, optional
            If ``True``, all ``.parquet`` files found recursively under
            the source path are scanned and combined into a single
            logical dataset. If ``False``, the source path is treated as
            a single Parquet file. Defaults to ``False``.

        low_mem : bool, optional
            If ``True``, enable low-memory safeguards, including
            reduced-memory scanning and a pre-execution memory safety
            check. Defaults to ``False``.

        columns : Iterable[str], optional
            Optional list of column names to project from the dataset.
            Column names must refer to the schema *after* entity-name
            mapping has been applied. When provided, only these columns
            are retained, reducing I/O, memory usage, and downstream
            execution cost.

        reverse_col_map : bool, optional
            If ``True``, apply the reverse entity-name mapping during
            schema adaptation. Defaults to ``False``.

        strict_col_map : bool, optional
            If ``True``, drop columns not explicitly defined in
            ``ENTITY_MAP`` during schema adaptation. Defaults to
            ``False``.

        Returns
        -------
        pl.LazyFrame
            A lazily evaluated Polars query representing the scanned
            dataset, with schema adaptation and optional projection
            applied.

        Raises
        ------
        ValueError
            If the source path was not defined during class 
            construction.

        MemoryError
            If ``low_mem`` is enabled and the estimated memory required
            to materialize the dataset exceeds the available memory
            budget.

        Notes
        -----
        - Schema adaptation is applied before column projection; the
          ``columns`` argument therefore operates on the adapted schema.
        - This method performs no materialization; downstream consumers
          must explicitly call ``collect()`` to execute the query.
        """
        if not self.source:
            msg = "Source not defined in class instance construction."
            raise ValueError(msg)
        
        if multi_file:
            # All .parquet files found in self.source
            root = list(self.source.rglob("*.parquet"))
        else:
            root = self.source
        
        lf = pl.scan_parquet( # type: ignore[reportUnknownMemberType]
            root,
            low_memory=low_mem
        )
        lf = apply_entity_mapping(
            lf,
            reverse=reverse_col_map,
            strict=strict_col_map
        )
        # Column projection
        if columns is not None:
            lf = lf.select(list(columns))

        self.low_mem_state(lf, low_mem)

        return lf

    def _write_parquet(self, data: DataFrame, target: Address) -> None:
        """
        Write a DataFrame-like object to Parquet format.

        This internal helper persists the provided dataset to disk in 
        Parquet format, selecting the appropriate write method based on 
        the input data type. Both eager and lazy Polars objects are 
        supported, as well as pandas DataFrames.

        Parameters
        ----------
        data : DataFrame
            DataFrame-like object to be written. Supported types include
            ``pl.LazyFrame``, ``pl.DataFrame``, and ``pd.DataFrame``.
        target : Address
            Destination path where the Parquet data will be written.

        Returns
        -------
        None
            Writes the dataset to disk and does not return a value.

        Raises
        ------
        TypeError
            If the provided data type is not supported.
        """
        if isinstance(data, pl.LazyFrame):
            data.sink_parquet(target)
        elif isinstance(data, pl.DataFrame):
            data.write_parquet(target)
        else:
            data.to_parquet(target, index=False)

    def write(
            self, 
            data: DataFrame, 
            reverse_mapping: bool = False,
            strict_mapping: bool = False,
        ) -> None:
        """
        Write a DataFrame-like object to Parquet at the configured 
        source path.

        This method persists a pandas or Polars DataFrame to disk in
        Parquet format using the source path defined during class
        construction. Prior to writing, schema adaptation (column
        selection and entity-name mapping) is delegated to
        ``apply_entity_mapping`` and may be applied in either direction
        depending on the provided flags.

        Parameters
        ----------
        data : DataFrame
            DataFrame-like object to be written. Supported types include
            pandas DataFrames and Polars DataFrames (eager or lazy).
        reverse_mapping : bool, optional
            If ``True``, apply the reverse entity-name mapping during
            schema adaptation before writing. Defaults to ``False``.
        strict_mapping : bool, optional
            If ``True``, drop columns not explicitly defined in
            ``ENTITY_MAP`` during schema adaptation. Defaults to
            ``False``.

        Returns
        -------
        None
            Writes the dataset to disk and does not return a value.

        Raises
        ------
        ValueError
            If the source path was not defined during class
            construction.

        Notes
        -----
        - Schema adaptation is applied uniformly across pandas and
          Polars inputs via ``apply_entity_mapping``.
        - This method does not perform any materialization beyond what
          is required by the underlying Parquet write operation.
        """
        if not self.source:
            msg = "Source not defined in class instance construction."
            raise ValueError(msg)
        data = apply_entity_mapping(
            data,
            reverse=reverse_mapping,
            strict=strict_mapping
        )
        self._write_parquet(data, self.source)
    
    def load(
            self, 
            location: Address, 
            name: str, 
            map_cols: bool = False, 
            reverse_map_cols: bool = False,
            strict_map: bool = False, 
            low_mem: bool = False
        ) -> pl.LazyFrame:
        """
        Lazily load a Parquet dataset from a specified location.

        This method scans a Parquet file located at ``location / name``
        and returns a Polars ``LazyFrame`` representing the dataset.
        Schema adaptation (column selection and entity-name mapping) may
        be applied at load time via ``apply_entity_mapping`` if enabled.
        Optional low-memory safety checks can be enforced prior to
        execution.

        The data is read lazily using ``pl.scan_parquet``; no
        materialization occurs within this method.

        Parameters
        ----------
        location : Address
            Base directory containing the Parquet dataset.
        name : str
            Name of the Parquet file (or relative path) to load from
            ``location``.
        map_cols : bool, optional
            If ``True``, apply entity-name mapping during schema
            adaptation after loading. Defaults to ``False``.
        reverse_map_cols : bool, optional
            If ``True``, apply the reverse entity-name mapping during
            schema adaptation. Ignored if ``map_cols`` is ``False``.
            Defaults to ``False``.
        strict_map : bool, optional
            If ``True``, drop columns not explicitly defined in
            ``ENTITY_MAP`` during schema adaptation. Defaults to
            ``False``.
        low_mem : bool, optional
            If ``True``, enforce low-memory safeguards by performing a
            pre-execution memory safety check on the lazy query.
            Defaults to ``False``.

        Returns
        -------
        pl.LazyFrame
            A lazy Polars query representing the loaded dataset, with
            optional schema adaptation applied.

        Raises
        ------
        MemoryError
            If ``low_mem`` is enabled and the estimated memory required
            to materialize the dataset exceeds the available memory
            budget.

        Notes
        -----
        - Schema adaptation is applied only when ``map_cols`` is
          enabled and is delegated to ``apply_entity_mapping``.
        - This method performs no materialization; downstream consumers
          must explicitly call ``collect()`` to execute the query.
        """
        location = validate_address(location)
        lf = pl.scan_parquet( # type: ignore[reportUnknownMemberType]
            location / name
        ) 
        self.low_mem_state(lf, low_mem)
        if map_cols:
            lf = apply_entity_mapping(
                lf, 
                reverse=reverse_map_cols,
                strict=strict_map,
            )

        return lf

def load_lazyframe(
        source: Address,
        *,
        verbose: Verbosity = 0,
        multi_file: bool = False,
        low_mem: bool = False,
        columns: Optional[Iterable[str]] = None,
        reverse_col_map: bool = False,
        strict_col_map: bool = False,
    ) -> pl.LazyFrame:
    """
    Lazily load a Parquet dataset as a Polars ``LazyFrame``.

    This utility function provides a standardized, memory-aware entry
    point for loading Parquet datasets using the project's I/O
    abstraction. One or more Parquet files are scanned from the given
    source address and returned as a Polars ``LazyFrame`` without
    immediately materializing the data into memory.

    All path validation, schema adaptation (column selection and
    entity-name mapping), optional column projection, and memory-safety
    checks are delegated to the ``Open`` context manager, ensuring
    consistent and centralized dataset access semantics across the
    codebase.

    Parameters
    ----------
    source : Address
        Path or address of the dataset to load. May refer to a single
        Parquet file or a directory containing multiple Parquet files,
        depending on ``multi_file``.
    verbose : Verbosity, optional
        Verbosity level forwarded to the ``Open`` context manager to
        control diagnostic output. Defaults to ``0``.
    multi_file : bool, optional
        If ``True``, all Parquet files found under ``source`` are 
        scanned and combined into a single logical lazy query. If 
        ``False``, only the file at ``source`` is scanned. Defaults to 
        ``False``.
    low_mem : bool, optional
        If ``True``, enable low-memory safeguards, including a
        pre-execution memory safety check. Defaults to ``False``.
    columns : Iterable[str], optional
        Optional list of column names to project from the dataset.
        Column names must refer to the schema *after* entity-name
        mapping has been applied. When provided, only these columns are
        retained, reducing I/O, memory usage, and downstream execution
        cost.
    reverse_col_map : bool, optional
        If ``True``, apply the reverse entity-name mapping during schema
        adaptation. Defaults to ``False``.
    strict_col_map : bool, optional
        If ``True``, drop columns not explicitly defined in
        ``ENTITY_MAP`` during schema adaptation. Defaults to ``False``.

    Returns
    -------
    pl.LazyFrame
        A lazily evaluated Polars ``LazyFrame`` representing the
        dataset, with schema adaptation, optional projection, and
        memory-safety checks applied.

    Raises
    ------
    RuntimeError
        If the dataset cannot be read or the ``LazyFrame`` fails to
        initialize.

    Notes
    -----
    - This function performs no materialization; downstream consumers
      must explicitly call ``collect()`` to execute the query.
    - Schema adaptation and projection semantics are inherited directly
      from ``Open.read``.
    """
    lf: pl.LazyFrame | None = None
    with Open(source, verbose=verbose) as o:
        lf = o.read(
            multi_file=multi_file, 
            low_mem=low_mem,
            columns=columns,
            reverse_col_map=reverse_col_map,
            strict_col_map=strict_col_map,
        )
    if lf is None:
        raise RuntimeError("Failed to read data")
    return lf

def load_pandas(
        source: Address,
        *,
        engine: CollectEngine = "streaming",
        verbose: Verbosity = 0,
        multi_file: bool = False,
        low_mem: bool = False,
        columns: Optional[Iterable[str]] = None,
        reverse_col_map: bool = False,
        strict_col_map: bool = False,
    ) -> pd.DataFrame:
    """
    Load a dataset into a pandas ``DataFrame`` via a Polars lazy
    execution plan.

    This function provides a convenience wrapper around
    ``load_lazyframe`` for workflows that require pandas compatibility.
    The dataset is scanned lazily using Polars, optionally subjected to
    schema adaptation (column selection and entity-name mapping),
    materialized into memory using the specified collection engine, and
    converted to a pandas ``DataFrame``.

    All path validation, schema adaptation, optional column projection,
    and memory-safety checks are delegated to the project's standardized
    I/O layer, ensuring consistent dataset access semantics across eager
    and lazy workflows.

    Parameters
    ----------
    source : Address
        Path or address of the dataset to load. May refer to a single
        file or a directory, depending on ``multi_file``.
    engine : CollectEngine, optional
        Polars collection engine used when materializing the lazy query
        plan (e.g., ``"streaming"`` or ``"gpu"``). Defaults to
        ``"streaming"``.
    verbose : Verbosity, optional
        Verbosity level forwarded to the underlying I/O layer. Defaults
        to ``0``.
    multi_file : bool, optional
        If ``True``, all compatible files under ``source`` are scanned
        and combined into a single logical dataset. Defaults to
        ``False``.
    low_mem : bool, optional
        If ``True``, enable memory-safety checks during loading and
        collection. Defaults to ``False``.
    columns : Iterable[str], optional
        Optional list of column names to project from the dataset.
        Column names must refer to the schema *after* entity-name
        mapping has been applied. When provided, only these columns are
        loaded, reducing I/O, memory usage, and conversion cost.
    reverse_col_map : bool, optional
        If ``True``, apply the reverse entity-name mapping during schema
        adaptation. Defaults to ``False``.
    strict_col_map : bool, optional
        If ``True``, drop columns not explicitly defined in
        ``ENTITY_MAP`` during schema adaptation. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        The fully materialized dataset as a pandas ``DataFrame``.

    Raises
    ------
    RuntimeError
        If the dataset cannot be read or materialization fails.

    Notes
    -----
    - This function eagerly materializes the dataset into memory; for
      large datasets, prefer ``load_lazyframe`` and defer collection.
    - Schema adaptation and projection semantics are inherited directly
      from ``load_lazyframe`` and ``Open.read``.
    """
    lf = load_lazyframe(
        source,
        verbose=verbose,
        multi_file=multi_file,
        low_mem=low_mem,
        columns=columns,
        reverse_col_map=reverse_col_map,
        strict_col_map=strict_col_map,
    )
    return lf.collect(engine=engine).to_pandas()
