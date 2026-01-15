# stdlib
from typing import Optional, Union, overload
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
from pv_inverter_modeling.config.private_map import (
    ENTITY_MAP, 
    REVERSE_ENTITY_MAP
)

class Open(MemoryAwareProcess):
    """
    Context-managed interface for reading and writing Parquet datasets.

    This class provides a unified IO wrapper for working with Parquet 
    files using Polars (lazy or eager) and pandas, with optional 
    memory-safety checks inherited from ``MemoryAwareProcess``. It is 
    designed to act as a lightweight context manager around dataset 
    access, handling path validation, read/write mode enforcement, and 
    verbosity-controlled logging.

    The class does not eagerly load data by default; read operations 
    return Polars ``LazyFrame`` objects unless explicitly materialized 
    downstream.

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
        super().__init__(verbose=verbose)
        self.mode = mode
        if source:
            self.source = validate_address(source, mode=mode)
    
    @overload
    def map_names(
            self, 
            data: pl.LazyFrame, 
            reverse: bool = False
        ) -> pl.LazyFrame: ...

    @overload
    def map_names(
            self, 
            data: pl.DataFrame, 
            reverse: bool = False
        ) -> pl.DataFrame: ...

    def map_names(
            self, 
            data: Union[pl.LazyFrame, pl.DataFrame],
            reverse: bool = False,
        ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Map column names using a predefined entity mapping and drop 
        unknown columns.

        This function applies a controlled column-selection and renaming
        step to a Polars DataFrame or LazyFrame. Only columns explicitly
        defined in ``ENTITY_MAP`` are retained; all other columns are 
        dropped. Column names are then renamed according to the mapping, 
        either from private to public names or in reverse, depending on 
        the ``reverse`` flag.

        The function is schema-aware: for lazy inputs, the schema is 
        obtained without materializing the data, ensuring compatibility 
        with lazy execution pipelines.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input Polars DataFrame containing columns to be selected and 
            renamed.
        reverse : bool, optional
            If ``False`` (default), map from private/internal column 
            names to public/exposed names. If ``True``, apply the 
            reverse mapping from public names back to private/internal 
            names.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            A DataFrame of the same type as the input with only mapped 
            columns retained and column names renamed according to 
            ``ENTITY_MAP``.
        """
        if isinstance(data, pl.LazyFrame):
            schema = data.collect_schema()
        else:
            schema = data.schema
        # Select known columns
        safe_select = [
            col for col in schema 
            if col in ENTITY_MAP.keys() 
        ]
        data = data.select(safe_select)
        # Rename columns
        if reverse:
            safe_map = {
                new: old
                for old, new in ENTITY_MAP.items()
                if new in schema
            }
        else:
            safe_map = {
                old: new
                for old, new in ENTITY_MAP.items()
                if old in schema
            }

        return data.rename(safe_map)

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
            multi_file: bool = False,  
            low_mem: bool = False
        ) -> pl.LazyFrame:
        """
        Read Parquet data from the configured source path.

        This method scans one or more Parquet files from the source path
        defined during class construction and returns a Polars 
        ``LazyFrame`` representing the dataset. When enabled, column 
        names are mapped to their public schema and optional low-memory 
        safety checks are applied before execution.

        The data is always read lazily using ``pl.scan_parquet``; no 
        materialization occurs inside this method.

        Parameters
        ----------
        multi_file : bool, optional
            If ``True``, all ``.parquet`` files found recursively under 
            the source path are scanned as a single dataset. If 
            ``False``, the source path is treated as a single Parquet 
            file. Defaults to ``False``.
        low_mem : bool, optional
            If ``True``, enable low-memory safeguards, including 
            reduced-memory scanning and a pre-execution memory safety 
            check. Defaults to ``False``.

        Returns
        -------
        pl.LazyFrame
            A lazy Polars query representing the scanned dataset with 
            mapped column names applied.

        Raises
        ------
        ValueError
            If the source path was not defined during class 
            construction.
        MemoryError
            If ``low_mem`` is enabled and the estimated memory required 
            to materialize the dataset exceeds the available memory 
            budget.
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
        lf = self.map_names(lf)
        self.low_mem_state(lf, low_mem)

        return lf

    def _apply_name_mapping(
            self,
            data: DataFrame,
            *,
            reverse: bool,
        ) -> DataFrame:
        """
        Apply optional reverse column-name mapping to a 
        DataFrame-like object.

        This internal helper conditionally applies a reverse column-name 
        mapping when ``reverse`` is enabled. Pandas and Polars inputs 
        are handled appropriately using the corresponding renaming 
        mechanisms. When ``reverse`` is ``False``, the input object is 
        returned unchanged.

        Parameters
        ----------
        data : DataFrame
            Input DataFrame-like object (pandas or Polars, eager or 
            lazy).
        reverse : bool
            If ``True``, apply the reverse entity-name mapping. If 
            ``False``, return the input data unchanged.

        Returns
        -------
        DataFrame
            DataFrame-like object of the same type as the input, with 
            column names optionally mapped in reverse.
        """
        if not reverse:
            return data
        if isinstance(data, pd.DataFrame):
            return data.rename(REVERSE_ENTITY_MAP)
        # Polars (lazy or eager)
        return self.map_names(data, reverse=True)

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
            reverse_mapping: bool = False
        ) -> None:
        """
        Write a DataFrame-like object to Parquet at the configured 
        source path.

        This method persists a pandas or Polars DataFrame to disk in 
        Parquet format using the source path defined during class 
        construction. An optional reverse column-name mapping can be 
        applied prior to writing, allowing internal column names to be 
        converted back to their original or private representations.

        Parameters
        ----------
        data : DataFrame
            DataFrame-like object to be written. Supported types include
            pandas DataFrames and Polars DataFrames (eager or lazy).
        reverse_mapping : bool, optional
            If ``True``, apply the reverse entity-name mapping before 
            writing the data. Defaults to ``False``.

        Returns
        -------
        None
            Writes the dataset to disk and does not return a value.

        Raises
        ------
        ValueError
            If the source path was not defined during class 
            construction.
        """
        if not self.source:
            msg = "Source not defined in class instance construction."
            raise ValueError(msg)
        
        data = self._apply_name_mapping(
            data,
            reverse=reverse_mapping,
        )
        self._write_parquet(data, self.source)
    
    def load(
            self, 
            location: Address, 
            name: str, 
            map_cols: bool = False, 
            reverse_map_cols: bool = False, 
            low_mem: bool = False
        ) -> pl.LazyFrame:
        """
        Lazily load a Parquet dataset from a specified location.

        This method scans a Parquet file located at ``location / name`` 
        and returns a Polars ``LazyFrame`` representing the dataset. 
        Optional column name mapping can be applied at load time, and 
        low-memory safety checks may be enforced prior to execution.

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
            If ``True``, apply the forward entity-name mapping defined in
            ``ENTITY_MAP`` to rename columns after loading. Defaults to 
            ``False``.
        reverse_map_cols : bool, optional
            If ``True``, apply the reverse entity-name mapping defined in
            ``REVERSE_ENTITY_MAP`` to rename columns after loading. 
            Ignored if ``map_cols`` is ``True``. Defaults to ``False``.
        low_mem : bool, optional
            If ``True``, enforce low-memory safeguards by performing a 
            pre-execution memory safety check on the lazy query. 
            Defaults to ``False``.

        Returns
        -------
        pl.LazyFrame
            A lazy Polars query representing the loaded dataset, with 
            optional column-name mapping applied.

        Raises
        ------
        MemoryError
            If ``low_mem`` is enabled and the estimated memory required 
            to materialize the dataset exceeds the available memory 
            budget.
        """
        location = validate_address(location)
        lf = pl.scan_parquet( # type: ignore[reportUnknownMemberType]
            location / name
        ) 
        self.low_mem_state(lf, low_mem)
        if map_cols:
            lf = lf.rename(ENTITY_MAP)
        elif reverse_map_cols:
            lf = lf.rename(REVERSE_ENTITY_MAP) 
        return lf

def load_lazyframe(
        source: Address,
        *,
        verbose: Verbosity = 0,
        multi_file: bool = False,
        low_mem: bool = False,
    ) -> pl.LazyFrame:
    """
    Load a dataset as a Polars LazyFrame using the project I/O 
    abstraction.

    This utility function reads one or more Parquet files from the given
    source address and returns a Polars ``LazyFrame`` without 
    immediately materializing the data into memory. It provides a 
    lightweight, standardized entry point for lazy, memory-aware data 
    processing workflows.

    The function delegates all validation, schema handling, and optional
    memory-safety checks to the ``Open`` context manager, ensuring
    consistent behavior across the codebase.

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
        If ``True``, all Parquet files found under ``source`` are read
        and combined into a single lazy query plan. If ``False``, only
        the file at ``source`` is read. Defaults to ``False``.
    low_mem : bool, optional
        If ``True``, enables low-memory safeguards during dataset
        loading. The underlying reader may refuse to proceed if
        available system memory is insufficient. Defaults to ``False``.

    Returns
    -------
    polars.LazyFrame
        A lazily evaluated Polars ``LazyFrame`` representing the 
        dataset.

    Raises
    ------
    RuntimeError
        If the dataset cannot be read or the ``LazyFrame`` fails to
        initialize.

    Notes
    -----
    - This function does not materialize the data; downstream operations
      must explicitly call ``collect()`` to trigger execution.
    - No dataset-specific constants or proprietary values are embedded
      in this function, making it safe for reuse across projects and
      environments.
    """
    lf: pl.LazyFrame | None = None
    with Open(source, verbose=verbose) as o:
        lf = o.read(multi_file=multi_file, low_mem=low_mem)
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
    ) -> pd.DataFrame:
    """
    Load a dataset into a pandas DataFrame via a Polars lazy execution 
    plan.

    This function reads a dataset from the given source using the 
    project's standardized I/O abstraction, materializes it from a 
    Polars ``LazyFrame`` into memory, and converts the result to a 
    pandas ``DataFrame``. It serves as a thin convenience wrapper for 
    workflows that require pandas compatibility while retaining 
    Polars-based loading and execution semantics.

    Internally, this function delegates dataset access to
    ``load_lazyframe`` to ensure consistent handling of multi-file 
    inputs, verbosity, and low-memory safeguards across the codebase.

    Parameters
    ----------
    source : Address
        Path or address of the dataset to load. May refer to a single
        file or a directory, depending on ``multi_file``.
    engine : CollectEngine, optional
        Polars collection engine used when materializing the lazy query
        plan (e.g., ``"streaming"`` or ``"eager"``). Defaults to
        ``"streaming"``.
    verbose : Verbosity, optional
        Verbosity level forwarded to the underlying I/O layer.
        Defaults to ``0``.
    multi_file : bool, optional
        If ``True``, all compatible files under ``source`` are read and
        combined into a single logical dataset. Defaults to ``False``.
    low_mem : bool, optional
        If ``True``, enables memory-safety checks during loading and
        collection. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        The fully materialized dataset as a pandas DataFrame.

    Raises
    ------
    RuntimeError
        If the dataset cannot be read or materialization fails.

    Notes
    -----
    - This function eagerly materializes the dataset into memory; for
      large datasets, prefer ``load_lazyframe`` and defer collection.
    - No dataset-specific constants, paths, or proprietary thresholds
      are embedded in this function.
    """
    lf = load_lazyframe(
        source,
        verbose=verbose,
        multi_file=multi_file,
        low_mem=low_mem,
    )
    return lf.collect(engine=engine).to_pandas()
