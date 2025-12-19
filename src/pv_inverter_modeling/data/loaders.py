# stdlib
from typing import Optional, Union
# thirdpartylib
import polars as pl
# projectlib
from pv_inverter_modeling.utils.typing import (
    Address, 
    OpenMode, 
    DataMode, 
    Verbosity
)
from pv_inverter_modeling.utils.memory import MemoryAwareProcess
from pv_inverter_modeling.utils.util import validate_address
from pv_inverter_modeling.config.private_map import (
    ENTITY_MAP, 
    REVERSE_ENTITY_MAP
)


class Open(MemoryAwareProcess):
    """Read and write parquets with polars."""

    def __init__(self, source: Optional[Address] = None, mode: OpenMode = "r",
                 data_mode: DataMode = "full", verbose: Verbosity = 0) -> None:
        super().__init__(verbose=verbose)
        self.mode = mode
        self.data_mode = data_mode
        if source:
            self.source = validate_address(source, mode=mode)

    def map_names(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Map private column names to public names and drop unused 
        columns.
        """
        schema = lf.collect_schema()
        # Select known columns
        safe_select = [
            col for col in schema 
            if col in ENTITY_MAP.keys() 
        ]
        lf = lf.select(safe_select)
        # Rename columns
        safe_map = {
            old: new
            for old, new in ENTITY_MAP.items()
            if old in schema
        }

        return lf.rename(safe_map)

    def low_mem_state(self, lf: pl.LazyFrame, low_mem: bool) -> None:
        """
        Determine data loading safety through memory checks if the 
        process is being run in a low memory state.
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
    
    def read(self, multi_file: bool = False,  
             low_mem: bool = False) -> pl.LazyFrame:
        """Read parquets from source path defined on construction."""
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

    def write(self, data: Union[pl.LazyFrame, pl.DataFrame], 
              reverse_mapping: bool = False) -> None:
        """Write polars frames to parquet using source path."""
        if not self.source:
            msg = "Source not defined in class instance construction."
            raise ValueError(msg)
        
        if reverse_mapping:
            data = data.rename(REVERSE_ENTITY_MAP)
        if isinstance(data, pl.LazyFrame):
            data.sink_parquet(self.source)
        else:
            data.write_parquet(self.source)
    
    def load(self, location: Address, name: str, map_cols: bool = False, 
             reverse_map_cols: bool = False, 
             low_mem: bool = False) -> pl.LazyFrame:
        """
        Read parquets specified by name and location in context manager 
        instance.
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
