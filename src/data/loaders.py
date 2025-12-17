# stdlib
from typing import Optional, Union
# thirdpartylib
import polars as pl
# projectlib
from utils.typing import (
    Address, 
    OpenMode, 
    DataMode, 
    Verbosity
)
from utils.memory import MemoryAwareProcess
from utils.util import validate_address
from config.private_map import ENTITY_MAP, REVERSE_ENTITY_MAP

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
        """Map private column names to public names."""
        return lf.rename(ENTITY_MAP)

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
    
    def read(self, low_mem: bool = False):
        """Read parquets from source path defined."""
        if not self.source:
            msg = "Source not defined in class instance construction."
            raise ValueError(msg)
        
        lf = pl.scan_parquet(self.source)
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
        lf = pl.scan_parquet(location / name)
        self.low_mem_state(lf, low_mem)
        if map_cols:
            lf = lf.rename(ENTITY_MAP)
        elif reverse_map_cols:
            lf = lf.rename(REVERSE_ENTITY_MAP) 
        return lf
