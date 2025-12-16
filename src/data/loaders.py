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


class Open(MemoryAwareProcess):
    """Read and write parquets with polars."""

    def __init__(self, source: Optional[Address] = None, mode: OpenMode = "r",
                 data_mode: DataMode = "full", verbose: Verbosity = 0) -> None:
        super().__init__(verbose=verbose)
        self.mode = mode
        self.data_mode = data_mode
        if source:
            self.source = validate_address(source, mode=mode)
    
    def read(self, low_mem: bool = False):
        """Read parquets from source path defined."""
        lf = pl.scan_parquet(self.source)
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
        return lf

    def write(self, data: Union[pl.LazyFrame, pl.DataFrame]) -> None:
        """Write polars frames to parquet using source path."""
        if isinstance(data, pl.LazyFrame):
            data.sink_parquet(self.source)
        else:
            data.write_parquet(self.source)
    
    def load(self, location: Address, name: str) -> pl.LazyFrame:
        """
        Read parquets specified by name and location in context manager 
        instance.
        """
        location = validate_address(location)
        return pl.scan_parquet(location / name)
