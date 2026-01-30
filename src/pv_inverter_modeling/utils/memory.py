# stdlib
import os
import sys
import gc
import ctypes
from ctypes import wintypes
from typing import Optional, cast, Any, Tuple
from types import TracebackType
from pathlib import Path
# thirdpartylib
import psutil
import polars as pl
# projectlib
from pv_inverter_modeling.utils.typing import Verbosity, Address, SizeUnit
from pv_inverter_modeling.utils.logging import Logger

class MemoryAwareProcess(object):
    """
    Parent class for memory intensive processes, meant for improving 
    memory efficiency and handling.
    """
    def __init__(self, verbose: Verbosity = 0, 
                 log_address: Address = Path.cwd(), 
                 write_log: bool = False) -> None:
        self.verbose = verbose
        self.logger = Logger(verbose, log_address, write_log)
    
    def __enter__(self):
        return self
    
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType]
    ) -> bool:
        self.__clean_up_memory()
        # return False to re-raise exceptions (default behavior)
        return False

    def __unit_mapping(self, var: float, unit: SizeUnit) -> float:
        """Map byte measurements to preffered unit size."""
        map = {
            "b": 1, 
            "kb": 1024, 
            "mb": 1048576,
            "gb": 1073741824,  
        }
        if unit not in map.keys():
            msg = f"Invalid unit {unit}. Choose from 'b', 'kb', 'mb', 'gb'."
            raise ValueError(msg)
        
        return var / map[unit]

    def __get_memory_usage(self, returns: bool = False, 
                           verbosity: Verbosity = 0, 
                           unit: SizeUnit = "mb") -> Optional[float]:
        """
        Log or return amount of physical RAM currently mapped into this 
        process.

        :param returns: Whether to return memory used
        :param verbosity: Verbosity level required to log output.
        :param unit: Unit of reported memory usage.
        """
        process = psutil.Process(os.getpid())
        mem_usage = self.__unit_mapping(process.memory_info().rss, unit)
        self.logger(f"Memory usage: {mem_usage:.2f} {unit.upper()}", verbosity)
        if returns:
            return mem_usage
    
    def __collect_garbage_ipython(self, clear_cache: bool = False) -> None:
        """
        Clear IPython cache, last variable from name space, and reclaim
        memory occupied by objects no longer in use.
        """
        if(clear_cache):
            try:
                from IPython.core.getipython import get_ipython
                ip = get_ipython()
                if ip and ip.history_manager is not None:
                    ip.history_manager.output_cache.clear()
                    user_ns = cast(
                        dict[str, Any], 
                        ip.user_ns # pyright: ignore[reportUnknownMemberType]
                    ) 
                    user_ns.pop('_', None)
            except Exception:
                pass
        gc.collect()
    
    def __release_os_memory_windows(self) -> None:
        """
        Request the Windows OS to trim the working set of the current 
        process, releasing unused physical memory back to the system.
        """
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
    
        GetCurrentProcess = kernel32.GetCurrentProcess
        GetCurrentProcess.restype = wintypes.HANDLE
    
        EmptyWorkingSet = psapi.EmptyWorkingSet
        EmptyWorkingSet.argtypes = [wintypes.HANDLE]
        EmptyWorkingSet.restype  = wintypes.BOOL
    
        hProcess = GetCurrentProcess()
        if not EmptyWorkingSet(hProcess):
            raise ctypes.WinError(ctypes.get_last_error())

    def __clean_up_memory(self, passes: int = 2) -> None:
        """
        Clean up memory logging pre and post cleaning memory usages if
        verbosity threshold is met.
        """
        self.__get_memory_usage(verbosity=2)
        for _ in range(passes):
            self.__collect_garbage_ipython()
            if sys.platform.startswith("win"):
                self.__release_os_memory_windows()
        self.__get_memory_usage(verbosity=1)
    
    def estimate_in_memory_size(self, lazy_frame: pl.LazyFrame, 
                                  unit: SizeUnit = "mb") -> float:
        """
        Return estimated size, in memory, of data based on estimated 
        size of single row.
        """
        # Sample size used for getting average row memory size
        n = 100_000
        # Get estimated size for a single row
        n_est = lazy_frame.limit(n).collect().estimated_size(unit)
        self.__clean_up_memory()
        single_est = n_est / n
        # Multiply the single row estimate by the total lazyframe size
        n_rows = (
            lazy_frame
            .select( # pyright: ignore[reportUnknownMemberType]
                pl.len()
            )
            .collect()
            .item()
        )
        total_est = single_est * n_rows
        
        return total_est

    def get_available_memory(self, unit: SizeUnit = "mb") -> float:
        """Return current available memory."""
        mem = psutil.virtual_memory()
        available = self.__unit_mapping(mem.available, unit)
        self.logger(f"Available memory: {available:.2f} {unit.upper()}")

        return available

    def memory_check(self, lazy_frame: pl.LazyFrame, 
                     unit: SizeUnit = "mb") -> Tuple[bool, float, float]:
        """
        Return True if available memory is greater than estimated in
        memory dataframe size.
        """
        available = self.get_available_memory(unit)
        size = self.estimate_in_memory_size(lazy_frame, unit)
        return available > size, available, size