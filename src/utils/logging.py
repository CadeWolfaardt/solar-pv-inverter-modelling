# stdlib
from pathlib import Path
# projectlib
from utils.util import validate_address
from utils.typing import Verbosity, Address

class Logger(object):
    """
    Simple class for logging; capable of either printing to terminal or
    logging to a .txt file
    """
    def __init__(self, verbose: Verbosity = 0, 
                 address: Address = Path.cwd(),
                 write_log: bool = False) -> None:
        self.verbose = verbose
        self.address = validate_address(address) / "log.txt"
        self.write_log = write_log

    def __call__(self, msg: str, verbosity: int = 0) -> None:
        if self.verbose >= verbosity:
            self.write(msg) if self.write_log else print(msg)
    
    def write(self, msg: str) -> None:
        with open(self.address, "a") as file:
            file.write(msg)