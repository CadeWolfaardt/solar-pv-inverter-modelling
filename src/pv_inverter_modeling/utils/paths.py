# stdlib
from pathlib import Path
from datetime import datetime
# projectlib
from pv_inverter_modeling.utils.typing import Address, OpenMode

def validate_address(address: Address, extension: str = ".parquet", 
                     mode: OpenMode = 'r') -> Path:
        """
        Validate a file/folder path and return the pathlib Path of the 
        path.
        """
        if isinstance(address, str):
            address = Path(address)
        # Return Path obj path points to a directory; evaluates False if
        # the path is invalid, inaccessible or missing, or if it points
        # to something other than a directory.
        if address.is_dir():
            return address
        # Validate parent directory
        if not address.parent.is_dir():
            pt = address.parent
            msg = f"Address path {pt} does not exist or is not a directory."
            raise NotADirectoryError(msg)
        # Validate file extension
        if not address.suffix == extension:
            address = address.with_suffix(extension)
        # Confirm source exists if it is a file and is meant to be read
        if mode == 'r' and not address.is_file():
            msg = f"{address} is not a file or does not exist."
            raise FileNotFoundError(msg)
        # Change file name if already exists and mode is write
        if address.exists() and mode == "w":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            address = address.with_name(
                f"{address.stem}_{timestamp}{address.suffix}"
            )
        
        return address