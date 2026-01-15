# stdlib
from pathlib import Path
from datetime import datetime
# projectlib
from pv_inverter_modeling.utils.typing import Address, OpenMode

def validate_address(
    address: Address,
    *, 
    extension: str = ".parquet", 
    mode: OpenMode = 'r',
    mkdir: bool = False,
) -> Path:
    """
    Validate and normalize a file or directory path.

    This utility converts the input to a ``pathlib.Path``, optionally
    creates directories, enforces a file extension, and validates
    existence based on the intended I/O mode.

    Parameters
    ----------
    address : Address
        File or directory path as a string or ``Path``.
    extension : str, default ".parquet"
        Expected file extension. If the path refers to a file and the
        extension does not match, it will be replaced.
    mode : OpenMode, default "r"
        Intended file access mode:
        - ``"r"``: path must exist if it refers to a file
        - ``"w"``: existing files will be renamed to avoid overwrite
    mkdir : bool, default False
        If True and the path refers to a directory, create it (including
        parents) if it does not already exist.

    Returns
    -------
    pathlib.Path
        Validated and normalized path.

    Raises
    ------
    NotADirectoryError
        If the parent directory does not exist.
    FileNotFoundError
        If ``mode="r"`` and the file does not exist.
    """
    # Normalize to Path
    if isinstance(address, str):
        address = Path(address)
    # Optionally create directory paths
    if mkdir:
        address.mkdir(parents=True, exist_ok=True)
    # If the path is an existing directory, return immediately
    if address.is_dir():
        return address
    # Validate parent directory for file paths
    if not address.parent.is_dir():
        msg = (
            f"Address path {address.parent}"
            " does not exist or is not a directory."
        )
        raise NotADirectoryError(msg)
    # Enforce file extension
    if address.suffix != extension:
        address = address.with_suffix(extension)
    # Reading requires the file to exist
    if mode == "r" and not address.is_file():
        msg = f"{address} is not a file or does not exist."
        raise FileNotFoundError(msg)
    # Writing: avoid overwriting existing files
    if mode == "w" and address.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        address = address.with_name(
            f"{address.stem}_{timestamp}{address.suffix}"
        )
    
    return address