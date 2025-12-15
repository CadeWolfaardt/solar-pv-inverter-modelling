# stdlib
from typing import Literal, Union
from pathlib import Path

# Verbosity for classes, functions, methods, etc.
type Verbosity = Literal[0, 1, 2]
# Mode for opening documents
type ReadMode = Literal["r"]
type WriteMode = Literal["w", "x"]
type OpenMode = Literal[ReadMode, WriteMode]
# Type alias for file/folder paths
type Address = Union[str, Path]
# Units for measuring digital information
type SizeUnit = Literal["b", "kb", "mb", "gb", "tb"]