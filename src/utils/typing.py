# stdlib
from typing import (
    Literal, 
    Union, 
    Tuple, 
    TypeAliasType, 
    get_args, 
    overload
)
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
# Options for data reading: full = entire feature set, baseline = sample 
# feature set
type DataMode = Literal["baseline", "full"]
# Types of type aliases
type TypeValues = Union[
    TypeAliasType, 
    Tuple[TypeAliasType, ...], 
    Tuple[str, ...]
]
# Type for either infering empirically or using a defined value
type InferedFloat = Union[Literal["infer"], float]

# Overloads for custom_get_args function
@overload
def custom_get_args(arg: TypeAliasType) -> tuple[str, ...]: ...
@overload
def custom_get_args(arg: tuple[TypeAliasType, ...]) -> tuple[str, ...]: ...
@overload
def custom_get_args(arg: tuple[str, ...]) -> tuple[str, ...]: ...

def custom_get_args(arg: TypeValues) -> TypeValues:
    """Obtain elements of defined type aliases."""
    # If a single alias object, unwrap and recurse
    if isinstance(arg, TypeAliasType):
        return custom_get_args(get_args(arg.__value__))
    # If already Tuple[str, ...] then return it
    if all(isinstance(item, str) for item in arg):
        return arg
    # If mixed tuple or Tuple[TypeAliasType, ...] then step through and 
    # concatenate tuples
    out = ()
    for t in arg:
        if isinstance(t, str):
            addend = tuple(t)
        else:
            addend = custom_get_args(t)
        out += addend
    return out