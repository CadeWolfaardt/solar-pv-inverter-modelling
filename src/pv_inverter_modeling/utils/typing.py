# stdlib
from typing import (
    Literal,
    List,
    Union, 
    Tuple, 
    TypeAliasType,
    TypedDict,
    get_args, 
    overload,
)
from pathlib import Path
# thirdpartylib
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame as PolarsLazyFrame
from pandas import DataFrame as PandasDataFrame
# projectlib
from pv_inverter_modeling.data.schemas import Column, Metric

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
# Mode for loss preprocessing.interpolation loss injection 
type LossMode = Literal[
    "random", 
    "blocks", 
    "markov", 
    "tod", 
    "periodic", 
    "stuck"
]
# Keyword arguments typing for loss injection internal methods
class LossKwargs(TypedDict, total=False):
    p: float
    n_blocks: int
    mean_len: float
    p_miss: float
    p_recover: float
    start_state: str
    p_by_hour: List[float]
    k: int
    jitter: int
    n_segments: int
# Keyword arguments for losss injection external call
class InjectKwargs(LossKwargs, total=False):
    mode: LossMode
# Available interpolation methods 
type InterpMethod = Literal[
    "linear", 
    "time", 
    "index", 
    "values", 
    "nearest",
    "zero", 
    "slinear", 
    "quadratic", 
    "cubic", 
    "polynomial", 
    "spline",
    "pchip", 
    "akima", 
    "cubicspline", 
    "from_derivatives",
]
type InterpMethods = Tuple[InterpMethod, ...]
# Possible columns
type Field = Union[Column, Metric]
# DataFrame for plotting
type DataFrame = Union[PolarsLazyFrame, PolarsDataFrame, PandasDataFrame]
# Aggregation functions for pvioting table
type Aggregation = Literal[
    'min',
    'max',
    'first',
    'last',
    'sum',
    'mean',
    'median',
    'len',
]
# Engine for collecting polars lazyframe to dataframe
type CollectEngine = Literal[
    'auto',
    'in-memory',
    'streaming',
    'gpu'
]
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