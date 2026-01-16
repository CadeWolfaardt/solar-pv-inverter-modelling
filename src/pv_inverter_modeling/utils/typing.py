# stdlib
from typing import (
    Literal,
    List,
    Union, 
    Tuple, 
    TypeAliasType,
    TypedDict,
    Callable,
    Dict,
    Sequence,
    Optional,
    TypeVar,
    TypeAlias,
    Any,
    NotRequired,
    get_args, 
    overload,
)
from pathlib import Path
# thirdpartylib
from numpy import ndarray
from polars.lazyframe.frame import LazyFrame as PolarsLazyFrame
from polars.dataframe.frame import DataFrame as PolarsDataFrame
from polars import Series as plSeries
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas import Series as pdSeries
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
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
# Aggregation functions for pivoting table
type GenericAggregation = Literal[
    'min',
    'max',
    'sum',
    'mean',
    'median',
]
type PlAggregation = Literal[
    GenericAggregation,
    'first',
    'last',
    'len',
]
type PdAggregation = Literal[
    GenericAggregation,
    'count',
    'std',
    'var',
]
type Aggregation = Union[PdAggregation, PlAggregation]
# Engine for collecting polars lazyframe to dataframe
type CollectEngine = Literal[
    'auto',
    'in-memory',
    'streaming',
    'gpu'
]
# Definition for evaluation.metrics
type ArrayLike1D = Union[
    Sequence[float],
    ndarray,
    pdSeries,
    plSeries,
]
# Forecast training function return type
type ForecastMetrics = Dict[str, float]
ModelT = TypeVar("ModelT")
ForecastTrainingResult: TypeAlias = tuple[
    Optional[ForecastMetrics],
    Optional[ModelT],
]
type BaselineForecastResult = tuple[
    ForecastMetrics,
    LinearRegression,
    XGBRegressor,
]
type ModelRunResult = Union[
    ForecastTrainingResult[Any],
    BaselineForecastResult,
]
# Types for Forecast training function registry
type TrainFn = Callable[..., ModelRunResult]
type ArgsFn = Callable[
    [PandasDataFrame, pdSeries, Path],
    tuple[Any, ...],
]
BaselineSaveFn: TypeAlias = Callable[
    [BaselineForecastResult], 
    Any
]
ForecastSaveFn: TypeAlias = Callable[
    [ForecastTrainingResult[Any]], 
    Any
]
SaveFn: TypeAlias = Union[BaselineSaveFn, ForecastSaveFn]

class BaselineRegistryEntry(TypedDict):
    kind: Literal["baseline"]
    fn: Callable[..., BaselineForecastResult]
    args: ArgsFn
    save: List[Tuple[str, BaselineSaveFn]]

class ForecastRegistryEntry(TypedDict):
    kind: Literal["forecast"]
    fn: Callable[..., ForecastTrainingResult[Any]]
    args: ArgsFn
    save: List[Tuple[str, ForecastSaveFn]]
    skip_if_none: NotRequired[bool]

ModelRegistryEntry = Union[
    BaselineRegistryEntry,
    ForecastRegistryEntry,
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