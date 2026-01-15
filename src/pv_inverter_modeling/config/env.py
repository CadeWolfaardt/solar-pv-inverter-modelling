# stdlib
import os
import json
from pathlib import Path
from typing import Callable, TypeVar, Dict, cast, Iterable, Tuple
# thirdpartylib
from dotenv import load_dotenv
# projectlib
from pv_inverter_modeling.utils.typing import InterpMethod
from pv_inverter_modeling.data.schemas import Metric

# Key-value types for JSON dict loading
K = TypeVar("K")
V = TypeVar("V")

def fetch_var(name: str) -> str:
    """Fetch a required environment variable or fail loudly."""
    try:
        value = os.environ[name].strip()
        if not value:
            raise RuntimeError(
                f"Environment variable '{name}' is empty."
            )
        return value
    except KeyError as e:
        raise RuntimeError(
            f"Environment variable '{name}' is not set. "
            "Create a .env file or define the variable."
        ) from e

def fetch_json_map(
        name: str, 
        key_cast: Callable[[str], K],
        value_cast: Callable[[Iterable[object]], V],
    ) -> Dict[K, V]:
    """Fetch daylight map stored as environment variable."""
    raw = fetch_var(name)
    try:
        parsed_any = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Environment variable '{name}' does not contain valid JSON."
        ) from e

    if not isinstance(parsed_any, dict):
        raise RuntimeError(
            f"Environment variable '{name}' must contain a JSON object."
        )
    parsed = cast(dict[str, object], parsed_any)
    return {
        key_cast(k): value_cast(v) # pyright: ignore[reportArgumentType]
        for k, v in parsed.items()
    }

# Load env variables
load_dotenv()

RAW_DATA_ROOT = Path(fetch_var("RAW_DATA_ROOT"))
DATA_ROOT = Path(fetch_var("DATA_ROOT"))
ANOMALY_DATA = Path(fetch_var("ANOMALY_DATA"))
ANOMALY_P_RATED = float(fetch_var("ANOMALY_P_RATED"))
ANOMALY_SAMPLE_DEVICE = fetch_var("ANOMALY_SAMPLE_DEVICE")
PDM_SAMPLE_DEVICE = fetch_var("PDM_SAMPLE_DEVICE")
SA_SAMPLE_DEVICE = fetch_var("SA_SAMPLE_DEVICE")
SITE_NAME = fetch_var("SITE_NAME")
LAT = float(fetch_var("LAT"))
LON = float(fetch_var("LON"))
SITE_TZ = fetch_var("SITE_TZ")
COUNTRY = fetch_var("COUNTRY")
DEVICE_RATING = float(fetch_var("DEVICE_RATING"))
DAYLIGHT_MAP = cast(
    Dict[int, Tuple[int, int]],
    fetch_json_map("DAYLIGHT_MAP_JSON", int, lambda v: tuple(v))
)
FULL_INTERPOLATION_MAP: Dict[Metric, InterpMethod] = cast(
    Dict[Metric, InterpMethod], 
    fetch_json_map("FULL_INTERP_MAP_JSON", str, str)
)
BASE_INTERPOLATION_MAP: Dict[Metric, InterpMethod] = cast(
    Dict[Metric, InterpMethod], 
    fetch_json_map("BASE_INTERP_MAP_JSON", str, str)
)