# stdlib
import os
from pathlib import Path
# thirdpartylib
from dotenv import load_dotenv

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


# Load env variables
load_dotenv()

RAW_DATA_ROOT = Path(fetch_var("RAW_DATA_ROOT"))
DATA_ROOT = Path(fetch_var("DATA_ROOT"))
ANOMALY_DATA = Path(fetch_var("ANOMALY_DATA"))
ANOMALY_P_RATED = float(fetch_var("ANOMALY_P_RATED"))
ANOMALY_SAMPLE_DEVICE = fetch_var("ANOMALY_SAMPLE_DEVICE")
