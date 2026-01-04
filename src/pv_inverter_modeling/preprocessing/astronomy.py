# stdlib
from zoneinfo import ZoneInfo
from datetime import timedelta
# thridpartylib
import pandas as pd
from astral import LocationInfo
from astral.sun import sun

def build_sun_table(
        d0: pd.Timestamp,
        d1: pd.Timestamp,
        site: LocationInfo,
        tz_local: ZoneInfo,
    ) -> pd.DataFrame:
    """
    Construct a daily sunrise and sunset table for a given location and 
    period.

    This function computes local sunrise and sunset times for each 
    calendar day between ``d0`` and ``d1`` (inclusive) using the 
    provided geographic location and time zone. The resulting times are 
    returned as naive datetimes representing local time.

    Parameters
    ----------
    d0 : pandas.Timestamp
        Start date of the period (inclusive).
    d1 : pandas.Timestamp
        End date of the period (inclusive).
    site : LocationInfo
        Astral location information describing the observation site.
    tz_local : zoneinfo.ZoneInfo
        Local time zone used for sunrise and sunset calculations.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per day containing the following columns:

        - ``date_local``: Local calendar date
        - ``sunrise_local``: Local sunrise time (naive datetime)
        - ``sunset_local``: Local sunset time (naive datetime)
    """
    rows = []

    # Convert timestamps to local calendar dates
    cur = pd.to_datetime(d0).date()
    end = pd.to_datetime(d1).date()

    # Compute sunrise and sunset for each day in the range
    while cur <= end:
        # Calculate solar events for the observer on the given date
        s = sun(site.observer, date=cur, tzinfo=tz_local)

        rows.append({  # pyright: ignore[reportUnknownMemberType]
            "date_local": cur,
            # Store sunrise and sunset as naive local datetimes
            "sunrise_local": s["sunrise"].replace(tzinfo=None),
            "sunset_local":  s["sunset"].replace(tzinfo=None),
        })

        cur += timedelta(days=1)

    # Assemble results into a DataFrame
    out = pd.DataFrame(rows)

    # Ensure datetime dtype for downstream time-based operations
    out["sunrise_local"] = pd.to_datetime(out["sunrise_local"])
    out["sunset_local"]  = pd.to_datetime(out["sunset_local"])

    return out
