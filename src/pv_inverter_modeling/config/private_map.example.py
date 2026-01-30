"""
Example private-to-public mapping configuration.

This file defines how raw (private, site- or vendor-specific) column
and metric names are mapped to the project's public schema.

INSTRUCTIONS
------------
1. Copy this file to `private_map.py`
2. Replace placeholder keys with real raw names from your data source
3. Do NOT commit the real `private_map.py`

This example contains NO real site-, vendor-, or OEM-specific 
identifiers.
"""

# stdlib
from typing import Dict
# projectlib
from pv_inverter_modeling.data.schemas import Column, Metric

# ---------------------------------------------------------------------
# Raw column name to public schema mapping
# ---------------------------------------------------------------------
RAW_COLUMN_MAP = {
    # Example raw identifiers (placeholders)
    "raw_device_id": Column.DEVICE.value,
    "raw_device_type": Column.TYPE.value,
    "raw_site_id": Column.SITE.value,
    "raw_site_capacity": Column.CAPACITY.value,
    "raw_timestamp": Column.TIMESTAMP.value,
    "raw_metric_name": Column.METRIC.value,
    "raw_metric_value": Column.VALUE.value,
}

# ---------------------------------------------------------------------
# Raw metric name to public metric mapping
# ---------------------------------------------------------------------
RAW_METRIC_MAP = {
    # Electrical power (examples)
    "RAW_METRIC_POWER_TOTAL": Metric.AC_POWER.value,
    "RAW_METRIC_POWER_PHASE_A": Metric.AC_POWER_A.value,
    "RAW_METRIC_POWER_PHASE_B": Metric.AC_POWER_B.value,
    "RAW_METRIC_POWER_PHASE_C": Metric.AC_POWER_C.value,
    # Currents
    "RAW_METRIC_CURRENT_TOTAL": Metric.AC_CURRENT.value,
    "RAW_METRIC_CURRENT_MAX": Metric.AC_CURRENT_MAX.value,
    "RAW_METRIC_CURRENT_PHASE_A": Metric.AC_CURRENT_A.value,
    "RAW_METRIC_CURRENT_PHASE_B": Metric.AC_CURRENT_B.value,
    "RAW_METRIC_CURRENT_PHASE_C": Metric.AC_CURRENT_C.value,
    # Voltages
    "RAW_METRIC_VOLTAGE_TOTAL": Metric.AC_VOLTAGE.value,
    "RAW_METRIC_VOLTAGE_AB": Metric.AC_LINE_AB.value,
    "RAW_METRIC_VOLTAGE_BC": Metric.AC_LINE_BC.value,
    "RAW_METRIC_VOLTAGE_CA": Metric.AC_LINE_CA.value,
    # Frequency and power quality
    "RAW_METRIC_FREQUENCY": Metric.FREQUENCY.value,
    "RAW_METRIC_POWER_FACTOR": Metric.POWER_FACTOR.value,
    # Apparent / reactive power
    "RAW_METRIC_SVA": Metric.SVA.value,
    "RAW_METRIC_VAR": Metric.VAR.value,
    "RAW_METRIC_ACTIVE_LIMIT": Metric.ACTIVE_LIMIT.value,
    "RAW_METRIC_SVA_LIMIT": Metric.SVA_LIMIT.value,
    "RAW_METRIC_VAR_LIMIT": Metric.VAR_LIMIT.value,
    # DC-side metrics
    "RAW_METRIC_DC_CURRENT": Metric.DC_CURRENT.value,
    "RAW_METRIC_DC_CURRENT_MAX": Metric.DC_CURRENT_MAX.value,
    "RAW_METRIC_DC_POWER": Metric.DC_POWER.value,
    "RAW_METRIC_DC_VOLTAGE": Metric.DC_VOLTAGE.value,
    "RAW_METRIC_DC_BUS_VOLTAGE": Metric.DC_BUS_VOLTAGE.value,
    "RAW_METRIC_DC_BATTERY_BUS": Metric.DC_BATT_BUS_VOLTAGE.value,
    # Temperatures
    "RAW_METRIC_TEMP_ADMISSION": Metric.ADMISSION_TEMP.value,
    "RAW_METRIC_TEMP_IGBT": Metric.IGBT_TEMP.value,
    "RAW_METRIC_TEMP_INTERNAL": Metric.INTERNAL_TEMP.value,
    "RAW_METRIC_TEMP_AMBIENT": Metric.AMB_TEMP.value,
    "RAW_METRIC_TEMP_PV": Metric.PV_TEMP.value,
    # Energy counters
    "RAW_METRIC_ENERGY_DELIVERED": Metric.ENERGY_DELIVERED.value,
    "RAW_METRIC_ENERGY_DAILY": Metric.ENERGY_DELIVERED_DAILY.value,
    "RAW_METRIC_ENERGY_RECEIVED": Metric.ENERGY_RECEIVED.value,
    # Reactive energy
    "RAW_METRIC_REACTIVE_DAILY": Metric.DAILY_REACTIVE_POWER.value,
    "RAW_METRIC_REACTIVE_MONTHLY": Metric.MONTHLY_REACTIVE_POWER.value,
    # Environmental / tracker data
    "RAW_METRIC_IRRADIANCE_GHI": Metric.GHI_IRRADIANCE.value,
    "RAW_METRIC_IRRADIANCE_POA": Metric.POA_IRRADIANCE.value,
    "RAW_METRIC_IRRADIANCE_REAR": Metric.REAR_IRRADIANCE.value,
    "RAW_METRIC_HUMIDITY": Metric.HUMIDITY.value,
    "RAW_METRIC_WIND_SPEED": Metric.WIND_SPEED.value,
    "RAW_METRIC_WIND_DIRECTION": Metric.WIND_DIRECTION.value,
    "RAW_METRIC_TRACKER_ANGLE": Metric.TRACKER_ANGLE.value,
    "RAW_METRIC_TRACKER_SETPOINT": Metric.TRACKER_ANGLE_SETPOINT.value,
    # Derived / aggregated metrics
    "RAW_METRIC_POA_MEDIAN": Metric.POA_MEDIAN.value,
    "RAW_METRIC_MEAN_POWER": Metric.MEAN_POWER.value,
}

# ---------------------------------------------------------------------
# Derived helper mappings
# ---------------------------------------------------------------------
# Combined raw to public mapping
COMPLETE_MAP = RAW_COLUMN_MAP | RAW_METRIC_MAP
# Ensure Dict[str, str] for Polars rename operations
ENTITY_MAP: Dict[str, str] = {
    key: str(value) for key, value in COMPLETE_MAP.items()
}
# Reverse lookup (public to raw)
REVERSE_ENTITY_MAP = {
    value: key for key, value in ENTITY_MAP.items()
}
# Lists of valid identifiers
PUBLIC_NAMES = (
    list(RAW_COLUMN_MAP.values())
    + list(RAW_METRIC_MAP.values())
)
PRIVATE_NAMES = (
    list(RAW_COLUMN_MAP.keys())
    + list(RAW_METRIC_MAP.keys())
)
