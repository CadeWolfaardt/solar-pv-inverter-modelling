from enum import StrEnum

class Column(StrEnum):
    """
    Public column identifiers defining dataset structure only.
    
    Note:
        These enums define schema-level identifiers only.
        No site-specific, temporal, or operational values are included.
    """
    DEVICE = 'device'
    TYPE = 'type' 
    SITE = 'site'
    CAPACITY = 'capacity' 
    TIMESTAMP = 'timestamp'
    METRIC = 'metric' 
    VALUE = 'value'

class Metric(StrEnum):
    """
    Public metric identifiers.

    Note:
        These enums define schema-level identifiers only.
        No metric values, units, timestamps, or site-specific
        information are exposed in the public repository.
    """
    AC_POWER = 'ac_power'
    AC_POWER_A = 'ac_power_a'
    AC_POWER_B = 'ac_power_b'
    AC_POWER_C = 'ac_power_c'
    AC_CURRENT = 'ac_current'
    AC_CURRENT_A =  'ac_current_a'
    AC_CURRENT_B =  'ac_current_b'
    AC_CURRENT_C =  'ac_current_c'
    AC_CURRENT_MAX = 'ac_current_max'
    AC_VOLTAGE = 'ac_voltage'
    AC_LINE_AB = 'line_ab'
    AC_LINE_BC = 'line_bc'
    AC_LINE_CA = 'line_ca'
    FREQUENCY = 'frequency' 
    POWER_FACTOR = 'power_factor'
    SVA = 'sva' 
    VAR = 'var'
    ACTIVE_LIMIT = 'active_limit'
    SVA_LIMIT = 'sva_limit'
    VAR_LIMIT = 'var_limit'
    DC_CURRENT = 'dc_current'
    DC_CURRENT_MAX = 'dc_current_max'
    DC_POWER = 'dc_power'
    DC_VOLTAGE = 'dc_voltage'
    DC_BUS_VOLTAGE = 'dc_bus'
    DC_BATT_BUS_VOLTAGE = 'dc_batt_bus'
    ADMISSION_TEMP = 'admission_temp'
    IGBT_TEMP = 'igbt_temp'
    INTERNAL_TEMP = 'internal_temp'
    AMB_TEMP = 'ambient_temp'
    PV_TEMP = 'pv_cell_temp'
    ENERGY_DELIVERED = 'energy_delivered'
    ENERGY_DELIVERED_DAILY = 'energy_delivered_daily'
    ENERGY_RECEIVED = 'energy_recieved'
    DAILY_REACTIVE_POWER = 'daily_reactive_power'
    MONTHLY_REACTIVE_POWER = 'monthly_reactive_power'
    GHI_IRRADIANCE = 'ghi_irradiance'
    POA_IRRADIANCE = 'poa_irradiance'
    REAR_IRRADIANCE = 'rear_irradiance'
    HUMIDITY = 'humidity'
    WIND_SPEED = 'wind_speed'
    WIND_DIRECTION = 'wind_direction'
    TRACKER_ANGLE = 'tracker_angle'
    TRACKER_ANGLE_SETPOINT = 'tracker_angle_setpoint'
    POA_MEDIAN = 'poa_median'
    MEAN_POWER = 'mean_ac_power'
    EFFICIENCY = 'efficiency'
    CONSTRAINED_REGIME = 'constrained_regime'
    NORMAL_REGIME = 'normal_regime'
    OTHER_REGIME = 'other_regime'

# Keys for sorting data
KEYS = (Column.DEVICE.value, Column.TIMESTAMP.value)