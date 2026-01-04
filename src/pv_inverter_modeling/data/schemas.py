from enum import Enum

class Column(str, Enum):
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

class Metric(str, Enum):
    """
    Public metric identifiers.

    Note:
        These enums define schema-level identifiers only.
        No metric values, units, timestamps, or site-specific
        information are exposed in the public repository.
    """
    AC_POWER = 'ac_power'
    AC_CURRENT_A =  'ac_current_a'
    AC_CURRENT_B =  'ac_current_b'
    AC_CURRENT_C =  'ac_current_c'
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
    DC_POWER = 'dc_power'
    DC_VOLTAGE = 'dc_voltage'
    DC_BUS_VOLTAGE = 'dc_bus'
    ADMISSION_TEMP = 'admission_temp'
    IGBT_TEMP = 'igbt_temp'
    INTERNAL_TEMP = 'internal_temp'
    POA_MEDIAN = 'poa_median'
    EFFICIENCY = 'efficiency'
    MEAN_POWER = 'mean_ac_power'

# Keys for sorting data
KEYS = (Column.DEVICE, Column.TIMESTAMP)