from enum import Enum

class Column(str, Enum):
    DEVICE = 'device'
    TYPE = 'type' 
    SITE = 'site'
    CAPACITY = 'capacity' 
    TIMESTAMP = 'timestamp'
    METRIC = 'metric' 
    VALUE = 'value'

class Metric(str, Enum):
    AC_POWER = 'ac_power'
    AC_PHASE_A =  'phase_a'
    AC_PHASE_B =  'phase_a'
    AC_PHASE_C =  'phase_a'
    AC_LINE_AB = 'line_ab'
    AC_LINE_BC = 'line_bc'
    AC_LINE_CA = 'line_ca'
    FREQUENCY = 'frequency' 
    POWER_FACTOR = 'power_factor'
    SVA = 'sva' 
    VAR = 'var'