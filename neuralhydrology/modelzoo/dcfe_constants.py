INITIAL_STATES = {
    "gw_reservoir_storage_m": 0.5,
    "soil_reservoir_storage_m": 0.6,
    "first_nash_storage": 0.0,
}  # There are more storage/fluxes but doesn't matter cuz we can just grab whatever I want


PARAMETER_RANGES = {
    "satdk": [0.0, 0.000726],  # Saturated hydraulic conductivity [m/hr]
    "Cgw": [0.0000018, 0.0018],  # Primary groundwater reservoir constant [m/hr]
    "bb": [0, 21.94],  # exponent on Clapp-Hornberg functin [-]
    "smcmax": [0.20554, 1],  # Max soil moisture content [m3/hr3]
    "slop": [0, 1],  # slope coefficient [-]
    "max_gw_storage": [0.01, 0.25],  # [m]
    "expon": [1, 8],  # A primary groundwater nonlinear reservoir exponential constant [-]
    "K_lf": [0, 1],  # Lateral flow coefficient
    "K_nash": [0, 1],  # Nash cascade discharge coefficient
    "satpsi": [0.05, 0.95],
}
