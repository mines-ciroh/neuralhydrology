# original shm packages
from typing import Dict, Union

import torch

from neuralhydrology.utils.DCFE_utils import physics_constants
from neuralhydrology.utils.config import Config

# packages from cfe.py


# this happens before the run
def initialize_basin_constants(
    cfg: Config,
    conceptual_forcing: torch.Tensor,
    cfe_params: Dict[str, torch.Tensor],
    hourly: bool,
) -> Dict[str, Union[int, float, Dict, torch.Tensor]]:
    """Module to initialize the basin constants and parameters for the CFE model before run.
    Args:
        conceptual_forcing (torch.Tensor): Tensor of size [batch_size, time_steps, n_inputs],
        cfe_params (Dict[str, torch.Tensor]): must contain basinCharacteristics, soil_params
        hourly (bool): TRUE if the time step is hourly, FALSE if daily

    Raises:
        ValueError: hourly must be True or False

    Returns:
        Dict[str, Union[int, float, Dict, torch.Tensor]]:
            constants: time and physics constants
            gw_reservoir: groundwater reservoir parameters/status
            soil_reservoir: soil reservoir parameters/status
            routing_info: routing parameters/status
            flux: flux parameters/status, only one in this function but will be updated later
    """

    device = conceptual_forcing.device
    batch_size = conceptual_forcing.shape[0]

    # time-related constants
    if not isinstance(hourly, bool):
        raise ValueError(f"'hourly' must be True or False, got {type(hourly)}")

    time = {
        "step_size": 3600 if hourly else 3600 * 24,  # num of [seconds]
        "hrs": (3600 if hourly else 3600 * 24) / 3600,  # num of [hours]
        "days": ((3600 if hourly else 3600 * 24) / 3600) / 24,  # time step in [days]
    }
    
    scheme = {
        "soil": cfg.dcfe_soil_scheme,  # choose between 'classic' or 'ode', 'ode' not available rn
        "partition": cfg.dcfe_partition_scheme,  # choose between 'Schaake' or 'Xinanjiang'
    }

    constants = {"time": time, "physics": physics_constants, "cfe_scheme": scheme}

    gw_reservoir = {
        "storage_max_m": cfe_params["basinCharacteristics"]["max_gw_storage"],
        "exponent_primary": cfe_params["basinCharacteristics"]["expon"],
        "storage_threshold_primary_m": 0,
        # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off.
        "storage_threshold_secondary_m": 0,
        "coeff_secondary": 0,
        "exponent_secondary": 1,
    }

    gw_reservoir["storage_m"] = 0.05 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)

    ## Soil Reservoir Configuration, not used outside this function
    soil_config = soil_reservoir_configuration(conceptual_forcing, cfe_params, constants)

    soil_reservoir = {
        "wilting_point_m": cfe_params["soil_params"]["wltsmc"] * cfe_params["soil_params"]["D"],  # 0.049668*2 = 0.09933
        "storage_max_m": cfe_params["soil_params"]["smcmax"] * cfe_params["soil_params"]["D"],  # 0.373*2 = 0.746
        #'coeff_primary': parameters['satdk'] * soil_params['slop'].unsqueeze(1)*time_step_size, #Eq.11, unit [m/s] * [3600s] now its [m/hr]. Define this in loop
        "exponent_primary": 1.0,  # fixed to 1 based on Eq. 11
        "storage_threshold_primary_m": soil_config["field_capacity_storage_threshold_m"],
        "coeff_secondary": cfe_params["basinCharacteristics"]["K_lf"],  # Controls lateral flow
        "exponent_secondary": 1.0,  # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
        "storage_threshold_secondary_m": soil_config[
            "lateral_flow_threshold_storage_m"
        ],  ## but this is the same as field_capacity_storage_threshold_m??
    }

    soil_reservoir["storage_m"] = 0.05 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)

    # put things used in Nash Cascade & GIUH under routing
    num_ordinates = cfe_params["basinCharacteristics"]["giuh_ordinates"].shape[1]  # Daniel changed this from .shape[0] to .shape[1]

    routing_info = {
        "num_ordinates": num_ordinates,  # giuh_ordinates are rows x 1 column for each basin, used in routing
        "runoff_queue_m_per_timestep": torch.ones((batch_size, num_ordinates + 1), dtype=torch.float32, device=device),  # nash cascade
        "num_reservoirs": cfe_params["basinCharacteristics"]["nash_storage"].shape[1],  # 2 reservoirs
    }

    flux = {"flux_perc_m": torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)}

    # assert torch.all(self.soil_reservoir['storage_m'] >= 0), "Variable went negative, stopping program."

    return constants, cfe_params, gw_reservoir, soil_reservoir, routing_info, flux


def timestep_basin_constants(
    conceptual_forcing_timestep: torch.Tensor,
    gw_reservoir: Dict[str, torch.Tensor],
    soil_reservoir: Dict[str, torch.Tensor],
    cfe_params: Dict[str, torch.Tensor],
    constants: Dict[str, Union[int, float, Dict, torch.Tensor]],
    timestep_params: Dict[str, Union[int, float, Dict, torch.Tensor]],
) -> Dict[str, Union[int, float, Dict, torch.Tensor]]:
    """Module to update the basin constants and parameters for the CFE model at every timestep.
    Args:
        conceptual_forcing (torch.Tensor): Tensor of size [batch_size, time_steps, n_inputs],
        cfe_params (Dict[str, torch.Tensor]): must contain basinCharacteristics, soil_params
        constants (Dict[str, Union[int, float, Dict, torch.Tensor]]): time and physics constants
        timestep_params (Dict[str, Union[int, float, Dict, torch.Tensor]]):
            Contains parameters for the timestep, outputed by LSTM, including:
                - bb_timestep: soil parameter bb
                - satdk_timestep: soil parameter satdk
                - smcmax_timestep: soil parameter smcmax
                - slop_timestep: soil parameter slop
                - satpsi_timestep: soil parameter satpsi
                - cgw_timestep: basin characteristic Cgw
                - max_gw_timestep: basin characteristic max_gw_storage
                - K_nash_timestep: basin characteristic K_nash
                - K_lf_timestep: basin characteristic K_lf
                - expon_timestep: basin characteristic expon

    returns:
        cfe_params (Dict[str, torch.Tensor]): updated cfe_params
        gw_reservoir (Dict[str, torch.Tensor]): updated gw_reservoir
        soil_reservoir (Dict[str, torch.Tensor]): updated soil_reservoir
    """
    # updating them into the cfe_params
    # TODO: Can we refactor the code below into a single function call or list comprehension?
    cfe_params["soil_params"]["bb"] = timestep_params["bb"]
    cfe_params["soil_params"]["satdk"] = timestep_params["satdk"]
    cfe_params["soil_params"]["smcmax"] = timestep_params["smcmax"]
    cfe_params["soil_params"]["slop"] = timestep_params["slop"]
    cfe_params["soil_params"]["satpsi"] = timestep_params["satpsi"]
    cfe_params["basinCharacteristics"]["Cgw"] = timestep_params["Cgw"]
    cfe_params["basinCharacteristics"]["max_gw_storage"] = timestep_params["max_gw_storage"]
    cfe_params["basinCharacteristics"]["K_nash"] = timestep_params["K_nash"]
    cfe_params["basinCharacteristics"]["K_lf"] = timestep_params["K_lf"]
    cfe_params["basinCharacteristics"]["expon"] = timestep_params["expon"]

    # cfe_params updates some reservoir parameters
    gw_reservoir["storage_max_m"] = cfe_params["basinCharacteristics"]["max_gw_storage"]
    gw_reservoir["coeff_primary"] = cfe_params["basinCharacteristics"]["Cgw"]
    gw_reservoir["exponent_primary"] = cfe_params["basinCharacteristics"]["expon"]

    # soil_reservoir parameters gets updated this timestep
    soil_config = soil_reservoir_configuration(conceptual_forcing_timestep, cfe_params, constants)

    # update soil reservoir parameters
    soil_reservoir["storage_max_m"] = cfe_params["soil_params"]["smcmax"] * cfe_params["soil_params"]["D"]
    soil_reservoir["storage_threshold_primary_m"] = soil_config["field_capacity_storage_threshold_m"]
    soil_reservoir["coeff_secondary"] = cfe_params["basinCharacteristics"]["K_lf"]
    soil_reservoir["storage_threshold_secondary_m"] = soil_config["lateral_flow_threshold_storage_m"]
    soil_reservoir["coeff_primary"] = timestep_params["satdk"] * timestep_params["slop"] * constants["time"]["step_size"]
    # Eq.11, unit [m/s] * [3600s] now its [m/hr]

    return cfe_params, gw_reservoir, soil_reservoir


def initialize_flux_timestep(conceptual_forcing_timestep: torch.Tensor, flux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Module to re-set some fluxes for the CFE model at every timestep

    Args:
        conceptual_forcing_timestep (torch.Tensor): Tensor of size [batch_size, n_inputs],
        flux (Dict[str, torch.Tensor]): flux parameters/status

    Returns:
        Dict[str, torch.Tensor]: flux parameters/status, set to 0
            'surface_runoff_depth_m': surface runoff depth in m/timestep
            'infilt_excess_m': infiltration excess in m/timestep
            'infiltration_depth_m': infiltration depth in m/timestep
            'infilt_depth_m': infiltration depth in m/timestep
            'actual_et_from_rain_m_per_timestep': actual ET from rain in m/timestep
            'actual_et_from_soil_m_per_timestep': actual ET from soil in m/timestep
            'actual_et_m_per_timestep': actual ET in m/timestep
            'reduced_potential_et_m_per_timestep': reduced potential ET in m/timestep
            'primary_flux_m': primary flux in m/timestep
            'secondary_flux_m': secondary flux in m/timestep
            'primary_flux_from_gw_m': primary flux from groundwater in m/timestep
            'secondary_flux_from_gw_m': secondary flux from groundwater in m/timestep
            'giuh_runoff_m': GIUH runoff in m/timestep
            'nash_lateral_runoff_m': Nash lateral runoff in m/timestep
            'from_deep_gw_to_chan_m': from deep groundwater to channel in m/timestep
            'tension_water_m': tension water in m/timestep
    """

    device = conceptual_forcing_timestep.device
    batch_size = conceptual_forcing_timestep.shape[0]

    # reset fluxes that can store information at every time-step. This will be #basin x
    flux["surface_runoff_depth_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["infilt_excess_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)

    # infilt excess & surface rnoff depth are the same? I'm using infilt excess to be consistent to cfe
    flux["infiltration_depth_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["infilt_depth_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["actual_et_from_rain_m_per_timestep"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["actual_et_from_soil_m_per_timestep"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["actual_et_m_per_timestep"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)

    # reset ET
    flux["reduced_potential_et_m_per_timestep"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["primary_flux_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["secondary_flux_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)

    # below are all added later, not in original initialization
    flux["primary_flux_from_gw_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["secondary_flux_from_gw_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["giuh_runoff_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["nash_lateral_runoff_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)
    flux["from_deep_gw_to_chan_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)

    # 'Xinanjiang' partition only
    flux["tension_water_m"] = torch.tensor(0.0, dtype=torch.float32, device=device).repeat(batch_size)

    return flux


def get_and_calculate_input_rainfall_and_ET(
    conceptual_forcing_timestep: torch.Tensor,
    flux: Dict[str, torch.Tensor],
    constants: Dict[str, Union[int, float, Dict, torch.Tensor]],
    hourly: bool,
) -> Dict[str, torch.Tensor]:
    """Module to get and calculate the input rainfall and ET for the CFE model at every timestep
    Args:
        conceptual_forcing_timestep (torch.Tensor): Tensor of size [batch_size, n_inputs],
            n_inputs = 3 for hourly:
                [,0] = precipitation [mm/timestep]
                [,1] = mean temperature [C]
                [,2] = shortwave radiation [W/m^2]
            n_inputs = 4 for daily:
                [,0] = precipitation [mm/timestep]
                [,1] = min temperature [C]
                [,2] = max temperature [C]
                [,3] = shortwave radiation [W/m^2]
        flux (Dict[str, torch.Tensor]): flux parameters/status
        constants (Dict[str, Union[int, float, Dict, torch.Tensor]]): time and physics constants
        hourly (bool): TRUE if the time step is hourly, FALSE if daily
        TshirtTest (bool): Defalt is FALSE. If TRUE, skip input-shape validation and use special hourly PET logic.
            This is used for testing purposes only, see detail : CFE output from https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py/blob/main/cat58_test_compare.csv

    Raises:
        ValueError:
        incorrect number of features in conceptual_forcing_timestep for the hourly or daily setting
        Must change this in dynamic_conceptual_inputs in config.

    Returns:
        flux (Dict[str, torch.Tensor]): flux parameters/status, updated with new
            'timestep_rainfall_input_m': rainfall in m/timestep
            'potential_et_m_per_timestep': potential ET in m/timestep
            'reduced_potential_et_m_per_timestep': reduced potential ET in m/timestep
    """
    expected_feats = 3 if hourly else 4
    if conceptual_forcing_timestep.shape[1] != expected_feats:
        raise ValueError(
            f"Expected {expected_feats} features for {'hourly' if hourly else 'daily'} data, "
            f"but got {conceptual_forcing_timestep.shape[1]}."
        )
    # convert and store rainfall mm/timestep to m/timestep
    flux["timestep_rainfall_input_m"] = conceptual_forcing_timestep[:, 0] / 1000.0

    # calculate PET from shortwave rad and mean temp using jensen_evaporation_2016 "https://github.com/pyet-org/pyet/blob/master/pyet/radiation.py"
    if hourly:  # hourly scheme
        mean_temp = conceptual_forcing_timestep[:, 1]
        lambd = 2.501 - 0.002361 * mean_temp  # using mean temp
        shortRad = (
            conceptual_forcing_timestep[:, 2] * constants["time"]["step_size"] / 1000000
        )  # convert shortwave radiation [W/m^2] to [MJ/m^2 hr]
        pet_m_per_timestep_calc = (0.025 * shortRad * (mean_temp - (-3.0)) / lambd) / 1000  # convert pet [mm/hr] to [m/hr]
        pet_m_per_timestep_mask = pet_m_per_timestep_calc < 0  # make mask for negative PET
        flux["potential_et_m_per_timestep"] = torch.where(
            pet_m_per_timestep_mask, 0, pet_m_per_timestep_calc
        )  # clip negative PET to 0
    else:  # daily scheme
        Tmin = conceptual_forcing_timestep[:, 1]
        Tmax = conceptual_forcing_timestep[:, 2]
        mean_temp = (Tmin + Tmax) / 2.0
        lambd = 2.501 - 0.002361 * mean_temp
        shortRad = (
            conceptual_forcing_timestep[:, 3] * constants["time"]["step_size"] / 1000000
        )  # convert shortwave radiation [W/m^2] to [MJ/m^2 day]
        pet_m_per_timestep_calc = (0.025 * shortRad * (mean_temp - (-3.0)) / lambd) / 1000  # convert pet [mm/day] to [m/day]
        pet_m_per_timestep_mask = pet_m_per_timestep_calc < 0  # make mask for negative PET
        flux["potential_et_m_per_timestep"] = torch.where(
            pet_m_per_timestep_mask, 0, pet_m_per_timestep_calc
        )  # clip negative PET to 0

    flux["reduced_potential_et_m_per_timestep"] = flux["potential_et_m_per_timestep"].clone()

    return flux


def calculate_evaporation_from_rainfall(
    flux: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Module to calculate the evaporation from rainfall for the CFE model at every timestep
    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status

    Returns:
        Dict[str, torch.Tensor]: flux parameters/status, updated with new
            'actual_et_from_rain_m_per_timestep': actual ET from rainfall in m/timestep
            'reduced_potential_et_m_per_timestep': reduced potential ET in m/timestep
            'timestep_rainfall_input_m': rainfall in m/timestep
            'actual_et_m_per_timestep': actual ET in m/timestep
    """

    rainfall_mask = flux["timestep_rainfall_input_m"] > 0
    if torch.any(rainfall_mask):  # if there's precip in this batch then
        ## Calculate et_from_rainfall

        # apply the mask for when there's rainfall
        rainfall = flux["timestep_rainfall_input_m"][rainfall_mask]
        pet = flux["potential_et_m_per_timestep"][rainfall_mask]

        # If rainfall exceeds PET, actual AET from rainfall is equal to the PET
        # Otherwise, actual ET equals to potential ET
        # condition is set for each sample within the same time-step
        rain_pet_condition = rainfall > pet
        actual_et_from_rain = torch.where(
            rain_pet_condition,
            pet,  # If P > PET, AET from P is equal to the PET
            rainfall,  # If P < PET, all P gets consumed as AET
        )

        reduced_rainfall = torch.where(
            rain_pet_condition,
            rainfall - actual_et_from_rain,  #  # If P > PET, part of P is consumed as AET
            torch.zeros_like(rainfall),  # If P < PET, all P gets consumed as AET
        )

        # storing results back to the flux
        flux["actual_et_from_rain_m_per_timestep"][rainfall_mask] = actual_et_from_rain
        flux["timestep_rainfall_input_m"][rainfall_mask] = reduced_rainfall  # adjusting precip based on evaporation from rainfall
        flux["reduced_potential_et_m_per_timestep"][rainfall_mask] = (
            pet - actual_et_from_rain
        )  # adjusting pet based on evaporation from rainfall

        # And track_volume_from_rainfall
        flux["actual_et_m_per_timestep"] = (
            flux["actual_et_m_per_timestep"] + flux["actual_et_from_rain_m_per_timestep"]
        )  # actual_et_m_per_timestep not used anywhere

    return flux


def calculate_evaporation_from_soil(
    flux: Dict[str, torch.Tensor],
    constants: Dict[str, Union[int, float, Dict, torch.Tensor]],
    soil_reservoir: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Module to calculate the evaporation from soil for the CFE model at every timestep

    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status
        constants (Dict[str, Union[int, float, Dict, torch.Tensor]]):  time and physics constants
        soil_reservoir (Dict[str, torch.Tensor]): soil reservoir parameters/status

    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'actual_et_from_soil_m_per_timestep': actual ET from soil in m/timestep
            'reduced_potential_et_m_per_timestep': reduced potential ET in m/timestep
            'actual_et_m_per_timestep': actual ET in m/timestep
        soil_reservoir parameters/status, updated with new
            'storage_m': soil storage in m
    """

    # Create mask for soil storage > wilting point, if this is true then soil from et happens
    soil_wilting_mask = soil_reservoir["storage_m"] > soil_reservoir["wilting_point_m"]

    if torch.any(soil_wilting_mask):
        if constants["cfe_scheme"]["soil"] not in ["classic", "ode"]:
            raise NotImplementedError(
                f"Soil scheme '{constants['cfe_scheme']['soil']}' is not implemented. Supported: ['classic', 'ode']."
            )
        elif constants["cfe_scheme"]["soil"] == "classic":
            # Create mask for when reduced PET > 0
            reduced_pet_mask = flux["reduced_potential_et_m_per_timestep"] > 0
            if torch.any(reduced_pet_mask):
                reduced_pet = flux["reduced_potential_et_m_per_timestep"][soil_wilting_mask & reduced_pet_mask]
                storage_threshold_prim = soil_reservoir["storage_threshold_primary_m"][soil_wilting_mask & reduced_pet_mask]
                actual_et_soil = flux["actual_et_from_soil_m_per_timestep"][soil_wilting_mask & reduced_pet_mask]
                soil_storage = soil_reservoir["storage_m"][soil_wilting_mask & reduced_pet_mask]
                wilting_point = soil_reservoir["wilting_point_m"][soil_wilting_mask & reduced_pet_mask]

                storage_threshold_mask = soil_storage >= storage_threshold_prim

                # for if soil_storage >= storage_threshold_prim
                actual_et_soil[storage_threshold_mask] = torch.min(
                    reduced_pet[storage_threshold_mask], soil_storage[storage_threshold_mask]
                )

                # for if soil_storage < storage_threshold_prim
                Budyko_numerator = soil_storage[~storage_threshold_mask] - wilting_point[~storage_threshold_mask]
                Budyko_denominator = storage_threshold_prim[~storage_threshold_mask] - wilting_point[~storage_threshold_mask]
                Budyko = Budyko_numerator / Budyko_denominator
                actual_et_soil[~storage_threshold_mask] = torch.min(
                    Budyko * reduced_pet[~storage_threshold_mask], soil_storage[~storage_threshold_mask]
                )

                # Store back to variabales
                flux["actual_et_from_soil_m_per_timestep"][soil_wilting_mask & reduced_pet_mask] = actual_et_soil

        elif constants["cfe_scheme"]["soil"] == "ode":
            raise NotImplementedError("ODE for soil scheme is not yet implemented. Change this in config's dcfe_soil_scheme:")

        # adjust soil storage w/ ET from soil
        soil_reservoir["storage_m"] = soil_reservoir["storage_m"] - flux["actual_et_from_soil_m_per_timestep"]

        # adjust PET w/ ET from soil
        flux["reduced_potential_et_m_per_timestep"] = (
            flux["reduced_potential_et_m_per_timestep"] - flux["actual_et_from_soil_m_per_timestep"]
        )
        flux["actual_et_m_per_timestep"] = flux["actual_et_m_per_timestep"] + flux["actual_et_from_soil_m_per_timestep"]

    return flux, soil_reservoir


def run_Schaake_subroutine(
    flux: Dict[str, torch.Tensor],
    constants: Dict[str, Union[int, float, Dict, torch.Tensor]],
    cfe_params: Dict[str, torch.Tensor],
    soil_reservoir: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Module to run the Schaake subroutine for the CFE model at every timestep
    This subtroutine takes water_input_depth_m and partitions it into surface_runoff_depth_m and
    infiltration_depth_m using the scheme from Schaake et al. 1996.

    There's no ice process.

    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status
        constants (Dict[str, Union[int, float, Dict, torch.Tensor]]): time and physics constants
        cfe_params (Dict[str, torch.Tensor]): must contain basinCharacteristics, soil_params
        soil_reservoir (Dict[str, torch.Tensor]):  soil reservoir parameters/status

    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'surface_runoff_depth_m': surface runoff depth in m/timestep
            'infiltration_depth_m': infiltration depth in m/timestep
        soil_reservoir parameters/status, updated with new
            'storage_deficit_m': storage deficit in m/timestep
            'Schaake_adjusted_magic_constant_by_soil_type': Schaake adjusted magic constant by soil type
    """

    soil_reservoir["storage_deficit_m"] = (
        cfe_params["soil_params"]["smcmax"] * cfe_params["soil_params"]["D"] - soil_reservoir["storage_m"]
    )
    soil_reservoir["Schaake_adjusted_magic_constant_by_soil_type"] = (
        cfe_params["basinCharacteristics"]["refkdt"] * cfe_params["soil_params"]["satdk"] / 2.0e-06
    )

    rainfall_mask = flux["timestep_rainfall_input_m"] > 0
    soil_noDeficit_mask = soil_reservoir["storage_deficit_m"] < 0  # mark ones w/o deficit
    soil_noDeficit_rain_mask = rainfall_mask & soil_noDeficit_mask
    soil_deficit_rain_mask = rainfall_mask & ~soil_noDeficit_mask

    if torch.any(rainfall_mask):
        # For soil_reservoir_storage_deficit_m < 0, excess = rain and depth = 0
        flux["surface_runoff_depth_m"][soil_noDeficit_rain_mask] = flux["timestep_rainfall_input_m"][soil_noDeficit_rain_mask]
        # Did not put in infiltration_depth_m as they are 0 in this case

        # For soil_reservoir_storage_deficit_m >= 0
        Schaake_parenthetical_term = 1 - torch.exp(
            -soil_reservoir["Schaake_adjusted_magic_constant_by_soil_type"][soil_deficit_rain_mask] * constants["time"]["days"]
        )
        Ic = soil_reservoir["storage_deficit_m"][soil_deficit_rain_mask] * Schaake_parenthetical_term
        Px = flux["timestep_rainfall_input_m"][soil_deficit_rain_mask]
        flux["infiltration_depth_m"][soil_deficit_rain_mask] = Px * (Ic / (Px + Ic))

        # From vector above, make another condition
        soil_excess_mask = (
            flux["timestep_rainfall_input_m"] - flux["infiltration_depth_m"] > 0
        )  # mask for if rainfall is more than infilt depth
        combined_soil_excess_mask = soil_deficit_rain_mask & soil_excess_mask  # mask for above chunk + excess rain
        combined_soil_noExcess_mask = soil_deficit_rain_mask & ~soil_excess_mask  # mask for above chunk + no excess rain
        flux["surface_runoff_depth_m"][combined_soil_excess_mask] = (
            flux["timestep_rainfall_input_m"] - flux["infiltration_depth_m"]
        )[combined_soil_excess_mask]
        # not written else surface_runoff_depth_m = 0, since initialzied at 0
        flux["infiltration_depth_m"][combined_soil_noExcess_mask] = (
            flux["timestep_rainfall_input_m"] - flux["surface_runoff_depth_m"]
        )[combined_soil_noExcess_mask]
        # not writtien, if no rainfall, surface_runoff_depth_m = 0 and infiltration_depth_m = 0 since initialized at 0

    return flux, soil_reservoir


def run_Xinanjiang_subroutine(
    timestep_conceptual_forcing: torch.Tensor, flux: Dict[str, torch.Tensor], soil_reservoir: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Module to run the Xinanjiang subroutine for the CFE model at every timestep, currently not used
    TODO: Verify w/ C-code. All parameteres a, b, x are calibratables, but due to differentiability issues
    we can only choose between Xianjiang and Schaake, and not enable both.

    Args:
        timestep_conceptual_forcing (torch.Tensor): Tensor of size [batch_size, n_inputs], used for device
        flux (Dict[str, torch.Tensor]): flux parameters/status
        soil_reservoir (Dict[str, torch.Tensor]): soil reservoir parameters/status

    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'surface_runoff_depth_m': surface runoff depth in m/timestep
            'infiltration_depth_m': infiltration depth in m/timestep
        soil_reservoir parameters/status, updated with new
            'storage_m': soil storage in m
    """

    device = timestep_conceptual_forcing.device
    batch_size = timestep_conceptual_forcing.shape[0]

    # partition the total soil water in the column between free water and tension water
    free_water_m = soil_reservoir["storage_m"] - soil_reservoir["storage_threshold_primary_m"]  # this storage was adjusted for ET

    # create mask for when free_water_m > 0
    free_water0_mask = free_water_m > 0.0
    free_water_m[~free_water0_mask] = 0.0  # set 0 or negative values to 0
    if torch.any(free_water0_mask):
        tension_water_m = torch.where(
            free_water0_mask, soil_reservoir["storage_threshold_primary_m"], soil_reservoir["storage_m"]
        )

    # estimate the maximum free water and tension water available in the soil column
    max_free_water_m = soil_reservoir["storage_max_m"] - soil_reservoir["storage_threshold_primary_m"]
    max_tension_water_m = soil_reservoir["storage_threshold_primary_m"]

    # Ensuring the variables free_water_m and tension_water_m are not out of bounds
    free_water_mask = max_free_water_m < free_water_m
    tension_water_mask = max_tension_water_m < tension_water_m

    # check that the free_water_m and tension_water_m do not exceed the maximum and if so, change to the max value
    if torch.any(free_water_m):
        free_water_m[free_water_mask] = max_free_water_m[free_water_mask]
    if torch.any(tension_water_mask):
        tension_water_m[tension_water_mask] = max_tension_water_m[tension_water_mask]
    """
    NOTE: the impervious surface runoff assumptions due to frozen soil used in NWM 3.0 have not been included.
    We are assuming an impervious area due to frozen soils equal to 0 (see eq. 309 from Knoben et al).

    The total (pervious) runoff is first estimated before partitioning into surface and subsurface components.
    See Knoben et al eq 310 for total runoff and eqs 313-315 for partitioning between surface and subsurface
    components.

    Calculate total estimated pervious runoff. 
    NOTE: If the impervious surface runoff due to frozen soils is added,
    the pervious_runoff_m equation will need to be adjusted by the fraction of pervious area.
    """

    a_Xinanjiang_inflection_point_parameter = torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
    b_Xinanjiang_shape_parameter = torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
    x_Xinanjiang_shape_parameter = torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)

    tension_frac_mask = (tension_water_m / max_tension_water_m) <= (
        0.5 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size) - a_Xinanjiang_inflection_point_parameter
    )

    pervious_runoff_m = torch.where(
        tension_frac_mask,
        (
            flux["timestep_rainfall_input_m"]
            * torch.pow(
                (
                    0.5 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
                    - a_Xinanjiang_inflection_point_parameter
                ),
                (torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size) - b_Xinanjiang_shape_parameter),
            )
            * torch.pow(
                (
                    torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
                    - (tension_water_m / max_tension_water_m)
                ),
                b_Xinanjiang_shape_parameter,
            )
        ),
        flux["timestep_rainfall_input_m"]
        * (
            torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
            - torch.pow(
                (
                    0.5 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
                    + a_Xinanjiang_inflection_point_parameter
                ),
                (torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size) - b_Xinanjiang_shape_parameter),
            )
            * torch.pow(
                (
                    torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
                    - (tension_water_m / max_tension_water_m)
                ),
                (b_Xinanjiang_shape_parameter),
            )
        ),
    )
    # Separate the surface water from the pervious runoff
    ## NOTE: If impervious runoff is added to this subroutine, impervious runoff should be added to
    ## the surface_runoff_depth_m.

    flux["surface_runoff_depth_m"] = pervious_runoff_m * (
        0.5 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
        - torch.pow(
            (0.5 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size) - (free_water_m / max_free_water_m)),
            x_Xinanjiang_shape_parameter,
        )
    )

    # The surface runoff depth is bounded by a minimum of 0 and a maximum of the water input depth.
    # Check that the estimated surface runoff is not less than 0.0 and if so, change the value to 0.0.
    surface_runoff0_mask = flux["surface_runoff_depth_m"] < 0.0  # we have a vector instead of scalar
    if torch.any(surface_runoff0_mask):
        flux["surface_runoff_depth_m"][surface_runoff0_mask] = torch.zeros(
            (torch.sum(surface_runoff0_mask), 1), dtype=torch.float32, device=device
        )

    # Check that the estimated surface runoff does not exceed the amount of water input to the soil surface.  If it does,
    # change the surface water runoff value to the water input depth.
    surface_runoff_rainfall_mask = flux["surface_runoff_depth_m"] > flux["timestep_rainfall_input_m"]
    if torch.any(surface_runoff_rainfall_mask):
        flux["surface_runoff_depth_m"][surface_runoff_rainfall_mask] = flux["timestep_rainfall_input_m"][
            surface_runoff_rainfall_mask
        ]

    # Separate the infiltration from the total water input depth to the soil surface.
    flux["infiltration_depth_m"] = flux["timestep_rainfall_input_m"] - flux["surface_runoff_depth_m"]

    return flux, soil_reservoir


def adjust_and_track_runoff_infiltration(
    flux: Dict[str, torch.Tensor], soil_reservoir: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Module to adjust and track the runoff and infiltration for the CFE model at every timestep
    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status
        soil_reservoir (Dict[str, torch.Tensor]): soil reservoir parameters/status
    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'surface_runoff_depth_m': surface runoff depth in m/timestep
            'infiltration_depth_m': infiltration depth in m/timestep
        soil_reservoir parameters/status, updated with new
            'storage_deficit_m': storage deficit in m/timestep
            'storage_m': soil storage in m
    """

    ### adjust_runoff_and_infiltration
    """
    Calculates saturation excess overland flow (SOF)
    This should be run after calculate_infiltration_excess_overland_flow, then,
    infiltration_depth_m and surface_runoff_depth_m get finalized
    """

    # If the infiltration is more than the soil moisture deficit,
    # additional runoff (SOF) occurs and soil get saturated

    # Creating a mask where soil deficit is less than infiltration
    excess_infil_mask = soil_reservoir["storage_deficit_m"] < flux["infiltration_depth_m"]

    # If there are any such basins, we apply the conditional logic element-wise
    if torch.any(excess_infil_mask):
        diff = (flux["infiltration_depth_m"] - soil_reservoir["storage_deficit_m"])[excess_infil_mask]

        # Adjusting the surface runoff and infiltration depths for the specific basins
        flux["surface_runoff_depth_m"][excess_infil_mask] = flux["surface_runoff_depth_m"][excess_infil_mask] + diff
        flux["infiltration_depth_m"][excess_infil_mask] = soil_reservoir["storage_deficit_m"][excess_infil_mask]

        # This was missing from original implementation, added by Ziyu 11/11/24 from c code line 142
        soil_reservoir["storage_m"][excess_infil_mask] = soil_reservoir["storage_max_m"][excess_infil_mask]
        # Setting the soil reservoir storage deficit to zero for the specific basins
        soil_reservoir["storage_deficit_m"][excess_infil_mask] = 0.0

    return flux, soil_reservoir


def run_classic_soil_moisture_subroutine(
    flux: Dict[str, torch.Tensor], soil_reservoir: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Module to run the classic soil moisture subroutine for the CFE model at every timestep

    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status
        soil_reservoir (Dict[str, torch.Tensor]): soil reservoir parameters/status

    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'surface_runoff_depth_m': surface runoff depth in m/timestep
            'infiltration_depth_m': infiltration depth in m/timestep
            'primary_flux_m': primary flux in m/timestep
            'secondary_flux_m': secondary flux in m/timestep
        soil_reservoir parameters/status, updated with new
            'storage_deficit_m': storage deficit in m/timestep
            'storage_m': soil storage in m
    """

    mask_perc_soil = flux["flux_perc_m"] > soil_reservoir["storage_deficit_m"]
    if torch.any(mask_perc_soil):
        diff_perc_soil = flux["flux_perc_m"][mask_perc_soil] - soil_reservoir["storage_deficit_m"][mask_perc_soil]
        flux["infiltration_depth_m"][mask_perc_soil] = soil_reservoir["storage_deficit_m"][mask_perc_soil]
        # not added: lines 170 & 171 send flow back to giuh in self.vol & correct over-prediction of infilt
        flux["surface_runoff_depth_m"][mask_perc_soil] = flux["surface_runoff_depth_m"][mask_perc_soil] + diff_perc_soil
        soil_reservoir["storage_deficit_m"][mask_perc_soil] = 0.0

    # Assumes we don't have a single outlet exponential gw storage...
    # Add infiltration flux and calculate the reservoir flux
    # this is adjusted for ET already (not sure where this is from)
    soil_reservoir["storage_m"] = soil_reservoir["storage_m"] + flux["infiltration_depth_m"]

    ## do soil_conceptual_reservoir_flux_calc
    # Calculate primary flux
    storage_above_threshold_primary = soil_reservoir["storage_m"] - soil_reservoir["storage_threshold_primary_m"]
    primary_flux_mask = storage_above_threshold_primary > 0.0
    if torch.any(primary_flux_mask):
        storage_diff_primary = (
            soil_reservoir["storage_max_m"][primary_flux_mask] - soil_reservoir["storage_threshold_primary_m"][primary_flux_mask]
        )
        storage_ratio_primary = storage_above_threshold_primary[primary_flux_mask] / storage_diff_primary
        storage_power_primary = torch.pow(
            storage_ratio_primary, soil_reservoir["exponent_primary"]
        )  # "exponent primary" is scalar for now but can try to init as tensor
        flux["primary_flux_m"][primary_flux_mask] = soil_reservoir["coeff_primary"][primary_flux_mask] * storage_power_primary

        # a mask for when primary_flux > storage_above_primary
        primary_above_mask = flux["primary_flux_m"] > storage_above_threshold_primary

        # if primary_flux_m > storage_above_threshold_m then
        flux["primary_flux_m"][primary_flux_mask & primary_above_mask] = storage_above_threshold_primary[
            primary_flux_mask & primary_above_mask
        ].clone()

    # Calculate secondary flux
    storage_above_threshold_secondary = soil_reservoir["storage_m"] - soil_reservoir["storage_threshold_secondary_m"]
    secondary_flux_mask = storage_above_threshold_secondary > 0.0
    if torch.any(secondary_flux_mask):
        storage_diff_secondary = (
            soil_reservoir["storage_max_m"][secondary_flux_mask]
            - soil_reservoir["storage_threshold_secondary_m"][secondary_flux_mask]
        )
        storage_ratio_secondary = storage_above_threshold_secondary[secondary_flux_mask] / storage_diff_secondary
        storage_power_secondary = torch.pow(
            storage_ratio_secondary, soil_reservoir["exponent_secondary"]
        )  # "exponent_secondary" is also a scalar for now
        flux["secondary_flux_m"][secondary_flux_mask] = (
            soil_reservoir["coeff_secondary"][secondary_flux_mask] * storage_power_secondary
        )
        # crate a mask for when secondary_flux > storage_above_secondary - primary_flux_m
        secondary_above_mask = flux["secondary_flux_m"] > (storage_above_threshold_secondary - flux["primary_flux_m"])

        # if above is true then
        flux["secondary_flux_m"][secondary_flux_mask & secondary_above_mask] = (
            storage_above_threshold_secondary - flux["primary_flux_m"]
        )[secondary_flux_mask & secondary_above_mask]

    return flux, soil_reservoir


def adjust_from_soil_outflux(
    flux: Dict[str, torch.Tensor],
    constants: Dict[str, Union[int, float, Dict, torch.Tensor]],
    soil_reservoir: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Module to adjust the soil outflux for the CFE model at every timestep,
    applies different logic for classic and ode soil schemes.
    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status
        constants (Dict[str, Union[int, float, Dict, torch.Tensor]]): time and physics constants
        soil_reservoir (Dict[str, torch.Tensor]): soil reservoir parameters/status
    Raises:
        NotImplementedError: if the soil scheme is not classic or ode, change in config's dcfe_soil_scheme
        NotImplementedError: if the soil scheme is ode, not yet implemented, change in config's dcfe_soil_scheme

    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'flux_perc_m': percolation flux in m/timestep
            'flux_lat_m': lateral flux in m/timestep, gets defined here
        soil_reservoir parameters/status, updated with new
            'storage_m': soil storage in m
    """

    flux["flux_perc_m"] = flux["primary_flux_m"]  # percolation_flux
    flux["flux_lat_m"] = flux["secondary_flux_m"]  # lateral_flux

    # If the soil moisture scheme is classic, take out the outflux from soil moisture storage
    # If ODE, outfluxes are already subtracted from the soil moisture storage
    if constants["cfe_scheme"]["soil"] not in ["classic", "ode"]:
        raise NotImplementedError(
            f"Soil scheme '{constants['cfe_scheme']['soil']}' is not implemented. Supported: ['classic', 'ode']."
        )
    elif constants["cfe_scheme"]["soil"] == "classic":
        soil_reservoir["storage_m"] = soil_reservoir["storage_m"] - flux["flux_perc_m"]
        soil_reservoir["storage_m"] = soil_reservoir["storage_m"] - flux["flux_lat_m"]
        # We can probably combine both above
    elif constants["cfe_scheme"]["soil"] == "ode":
        raise NotImplementedError("ODE for soil scheme is not yet implemented. Change this in config's dcfe_soil_scheme:")

    return flux, soil_reservoir


def percolation_and_lateral_flow(flux: Dict[str, torch.Tensor], gw_reservoir: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Module to calculate the percolation and lateral flow for the CFE model at every timestep
    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status
        gw_reservoir (Dict[str, torch.Tensor]): groundwater reservoir parameters/status
    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'furface_runoff_depth_m': surface runoff depth in m/timestep
            'flux_perc_m': percolation flux in m/timestep
        gw_reservoir parameters/status, updated with new
            'storage_deficit_m': storage deficit in m/timestep
            'storage_m': groundwater storage in m
    """

    ### calculate_groundwater_storage_deficit
    gw_reservoir["storage_deficit_m"] = gw_reservoir["storage_max_m"] - gw_reservoir["storage_m"]
    ### adjust_precolation_to_gw
    overflow_mask = flux["flux_perc_m"] > gw_reservoir["storage_deficit_m"]

    # When the groundwater storage is full, the overflowing amount goes to direct runoff
    if torch.any(overflow_mask):
        # Calculate the amount of overflow
        diff = (flux["flux_perc_m"] - gw_reservoir["storage_deficit_m"])[overflow_mask].clone()
        # there's another variable previously named as diff, maybe we should choose a better name?

        # Overflow goes to surface runoff ## not sure where this is from.
        flux["surface_runoff_depth_m"][overflow_mask] = flux["surface_runoff_depth_m"][overflow_mask] + diff

        # Reduce the infiltration (maximum possible flux_perc_m is equal to gw_reservoir_storage_deficit_m)
        flux["flux_perc_m"][overflow_mask] = gw_reservoir["storage_deficit_m"][overflow_mask].clone()

        # Saturate the Groundwater storage # I believe storage + flux_prec_m saturated = storage max
        gw_reservoir["storage_m"][overflow_mask] = gw_reservoir["storage_max_m"][overflow_mask].clone()
        # ^^^^ this is redundant, we could just do storage + flux_perc for whole vector
        gw_reservoir["storage_deficit_m"][overflow_mask] = 0.0

    # Otherwise all the percolation flux goes to the storage
    # Apply the "otherwise" part of your condition, to all basins where overflow_mask is False
    no_overflow_mask = ~overflow_mask
    if torch.any(no_overflow_mask):
        gw_reservoir["storage_m"][no_overflow_mask] = (
            gw_reservoir["storage_m"][no_overflow_mask] + flux["flux_perc_m"][no_overflow_mask].clone()
        )

    return flux, gw_reservoir


def calculate_gw_reservoir_flux(
    timestep_conceptual_forcing: torch.Tensor,
    flux: Dict[str, torch.Tensor],
    cfe_params: Dict[str, torch.Tensor],
    gw_reservoir: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """This calculates the flux from a linear, or nonlinear
        conceptual reservoir with one or two outlets, or from an
        exponential nonlinear conceptual reservoir with only one outlet.
        In the non-exponential instance, each outlet can have its own
        activation storage threshold.  Flow from the second outlet is
        turned off by setting the discharge coeff. to 0.0.

    Args:
        timestep_conceptual_forcing (torch.Tensor): Tensor of size [batch_size, n_inputs], used for device
        flux (Dict[str, torch.Tensor]): flux parameters/status
        cfe_params (Dict[str, torch.Tensor]): must contain basinCharacteristics, soil_params
        gw_reservoir (Dict[str, torch.Tensor]): groundwater reservoir parameters/status

    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'primary_flux_from_gw_m': primary flux from groundwater in m/timestep
            'deep_gw_to_chan_m': deep groundwater to channel in m/timestep
        gw_reservoir parameters/status, updated with new
            'storage_m': groundwater storage in m
            'storage_deficit_m': storage deficit in m/timestep
    """

    device = timestep_conceptual_forcing.device
    batch_size = timestep_conceptual_forcing.shape[0]

    # This is basically only running for GW, so changed the variable name from primary_flux to primary_flux_from_gw_m to avoid confusion
    # if reservoir['is_exponential'] == True:
    flux_exponential = torch.exp(
        gw_reservoir["exponent_primary"] * gw_reservoir["storage_m"] / gw_reservoir["storage_max_m"]
    ) - torch.ones((batch_size), dtype=torch.float32, device=device)

    flux["primary_flux_from_gw_m"] = torch.where(
        cfe_params["basinCharacteristics"]["Cgw"] * flux_exponential < gw_reservoir["storage_m"],
        cfe_params["basinCharacteristics"]["Cgw"] * flux_exponential,
        gw_reservoir["storage_m"],
    )

    flux["from_deep_gw_to_chan_m"] = (
        flux["primary_flux_from_gw_m"] + flux["secondary_flux_from_gw_m"]
    )  # there's no 2nd flux since exponential

    ### track_volume_from_gw
    gw_reservoir["storage_m"] = gw_reservoir["storage_m"] - flux["from_deep_gw_to_chan_m"].clone()

    return flux, gw_reservoir


def calculate_convolutional_integral_for_GIUH(
    flux: Dict[str, torch.Tensor],
    routing_info: Dict[str, torch.Tensor],
    cfe_params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """This solves the convolution integral involving N GIUH ordinates.

    Args:
        flux (Dict[str, torch.Tensor]): flux parameters/status
        routing_info (Dict[str, torch.Tensor]): routing information
        cfe_params (Dict[str, torch.Tensor]): must contain basinCharacteristics, soil_params
    Returns:
        Dict[str, torch.Tensor]:
        flux parameters/status, updated with new
            'giuh_runoff_m': GIUH runoff in m/timestep
            'runoff_queue_m_per_timestep': runoff queue in m/timestep
        routing_info parameters/status, updated with new
            'runoff_queue_m_per_timestep': runoff queue in m/timestep
    """

    # Set the last element in the runoff queue as zero (runoff_queue[:-1] were pushed forward in the last timestep)
    routing_info["runoff_queue_m_per_timestep"][:, routing_info["num_ordinates"]] = 0.0

    routing_info["runoff_queue_m_per_timestep"][:, routing_info["num_ordinates"]] = 0.0

    # Add incoming surface runoff to the runoff queue
    routing_info["runoff_queue_m_per_timestep"][:, :-1] = routing_info["runoff_queue_m_per_timestep"][:, :-1] + (
        cfe_params["basinCharacteristics"]["giuh_ordinates"] * flux["surface_runoff_depth_m"].expand(routing_info["num_ordinates"], -1).T
    )

    # Take the top one in the runoff queue as runoff to channel
    flux["giuh_runoff_m"] = routing_info["runoff_queue_m_per_timestep"][:, 0].clone()

    # Shift all the entries forward in preperation for the next timestep
    routing_info["runoff_queue_m_per_timestep"][:, :-1] = routing_info["runoff_queue_m_per_timestep"][:, 1:].clone()

    return flux, routing_info


def run_nash_cascade(
    flux: Dict[str, torch.Tensor],
    cfe_params: Dict[str, torch.Tensor],
    routing_info: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """_summary_

    Returns:
        _type_: _description_
    """

    nash_storage_timestep = cfe_params["basinCharacteristics"]["nash_storage"].clone()

    # Calculate the discharge from each Nash storage
    Q = cfe_params["basinCharacteristics"]["K_nash"].unsqueeze(1) * nash_storage_timestep  # first pass would be 0

    # Update Nash storage with discharge
    nash_storage_timestep = nash_storage_timestep - Q  # first pass would be 0

    # The first storage receives the lateral flow outflux from soil storage
    nash_storage_timestep[:, 0] = nash_storage_timestep[:, 0] + flux["flux_lat_m"]

    # The remaining storage receives the discharge from the upper Nash storage
    if routing_info["num_reservoirs"] > 1:
        nash_storage_timestep[:, 1:] = nash_storage_timestep[:, 1:] + Q[:, :-1]
        # basinCharacteristics['nash_storage'][:, 1:] = basinCharacteristics['nash_storage'][:, 1:] + Q[:, :-1]

    # Update the state
    cfe_params["basinCharacteristics"]["nash_storage"] = nash_storage_timestep.clone()

    # The final discharge at the timestep from Nash cascade is from the lowermost Nash storage
    flux["nash_lateral_runoff_m"] = Q[:, -1].clone()

    return flux, routing_info, cfe_params


### sub-functions used in the file ###
def soil_reservoir_configuration(
    conceptual_forcing: torch.Tensor,
    cfe_params: Dict[str, torch.Tensor],
    constants: Dict[str, Union[int, float, Dict, torch.Tensor]],
) -> Dict[str, Union[int, float, Dict, torch.Tensor]]:
    """sub-function to calculate the soil reservoir configuration parameters for use in initializing the CFE model.
    Used before run, and at every timestep.
    Args:
        conceptual_forcing (torch.Tensor): Tensor of size [batch_size, time_steps, n_inputs],
        cfe_params (Dict[str, torch.Tensor]): must contain basinCharacteristics, soil_params
        constants (Dict[str, Union[int, float, Dict, torch.Tensor]]): time and physics constants

    Returns:
        Dict[str, Union[int, float, Dict, torch.Tensor]]:
            field_capacity_storage_threshold_m: field capacity storage threshold in meters, tensor of size [batch_size]
            lateral_flow_threshold_storage_m: lateral flow threshold in meters, tensor of size [batch_size]
        They are equal in the original CFE code, but we keep them separate for clarity.
    """

    device = conceptual_forcing.device
    batch_size = conceptual_forcing.shape[0]

    trigger_z_m = 0.5 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)
    field_capacity_atm_press_fraction = cfe_params["basinCharacteristics"]["alpha_fc"]
    # soil outflux calculation, Eq. 3
    H_water_table_m = (
        field_capacity_atm_press_fraction
        * constants["physics"]["atm_press_Pa"]
        / constants["physics"]["unit_weight_water_N_per_m3"]
    )  # [m]
    Omega = H_water_table_m - trigger_z_m
    # upper & lower limit of the integral in Eq. 4
    lower_lim = torch.pow(Omega, (1.0 - 1.0 / cfe_params["soil_params"]["bb"])) / (1.0 - 1.0 / cfe_params["soil_params"]["bb"])
    upper_lim = torch.pow(Omega + cfe_params["soil_params"]["D"], (1.0 - 1.0 / cfe_params["soil_params"]["bb"])) / (
        1.0 - 1.0 / cfe_params["soil_params"]["bb"]
    )
    # integral & power term in Eq.4, Eq.5
    storage_thresh_pow_term = torch.pow(1.0 / cfe_params["soil_params"]["satpsi"], (-1.0 / cfe_params["soil_params"]["bb"]))
    lim_diff = upper_lim - lower_lim
    field_capacity_storage_threshold_m = cfe_params["soil_params"]["smcmax"] * storage_thresh_pow_term * lim_diff
    # lateral flow function parameters
    lateral_flow_threshold_storage_m = field_capacity_storage_threshold_m

    return {
        "field_capacity_storage_threshold_m": field_capacity_storage_threshold_m,
        "lateral_flow_threshold_storage_m": lateral_flow_threshold_storage_m,
    }


def timestep_CFE_new(
    x_conceptual_timestep: torch.Tensor,
    cfe_params: Dict[str, torch.Tensor],
    timestep_parameters: Dict[str, torch.Tensor],
    constants: Dict[str, torch.Tensor],
    gw_reservoir: Dict[str, torch.Tensor],
    soil_reservoir: Dict[str, torch.Tensor],
    routing_info: Dict[str, torch.Tensor],
    flux: Dict[str, torch.Tensor],
    hourly: bool = False,
):
    # grab LSTM parameters, adapt them into the CFE params & reservoirs
    cfe_params, gw_reservoir, soil_reservoir = timestep_basin_constants(
        conceptual_forcing_timestep=x_conceptual_timestep,
        gw_reservoir=gw_reservoir,
        soil_reservoir=soil_reservoir,
        cfe_params=cfe_params,
        constants=constants,
        timestep_params=timestep_parameters,
    )

    # initialize fluxes
    flux = initialize_flux_timestep(conceptual_forcing_timestep=x_conceptual_timestep, flux=flux)

    # calculate pet and get rainfall
    flux = get_and_calculate_input_rainfall_and_ET(
        conceptual_forcing_timestep=x_conceptual_timestep, flux=flux, constants=constants, hourly=hourly
    )

    # calculate evaporation
    # from rain
    flux = calculate_evaporation_from_rainfall(flux=flux)
    # from soil
    flux, soil_reservoir = calculate_evaporation_from_soil(flux=flux, constants=constants, soil_reservoir=soil_reservoir)

    # infiltration partitioning
    if constants["cfe_scheme"]["partition"] not in ["Schaake", "Xinanjiang"]:
        raise NotImplementedError(
            f"Partition scheme '{constants['cfe_scheme']['partition']}' is not implemented. "
            "Supported: ['Schaake', 'Xinanjiang']. Change in dcfe_partition_scheme in config."
        )
    elif constants["cfe_scheme"]["partition"] == "Schaake":
        flux, soil_reservoir = run_Schaake_subroutine(
            flux=flux, constants=constants, cfe_params=cfe_params, soil_reservoir=soil_reservoir
        )
    elif constants["cfe_scheme"]["partition"] == "Xinanjiang":
        raise NotImplementedError(
            " Xinanjiang parition shceme is coded but not tested, choose different scheme in config's dcfe_partition_scheme."
        )

    flux, soil_reservoir = adjust_and_track_runoff_infiltration(flux=flux, soil_reservoir=soil_reservoir)

    # soil moisture reservoir
    if constants["cfe_scheme"]["soil"] not in ["classic", "ode"]:
        raise NotImplementedError(
            f"Soil scheme '{constants['cfe_scheme']['soil']}' is not implemented. "
            "Supported: ['classic', 'ode']. Change in dcfe_soil_scheme in config."
        )
    elif constants["cfe_scheme"]["soil"] == "classic":
        flux, soil_reservoir = run_classic_soil_moisture_subroutine(flux=flux, soil_reservoir=soil_reservoir)
    elif constants["cfe_scheme"]["soil"] == "ode":
        raise NotImplementedError("ODE mode is not in NH-dCFE, change in config's dcfe_soil_scheme.")

    flux, soil_reservoir = adjust_from_soil_outflux(flux=flux, constants=constants, soil_reservoir=soil_reservoir)

    # gw reservoir processes
    flux, gw_reservoir = percolation_and_lateral_flow(flux=flux, gw_reservoir=gw_reservoir)

    flux, gw_reservoir = calculate_gw_reservoir_flux(
        timestep_conceptual_forcing=x_conceptual_timestep, flux=flux, cfe_params=cfe_params, gw_reservoir=gw_reservoir
    )

    # surface runoff routing
    flux, routing_info = calculate_convolutional_integral_for_GIUH(flux=flux, routing_info=routing_info, cfe_params=cfe_params)

    # lateral flow routing
    flux, routing_info, cfe_params = run_nash_cascade(flux=flux, routing_info=routing_info, cfe_params=cfe_params)

    # outflow in m
    flux["Qout_m"] = flux["giuh_runoff_m"] + flux["nash_lateral_runoff_m"] + flux["from_deep_gw_to_chan_m"]

    return cfe_params, gw_reservoir, soil_reservoir, routing_info, flux
