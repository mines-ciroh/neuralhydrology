# original shm packages
import torch
import torch.nn as nn
from typing import Dict, Union
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config

# packages from cfe.py
import time
import numpy as np
import pandas as pd
import sys
import math
# import torch
import torch.nn.functional as F
from torchdiffeq import odeint

# packages from bmi_cfe.py
import matplotlib.pyplot as plt
from torch import Tensor

class dCFE(BaseConceptualModel):
    """
    This is an attempt to make a dCFE model based on 
    https://github.com/NWC-CUAHSI-Summer-Institute/ngen-aridity/blob/main/Project%20Manuscript_LongForm.pdf
    
    
    Last edited: by Ziyu, 09/01/2024
    
    General outline:
    Takes raw LSTM output, shape them within possible ranges of the Cgw and satdk parameters. Together with
    other basin-specific parameters and forcings (precip and pet) and pass through the CFE for runoff predictions. 
    This model is tailored to basin ID: 01022500 right now, but can be worked on later to train multi-basin. 
    
    The physics is done and forward process & backward processes work so far with this specific basin and time-period of data. 
    
    TODO: 
    Debug/double check for correct physical model & magnitudes
    Defining "states" differently to help organize better
    Incorporate multi-basin training
    Improve readability
    """
    
    def __init__(self, cfg: Config):
        super(dCFE, self).__init__(cfg=cfg)
        
        
        
    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Perform a forward pass in the CFE.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs is the number of normalized forcings. 
            This will be used in the NN part only to get the 9 calibration parameters. (Maybe)
        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters]. This tensor comes from the data-driven model,
            and will be used to obtain the dynamic parameterization of the conceptual model. 
            
        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
                There are 9 parameters total, 6 tracked parameters (maybe) for futre work,
                but for right now it would be satdk and Cgw like the original dCFE proj. 
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model
                This will have to be different from cfe_states in the original framework,
                still working on this.

        """
        # get model params thru baseconceptualmodel.py's function, 
        # this ensure that the output from NN is within the correct range, built into NH
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)


        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)
        
        
        # This contains traits specific to the basin. Cgw is removed from this section for better gradient tracking
        # and gw_storage removed since it was not used in the code. 
        # key for elements in basinChracteristics:
        # - catchment_area_km2: tensor, [batch_size, timestep]. Area of the basin [km2]
        # - redfdk: tensor, [batch_size, timestep]. Runoff partitioning parameter [-] 
        # - max_gw_storage: tensor, [batch_size, timestep]. Max groundwater storage [m]. Part of CFE calibration
        # - expon: tensor, [batch_size, timestep]. A primary groundwater nonlinear reservoir exponential constant [-]. Part of CFE calibration
        # - alpha_fc: tensor, [batch_size, timestep]. 
        # - K_nash: tensor, [batch_size, timestep]. Nash cascade discharge coefficient [-]. Part of CFE calibration
        # - K_lf: tensor, [batch_size, timestep]. Lateral flow coefficient [-]. Part of CFE calibration
        # - nash_storage: tensor, [batch_size, 2]. 2 columns for 2 "buckets" of Nash cascade
        # - giuh_ordinates: tensor, [num_coordinates]. 
        basinCharacteristics = {
            'catchment_area_km2': 573.6 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), 
            'refkdt': 3.8266861353378374 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'max_gw_storage': 0.021342666010108112 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            #Cgw = parameters['Cgw'] # the below is going to be a parameter from NN
            'expon': 6.72972972972973 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            #gw_storage = 0.05 * torch.ones((x_conceptual.shape[0], x_conceptual.shape[1]), dtype=torch.float32, device=x_conceptual.device)
            'alpha_fc': 0.33 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'K_nash': 0.03 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), 
            'K_lf': 0.01 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), 
            'nash_storage': torch.zeros((x_conceptual.shape[0],2), dtype=torch.float32, device=x_conceptual.device),
            'giuh_ordinates': torch.tensor([0.2, 0.3, 0.3, 0.1, 0.1], dtype=torch.float32, device=x_conceptual.device)
        }
        
        # TODO:
        # Change config.py to allow for soil_scheme and partition_scheme to be recongnized as valid in NH, instead of defining it here
        soil_scheme = 'classic'
        partition_scheme = 'Schaake'
        # Re-organize by creating dicttionaries for variables such as initialization and basin attributes
        
        
        # ________some other constants_______
        # time-related constants
        time_step_size = 3600 # num of [seconds] per hour, we go by 3600s each time step
        timestep_h = time_step_size/3600 # time step in [hours]
        timestep_d = timestep_h/24 # time step in [days]
        
        atm_press_Pa = 101325.0 # [Pa]
        unit_weight_water_N_per_m3 = 9810.0 # [N/m3]
        
            # ______________defining parameters that are specific to this basin, from bmi_config_cfe.json_______
        # soil_params will have tensor elements of size [batch_size, time_steps]. For non-NN parameters, they would be basin-specific
        # but right now it's just 01022500 so these constants are at every cell.
        # In the future, they can be defined differently based on different basins. 
        soil_params = {
            'depth': 2.0 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # not sure where they got these values, they don't match CAMELS, [m]
            'bb': 8.013513513513514 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # exponent on Clapp-Hornberger function, part of calibration
            # satdk is from NH
            'satdk': parameters['satdk'], # saturated hydraulic conductivity [m/hr], part of calibration
            'satpsi': 0.1647076737162162 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), 
            'slop': 0.08824091635135137 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # slope coefficient, part of calibration
            'smcmax': 0.37300223004054056 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # maximum soil moisture content [m3/m3], part of calibration
            'wltsmc': 0.04966811960810811* torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'D': 2.0 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'mult': 1000.0 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
        }
        
        ###  defining soil_reservoir & gw_reservoir constants that are basin specific
        # __________modified from bmi_cfe.py____________
        ## Groundwater/subsurface reservoir (need to be batchsize x timesteps)
        gw_reservoir = {
            'storage_max_m': basinCharacteristics['max_gw_storage'],
            # "coeff_primary": self.Cgw, -> this has been changed to dynamic parameter
            'exponent_primary': basinCharacteristics['expon'],
            'storage_threshold_primary_m': 0 ,
            # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off.
            'storage_threshold_secondary_m': 0,
            'coeff_secondary': 0,
            'exponent_secondary': 1,
        }
        gw_reservoir['storage_m'] = gw_reservoir['storage_max_m'] * 0.01
        # volstart = volstart.add(gw_reservoir["storage_m"]) not sure what this does
        vol_in_gw_start = gw_reservoir["storage_m"]
        #^^^for ODE part
        
        ## Soil Reservoir Configuration
        # local values to be used in setting up soil reservoir
        trigger_z_m = 0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
        field_capacity_atm_press_fraction = basinCharacteristics['alpha_fc']
        # soil outflux calculation, Eq. 3
        H_water_table_m = field_capacity_atm_press_fraction * atm_press_Pa/unit_weight_water_N_per_m3 # [m]
        Omega = H_water_table_m - trigger_z_m
        # upper & lower limit of the integral in Eq. 4
        lower_lim = torch.pow(Omega, (1.0 - 1.0 / soil_params["bb"])) / (
            1.0 - 1.0 / soil_params["bb"])
        upper_lim = torch.pow(Omega + soil_params["D"], (1.0 - 1.0 / soil_params["bb"])) / (1.0 - 1.0 / soil_params["bb"])
        # integral & power term in Eq.4, Eq.5
        storage_thresh_pow_term = torch.pow(1.0 / soil_params["satpsi"], (-1.0 / soil_params["bb"]))
        lim_diff = upper_lim - lower_lim
        field_capacity_storage_threshold_m = (soil_params["smcmax"] * storage_thresh_pow_term * lim_diff)
        # lateral flow function parameters
        lateral_flow_threshold_storage_m = field_capacity_storage_threshold_m
        # soil_reservoir_storage_deficit_m = torch.tensor([0.0], dtype=torch.float64) # Not sure why this is here, defined later to be something different
        
        soil_reservoir = {
            'wilting_point_m': soil_params['wltsmc']*soil_params['D'],
            'storage_max_m': soil_params['smcmax']*soil_params['D'],
            'coeff_primary': parameters['satdk'] * soil_params['slop'].unsqueeze(1)/time_step_size, #Eq.11, unit [m/hr] * [3600s]??? Should be m/s
            'exponent_primary': 1.0, # fixed to 1 based on Eq. 11
            'storage_threshold_primary_m': field_capacity_storage_threshold_m, # place holder for now, this is smcmax * storage_thresh_pow_term*lim_diff
            'coeff_secondary': basinCharacteristics['K_lf'],  # Controls lateral flow
            'exponent_secondary': 1.0,  # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
            'storage_threshold_secondary_m': lateral_flow_threshold_storage_m, ## but this is the same as field_capacity_storage_threshold_m??
        }
        soil_reservoir['storage_m'] = soil_reservoir['storage_max_m'] * 0.6
        # volstart = volstart.add(soil_reservoir['storage_m'])
        vol_soil_start = soil_reservoir['storage_m'] # not used
        
        ## Schaake partitioning Constants
        Schaake_adjusted_magic_constant_by_soil_type = basinCharacteristics['refkdt'].unsqueeze(1) * parameters['satdk'] / 2.0e-06 # would be basin_size x timestep
        
        N = basinCharacteristics['giuh_ordinates'].shape[0] # giuh_ordinates are rows x 1 column for each basin, used in routing
            
        # initialized only once, for Nash cascade
        runoff_queue_m_per_timestep = torch.ones((x_conceptual.shape[0], N + 1), dtype=torch.float32, device=x_conceptual.device)
        
        
        # reset volume tracking, part of reset_volume_tracking
        # this contains all variables from CFE that starts with vol_ and all are 1D tensors of dimension [batch_size]
        vol = {
            'PET': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'partition_runoff': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'partition_infilt': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'et_from_rain': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'et_from_soil': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'et_to_atm': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'to_gw': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'to_soil': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'soil_to_gw': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'soil_to_lat_flow': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'from_gw': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'out_giuh': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'in_nash': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'out_nash': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'out': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
        }
        
        # loop through each timestep
        for j in range(x_conceptual.shape[1]):
            # reset fluxes that can store information at every time-step. This will be #basin x 
            surface_runoff_depth_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            infiltration_depth_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            actual_et_from_rain_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            actual_et_from_soil_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            actual_et_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            # reset ET
            reduced_potential_et_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            
            primary_flux_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            secondary_flux_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            # below are all added later, not in original initialization
            primary_flux_from_gw_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            secondary_flux_from_gw_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            flux_nash_lateral_runoff_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            flux_from_deep_gw_to_chan_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
            
            
            # read forcings for time step
            potential_et_m_per_s = x_conceptual[:,j,1]/1000/3600 # convert PET mm/hr to [m/s]
            timestep_rainfall_input_m = x_conceptual[:,j,0]/1000 # convert precip mm/hr to [m/hr]
            ####__________Rainfall and ET___________####
            
            ### calculate input rainfall and ET
            potential_et_m_per_timestep = potential_et_m_per_s * time_step_size # results in [m/hr]
            # potential_et_m_per_timestep = potential_et_m_per_timestep.view(-1,1)
            vol['PET'] = vol['PET'] + potential_et_m_per_timestep
            reduced_potential_et_m_per_timestep = potential_et_m_per_s * time_step_size # same as potential_et_m_per_timestep?
            
            ### calculate evaporation from rainfall
            # Creating a mask for elements where timestep_rainfall_input_m > 0
            rainfall_mask = timestep_rainfall_input_m > 0 ### couldn't find where timestep_rainfall_input_m is defined, but pretty sure this is precip
            if torch.any(rainfall_mask): # if there's precip in this batch then
                
                ## Calculate et_from_rainfall
                # apply the mask
                rainfall = timestep_rainfall_input_m[rainfall_mask]
                pet = potential_et_m_per_timestep[rainfall_mask]
                
                # If rainfall exceeds PET, actual AET from rainfall is equal to the PET
                # Otherwise, actual ET equals to potential ET
                # condition is set for each sample within the same time-step
                condition = rainfall > pet

                actual_et_from_rain = torch.where(
                    condition,
                    pet,  # If P > PET, AET from P is equal to the PET
                    rainfall,  # If P < PET, all P gets consumed as AET
                    )
                
                reduced_rainfall = torch.where(
                    condition,
                    rainfall - actual_et_from_rain,  #  # If P > PET, part of P is consumed as AET
                    torch.zeros_like(rainfall),  # If P < PET, all P gets consumed as AET
                    )
                
                reduced_potential_et = pet - actual_et_from_rain
            
                # storing results back to the states
                actual_et_from_rain_m_per_timestep[rainfall_mask] = actual_et_from_rain
                # should the below be states????? They are not right now.
                timestep_rainfall_input_m[rainfall_mask] = reduced_rainfall # adjusting precip based on evaporation from rainfall
                reduced_potential_et_m_per_timestep[rainfall_mask] = reduced_potential_et # adjusting pet based on evaporation from rainfall
                
                ## And track_volume_from_rainfall (should the following be states as well???)
                vol['et_from_rain'] = vol['et_from_rain'] + actual_et_from_rain_m_per_timestep
                vol['et_to_atm'] = vol['et_to_atm'] + actual_et_from_rain_m_per_timestep
                vol['out'] = vol['out'] + actual_et_from_rain_m_per_timestep
                actual_et_m_per_timestep = actual_et_m_per_timestep + actual_et_from_rain_m_per_timestep
                
            ### Calculate evaporation from soil ###
            # for this sub-module, check if the soil scheme is "classic" or "ode". If this is "ode", do nothing.
            if soil_scheme == "classic":
                # Creating a mask for elements where excess soil moisture > 0
                excess_sm_for_ET_mask = (soil_reservoir["storage_m"] > soil_reservoir["wilting_point_m"]) # this would be one element, Y or N
                et_mask = reduced_potential_et_m_per_timestep > 0
                 # Combine both masks
                combined_mask = et_mask & excess_sm_for_ET_mask
                # If the soil moisture storage is more than wilting point, and PET is not zero, calculate ET from soil
                if torch.any(combined_mask):
                    # calculate et_from_soil: take AET from soil moisture storage & use Budyko curve...
                    storage = soil_reservoir["storage_m"][combined_mask]
                    threshold = soil_reservoir["storage_threshold_primary_m"][combined_mask]
                    wilting_point = soil_reservoir["wilting_point_m"][combined_mask]
                    reduced_pet = reduced_potential_et_m_per_timestep[combined_mask]
                    
                    condition1 = storage >= threshold
                    condition2 = (storage > wilting_point) & (storage < threshold)
                    
                    actual_et_from_soil = torch.where(
                        condition1, 
                        torch.where(
                            reduced_pet < storage, 
                            reduced_pet,
                            storage),# If storage is above the FC threshold, AET = PET
                        torch.where(
                            condition2,
                            torch.where(
                                (storage - wilting_point)/(threshold - wilting_point) * reduced_pet < storage,
                                (storage - wilting_point)/(threshold - wilting_point) * reduced_pet,
                                storage
                            ),# If storage is in bewteen ßthe FC and WP threshold, calculate the Budyko type of AET
                            torch.zeros_like(storage) # If storage is less than WP, AET=0
                        )
                    )
                    
                    
                    # update the timestep's et from soil????
                    actual_et_from_soil_m_per_timestep[combined_mask] = actual_et_from_soil
                    # adjust soil storage by subtracting et from soil
                    soil_reservoir["storage_m"][combined_mask] = soil_reservoir["storage_m"][combined_mask] - actual_et_from_soil
                    ###### Should below be batch_size x timestep as well??? This is initialized every timestep right now
                    reduced_potential_et_m_per_timestep[combined_mask] = reduced_potential_et_m_per_timestep[combined_mask] - actual_et_from_soil
                    
                    # and now track_volume_et_from_soil. Should the following be states as well??
                    vol['et_from_soil'] = vol['et_from_soil'] + actual_et_from_soil_m_per_timestep
                    vol['et_to_atm'] = vol['et_to_atm'] + actual_et_from_soil_m_per_timestep
                    vol['out'] = vol['out'] + actual_et_from_soil_m_per_timestep
                    actual_et_m_per_timestep = actual_et_m_per_timestep + actual_et_from_soil_m_per_timestep
            
            ####__________Infiltration partitioning________###
            ### calculate_the_soil_moisture_deficit
            soil_reservoir_storage_deficit_m = soil_params["smcmax"] * soil_params["D"] - soil_reservoir["storage_m"]
            ### calculate_infiltration_excess_overland_flow by running partition scheme based on choice set in the configuration file
            # rainfall_mask = timestep_rainfall_input_m > 0.0 # do not need this as it is defined before
            # The original code in cfe.py has each basin might have different partition schemes but not sure how this can be implemented,
            # maybe it makes more sense to say one scheme for all basins
            if partition_scheme == "Schaake":
                # copied from cfe.py
                """
                This subtroutine takes water_input_depth_m and partitions it into surface_runoff_depth_m and
                infiltration_depth_m using the scheme from Schaake et al. 1996.
                !--------------------------------------------------------------------------------
                modified by FLO April 2020 to eliminate reference to ice processes,
                and to de-obfuscate and use descriptive and dimensionally consistent variable names.

                inputs:
                timestep_d
                Schaake_adjusted_magic_constant_by_soil_type = C*Ks(soiltype)/Ks_ref, where C=3, and Ks_ref=2.0E-06 m/s
                column_total_soil_moisture_deficit_m (soil_reservoir_storage_deficit_m)
                water_input_depth_m (timestep_rainfall_input_m) amount of water input to soil surface this time step [m]
                outputs:
                surface_runoff_depth_m      amount of water partitioned to surface water this time step [m]
                infiltration_depth_m
                """
                rainfall = timestep_rainfall_input_m[rainfall_mask] # this is rainfall, here it is adjusted from ET before..
                deficit = soil_reservoir_storage_deficit_m[rainfall_mask]
                magic_const = Schaake_adjusted_magic_constant_by_soil_type[rainfall_mask,j]
                
                exp_term = torch.exp(-magic_const * timestep_d)
                Ic = deficit * (1 - exp_term)
                Px = rainfall
                infilt = Px * (Ic / (Px + Ic))
                
                # If the rainfall > infiltration, runoff is generated
                # If rainfall < infiltration, no runoff, all of the preciptiation are infiltratied
                runoff = torch.where(
                    rainfall - infilt > 0, 
                    rainfall - infilt, 
                    torch.zeros_like(rainfall))
                infilt = rainfall - runoff

                surface_runoff_depth_m[rainfall_mask] = runoff
                infiltration_depth_m[rainfall_mask] = infilt
            elif partition_scheme == "Xinanjiang":
                """
                TODO: THIS MODULE IS NOT PREPARED FOR MULTI_BASIN RUN YET

                This module takes the water_input_depth_m and separates it into surface_runoff_depth_m
                and infiltration_depth_m by calculating the saturated area and runoff based on a scheme developed
                for the Xinanjiang model by Jaywardena and Zhou (2000). According to Knoben et al.
                (2019) "the model uses a variable contributing area to simulate runoff.  [It] uses
                a double parabolic curve to simulate tension water capacities within the catchment,
                instead of the original single parabolic curve" which is also used as the standard
                VIC fomulation.  This runoff scheme was selected for implementation into NWM v3.0.
                REFERENCES:
                1. Jaywardena, A.W. and M.C. Zhou, 2000. A modified spatial soil moisture storage
                capacity distribution curve for the Xinanjiang model. Journal of Hydrology 227: 93-113
                2. Knoben, W.J.M. et al., 2019. Supplement of Modular Assessment of Rainfall-Runoff Models
                Toolbox (MARRMoT) v1.2: an open-source, extendable framework providing implementations
                of 46 conceptual hydrologic models as continuous state-space formulations. Supplement of
                Geosci. Model Dev. 12: 2463-2480.
                -------------------------------------------------------------------------
                Written by RLM May 2021
                Adapted by JMFrame September 2021 for new version of CFE
                Further adapted by QiyueL August 2022 for python version of CFE
                ------------------------------------------------------------------------
                Inputs
                double  time_step_rainfall_input_m           amount of water input to soil surface this time step [m]
                double  field_capacity_m                     amount of water stored in soil reservoir when at field capacity [m]
                double  max_soil_moisture_storage_m          total storage of the soil moisture reservoir (porosity*soil thickness) [m]
                double  column_total_soil_water_m     current storage of the soil moisture reservoir [m]
                double  a_inflection_point_parameter  a parameter
                double  b_shape_parameter             b parameter
                double  x_shape_parameter             x parameter
                //
                Outputs
                double  surface_runoff_depth_m        amount of water partitioned to surface water this time step [m]
                double  infiltration_depth_m          amount of water partitioned as infiltration (soil water input) this time step [m]
                -------------------------------------------------------------------------
                """
                # partition the total soil water in the column between free water and tension water
                free_water_m = soil_reservoir["storage_m"] - soil_reservoir["storage_threshold_primary_m"] # this storage was adjusted for ET
                if free_water_m > 0.0:
                    tension_water_m = soil_reservoir["storage_threshold_primary_m"]
                else:
                    free_water_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                    tension_water_m = soil_reservoir["storage_m"]
                
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
                a_Xinanjiang_inflection_point_parameter = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                b_Xinanjiang_shape_parameter = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                x_Xinanjiang_shape_parameter = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                
                if (tension_water_m / max_tension_water_m) <= (
                    0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                    - a_Xinanjiang_inflection_point_parameter
                ):
                    pervious_runoff_m = timestep_rainfall_input_m * (
                        torch.pow(
                            (
                                0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                                - a_Xinanjiang_inflection_point_parameter
                            ),
                            (
                            torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                            - b_Xinanjiang_shape_parameter
                            ),
                        )
                        * torch.pow(
                            (
                                torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                                - (tension_water_m / max_tension_water_m)
                            ),
                            b_Xinanjiang_shape_parameter,
                        )
                    )
                
                else:
                    pervious_runoff_m = timestep_rainfall_input_m * (
                        torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                        - torch.pow(
                            (
                                0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                                + a_Xinanjiang_inflection_point_parameter
                            ),
                            (
                                torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                                - b_Xinanjiang_shape_parameter
                            ),
                        )
                        * torch.pow(
                            (
                                torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                                - (tension_water_m / max_tension_water_m)
                            ),
                            (b_Xinanjiang_shape_parameter),
                        )
                    )
                    
                # Separate the surface water from the pervious runoff
                ## NOTE: If impervious runoff is added to this subroutine, impervious runoff should be added to
                ## the surface_runoff_depth_m.
                
                surface_runoff_depth_m = pervious_runoff_m * (
                    0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                    - torch.pow(
                        (
                            0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
                            - (free_water_m / max_free_water_m)
                        ),
                        x_Xinanjiang_shape_parameter,
                    )
                )
                
                # The surface runoff depth is bounded by a minimum of 0 and a maximum of the water input depth.
                # Check that the estimated surface runoff is not less than 0.0 and if so, change the value to 0.0.
                surface_runoff0_mask = surface_runoff_depth_m < 0.0 # we have a vector instead of scalar
                if torch.any(surface_runoff0_mask):
                    surface_runoff_depth_m[surface_runoff0_mask] = torch.zeros(
                        (torch.sum(surface_runoff0_mask), 1), 
                        dtype=torch.float32, device=x_conceptual.device
                    )
                
                # Check that the estimated surface runoff does not exceed the amount of water input to the soil surface.  If it does,
                # change the surface water runoff value to the water input depth.
                surface_runoff_rainfall_mask = surface_runoff_depth_m > timestep_rainfall_input_m
                if torch.any(surface_runoff_rainfall_mask):
                    surface_runoff_depth_m[surface_runoff_rainfall_mask] = timestep_rainfall_input_m[surface_runoff_rainfall_mask]
                    
                # Separate the infiltration from the total water input depth to the soil surface.
                infiltration_depth_m = timestep_rainfall_input_m - surface_runoff_depth_m
                ##### infiltration_depth_m has been initialized at 2 different spots, need to decide where to put them. 
            else:
                print(
                    "Problem: must specify one of Schaake or Xinanjiang partitioning scheme."
                )
                print("Program terminating.:( \n")
                
            ### adjust_runoff_and_infiltration
            """Calculates saturation excess overland flow (SOF)
            This should be run after calculate_infiltration_excess_overland_flow, then,
            infiltration_depth_m and surface_runoff_depth_m get finalized
            """
            # If the infiltration is more than the soil moisture deficit,
            # additional runoff (SOF) occurs and soil get saturated
            
            # Creating a mask where soil deficit is less than infiltration
            excess_infil_mask = soil_reservoir_storage_deficit_m < infiltration_depth_m
            
            # If there are any such basins, we apply the conditional logic element-wise
            if torch.any(excess_infil_mask):
                diff = (infiltration_depth_m - soil_reservoir_storage_deficit_m)[excess_infil_mask]
                # Adjusting the surface runoff and infiltration depths for the specific basins
                surface_runoff_depth_m[excess_infil_mask] = surface_runoff_depth_m[excess_infil_mask] + diff
                infiltration_depth_m[excess_infil_mask] = infiltration_depth_m[excess_infil_mask] - diff

                # Setting the soil reservoir storage deficit to zero for the specific basins
                soil_reservoir_storage_deficit_m[excess_infil_mask] = 0.0
            
            ## track_infiltration_and_runoff
            """Tracking runoff & infiltraiton volume with final infiltration & runoff values"""
            vol['partition_runoff'] = vol['partition_runoff'] + surface_runoff_depth_m
            vol['partition_infilt'] = vol['partition_infilt'] + infiltration_depth_m
            vol['to_soil'] = vol['to_soil'] + infiltration_depth_m
            
            ####______________Soil moisture reservoir_____________####
            ### Start run_soil_moisture_scheme
            if soil_scheme == "classic":
                # Add infiltration flux and calculate the reservoir flux
                # this is adjusted for ET already
                soil_reservoir["storage_m"] = soil_reservoir["storage_m"] + infiltration_depth_m
                
                ## do soil_conceptual_reservoir_flux_calc
                # Calculate primary flux
                storage_above_threshold_primary = soil_reservoir["storage_m"] - soil_reservoir["storage_threshold_primary_m"]
                primary_flux_mask = storage_above_threshold_primary > 0.0
                if torch.any(primary_flux_mask):
                    storage_diff_primary = soil_reservoir["storage_max_m"] - soil_reservoir["storage_threshold_primary_m"]
                    storage_ratio_primary = storage_above_threshold_primary / storage_diff_primary
                    storage_power_primary = torch.pow(storage_ratio_primary, soil_reservoir["exponent_primary"]) # "exponent primary" is scalar for now but can try to init as tensor
                    primary_flux = soil_reservoir["coeff_primary"][:,j] * storage_power_primary
                    primary_flux_m[primary_flux_mask] = torch.where(
                        primary_flux < storage_above_threshold_primary, 
                        primary_flux, 
                        storage_above_threshold_primary)[primary_flux_mask]
                    
                # Calculate secondary flux
                storage_above_threshold_secondary = soil_reservoir["storage_m"] - soil_reservoir["storage_threshold_secondary_m"]
                secondary_flux_mask = storage_above_threshold_secondary > 0.0
                if torch.any(secondary_flux_mask):
                    storage_diff_secondary = soil_reservoir["storage_max_m"] - soil_reservoir["storage_threshold_secondary_m"]
                    storage_ratio_secondary = storage_above_threshold_secondary / storage_diff_secondary
                    storage_power_secondary = torch.pow(storage_ratio_secondary, soil_reservoir["exponent_secondary"]) # "exponent_secondary" is also a scalar for now
                    secondary_flux = soil_reservoir["coeff_secondary"] * storage_power_secondary
                    secondary_flux_m[secondary_flux_mask] = torch.where(
                        secondary_flux < (storage_above_threshold_secondary - primary_flux_m),
                        secondary_flux,
                        storage_above_threshold_secondary - primary_flux_m)[secondary_flux_mask]

            elif soil_scheme == "ode":
                print(
                    "ODE for soil scheme is not yet implemented."
                ) # we can come back and implement this later
                print("Program terminating.:( \n")
            else:
                print(
                    "Either a classic or ode soil scheme must be chosen."
                )
                print("Program terminating.:( \n")
                
            ### update_outflux_from_soil
            flux_perc_m = primary_flux_m  # percolation_flux
            flux_lat_m = secondary_flux_m# lateral_flux
            
            # If the soil moisture scheme is classic, take out the outflux from soil moisture storage
            # If ODE, outfluxes are already subtracted from the soil moisture storage
            if soil_scheme == "classic":
                soil_reservoir['storage_m'] = soil_reservoir['storage_m'] - flux_perc_m
                soil_reservoir["storage_m"]= soil_reservoir["storage_m"]- flux_lat_m
                # We can probably combine both above
            elif soil_scheme == "ode":
                print(
                    "ODE for soil scheme is not yet implemented, and storage cannot be stored."
                ) # we can come back and implement this later
                print("Program terminating.:( \n")
            
            #### ______ groundwater reservoir _________
            ### calculate_groundwater_storage_deficit
            gw_reservoir_storage_deficit_m = gw_reservoir["storage_max_m"]- gw_reservoir["storage_m"]
            ### adjust_precolation_to_gw
            overflow_mask = flux_perc_m > gw_reservoir_storage_deficit_m
            
            # When the groundwater storage is full, the overflowing amount goes to direct runoff
            if torch.any(overflow_mask):
                # Calculate the amount of overflow
                diff = (flux_perc_m - gw_reservoir_storage_deficit_m)[overflow_mask].clone()
                # there's another variable previously named as diff, maybe we should choose a better name?

                # Overflow goes to surface runoff
                surface_runoff_depth_m[overflow_mask] = surface_runoff_depth_m[overflow_mask] + diff

                # Reduce the infiltration (maximum possible flux_perc_m is equal to gw_reservoir_storage_deficit_m)
                flux_perc_m[overflow_mask] = gw_reservoir_storage_deficit_m[overflow_mask].clone()

                # Saturate the Groundwater storage
                gw_reservoir["storage_m"][overflow_mask] = gw_reservoir["storage_max_m"][overflow_mask].clone()
                gw_reservoir_storage_deficit_m[overflow_mask] = 0.0

                # Track volume
                vol['partition_runoff'][overflow_mask] = vol['partition_runoff'][overflow_mask] + diff
                vol['partition_infilt'][overflow_mask] = vol['partition_infilt'][overflow_mask] + diff
                
            # Otherwise all the percolation flux goes to the storage
            # Apply the "otherwise" part of your condition, to all basins where overflow_mask is False
            no_overflow_mask = ~overflow_mask
            if torch.any(no_overflow_mask):
                gw_reservoir["storage_m"][no_overflow_mask] = gw_reservoir["storage_m"][no_overflow_mask] + flux_perc_m[no_overflow_mask].clone()
            
            ### track_volume_from_percolation_and_lateral_flow
            vol['to_gw'] = vol['to_gw'] + flux_perc_m
            vol['soil_to_gw'] = vol['soil_to_gw'] + flux_perc_m
            vol['soil_to_lat_flow'] = vol['soil_to_lat_flow'] + flux_lat_m
            vol['out'] = vol['out'] + flux_lat_m
            
            ### gw_conceptual_reservoir_flux_calc
            """
            This calculates the flux from a linear, or nonlinear
            conceptual reservoir with one or two outlets, or from an
            exponential nonlinear conceptual reservoir with only one outlet.
            In the non-exponential instance, each outlet can have its own
            activation storage threshold.  Flow from the second outlet is
            turned off by setting the discharge coeff. to 0.0.
            """
            # This is basically only running for GW, so changed the variable name from primary_flux to primary_flux_from_gw_m to avoid confusion
            # if reservoir['is_exponential'] == True:
            flux_exponential = torch.exp(
                gw_reservoir["exponent_primary"] * gw_reservoir["storage_m"]/ gw_reservoir["storage_max_m"]
                ) - torch.ones((x_conceptual.shape[0]), dtype=torch.float32, device=x_conceptual.device)
            primary_flux_from_gw_m = torch.where(
                parameters['Cgw'][:, j] * flux_exponential < gw_reservoir["storage_m"],
                parameters['Cgw'][:, j] * flux_exponential,
                gw_reservoir["storage_m"]
                )
            flux_from_deep_gw_to_chan_m = primary_flux_from_gw_m + secondary_flux_from_gw_m
            
            ### track_volume_from_gw
            gw_reservoir["storage_m"] = gw_reservoir["storage_m"] - flux_from_deep_gw_to_chan_m.clone()
            # Mass balance
            vol['from_gw'] = vol['from_gw'] + flux_from_deep_gw_to_chan_m
            vol['out'] = vol['out'] + flux_from_deep_gw_to_chan_m
            
            ####________________surface runoff routing____________
            ### convolutional_integral
            """
            This solves the convolution integral involving N GIUH ordinates.

            Inputs:
                Schaake_output_runoff_m
                num_giuh_ordinates
                giuh_ordinates
            Outputs:
                runoff_queue_m_per_timestep
            """
            
            # Set the last element in the runoff queue as zero (runoff_queue[:-1] were pushed forward in the last timestep)
            runoff_queue_m_per_timestep[:, N] = 0.0

            # Add incoming surface runoff to the runoff queue
            runoff_queue_m_per_timestep[:, :-1] = runoff_queue_m_per_timestep[:, :-1] + (
            basinCharacteristics['giuh_ordinates'] * surface_runoff_depth_m.expand(N, -1).T
            )
            # Take the top one in the runoff queue as runoff to channel
            flux_giuh_runoff_m = runoff_queue_m_per_timestep[:, 0].clone()

            # Shift all the entries forward in preperation for the next timestep
            runoff_queue_m_per_timestep[:, :-1] = runoff_queue_m_per_timestep[:, 1:].clone()

            ### track_volume_from_giuh
            vol['out_giuh'] = vol['out_giuh'] + flux_giuh_runoff_m
            vol['out'] = vol['out'] + flux_giuh_runoff_m
            
            ####_______________lateral flow routing_______________
            ### nash_cascade
            """
            Solve for the flow through the Nash cascade to delay the
            arrival of the lateral flow into the channel
            Currently only accepts the same number of nash reservoirs for all watersheds
            """
            num_reservoirs = basinCharacteristics['nash_storage'].shape[1] # 2 reservoirs
            nash_storage_timestep = basinCharacteristics['nash_storage'].clone() 
           
            # Calculate the discharge from each Nash storage
            Q = basinCharacteristics['K_nash'].unsqueeze(1) * nash_storage_timestep # first pass would be 0

            # Update Nash storage with discharge
            nash_storage_timestep = nash_storage_timestep - Q # first pass would be 0

            # The first storage receives the lateral flow outflux from soil storage
            nash_storage_timestep[:, 0] = nash_storage_timestep[:, 0] + flux_lat_m

            # The remaining storage receives the discharge from the upper Nash storage
            if num_reservoirs > 1:
                basinCharacteristics['nash_storage'][:, 1:] = basinCharacteristics['nash_storage'][:, 1:] + Q[:, :-1]

            # Update the state
            basinCharacteristics['nash_storage'] = nash_storage_timestep.clone()

            # The final discharge at the timestep from Nash cascade is from the lowermost Nash storage
            flux_nash_lateral_runoff_m= Q[:, -1].clone()

            ### track_volume_from_nash_cascade
            vol['in_nash'] = vol['in_nash'] + flux_lat_m
            vol['out_nash'] = vol['out_nash'] + flux_nash_lateral_runoff_m
            
            ### add_up_total_flux_discharge
            flux_Qout_m = flux_giuh_runoff_m + flux_nash_lateral_runoff_m + flux_from_deep_gw_to_chan_m
    
            #states[] are arbitrary for now, can be modified later.
            states['gw_reservoir_storage_m'][:,j] = gw_reservoir['storage_m']
            states['soil_reservoir_storage_m'][:,j] = soil_reservoir['storage_m']
            states['first_nash_storage'][:,j] = basinCharacteristics['nash_storage'][:,1]
            
            # out[:,j,0] = flux_Qout_m * basinCharacteristics['catchment_area_km2'][:,j] * 1000000.0/ time_step_size is incorrect
            out[:,j,0] = flux_Qout_m*1000 
            
        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}


    #______________________defining states and parameter properties relavent to NH________________
    @property
    def initial_states(self):
        return {'gw_reservoir_storage_m': 0.01,
                'soil_reservoir_storage_m': 0.6,
                'first_nash_storage': 0.0} # There are more storage/fluxes but doesn't matter cuz we can just grab whatever I want

    @property
    def parameter_ranges(self):
        return {'satdk': [0.0, 0.000726], # Saturated hydraulic conductivity
                'Cgw': [0.0000018, 0.0018], # Primary groundwater reservoir constant
                } # the only 2 parameters from NN that was from the original dCFE model was Cgw & satdk
    #___________________end of defining states and parameters relevant to NH __________________
