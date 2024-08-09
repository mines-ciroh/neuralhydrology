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
    
    Last edited: by Ziyu, 08/03/2024
    
    General outline:
    The model takes in output from a NN (like a LSTM), and forcings needed for CFE, which are just PET and Precip. 
    
    This starts out as a code-vomit of cfe.py (the physics) & bmi_cfe.py (configuration & some parameter settings) 
    into the NH framework and will follow the general formatting of shm.py + Example 02. The idea is to run this 
    dcfe model via hybrid_model. Classes defined in cfe.py & bmi_cfe.py could be referred to from __init__.
    
    This is still being updated and connected into the NH framework. 
    
    NEW: Some parameter inputs are basin-specific, this model will be tailored to basin ID: 01022500
    
    
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
        

        # initialize structures to store information on states, built into NH
        state, out =  self._initialize_information(conceptual_inputs=x_conceptual)
        
        # initialize constants for the specific basin
        # ______from bmi_config_cfe.json________
        # tensor elements of size [batch_size, time_steps], can be modified later for multi-basin
        catchment_area_km2 = 573.6 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device) # not sure where they got this from, CAMELS has slightly different num
        refkdt = 3.8266861353378374 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device)
        max_gw_storage = 0.021342666010108112 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device) # max groundwater storage [m], part of calibration
        # the below is going to be a parameter from NN
        Cgw = parameters['Cgw'] # primary groundwater nonlinear reservoir constant [m/hr], part of calibration
        expon = 6.72972972972973 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device) # primary gorundwater nonlinear reservoir exponential constant, part of calibration
        gw_storage = 0.05 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device)
        alpha_fc = 0.33 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device)
        K_nash = 0.03 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device) # Nash cascade discharge coefficient, part of calibration
        K_lf = 0.01 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device) # Lateral flow coefficient, part of calibration
        nash_storage = torch.tensor([0.0, 0.0], dtype=torch.float32, device=x_conceptual.device)
        giuh_ordinates = torch.tensor([0.93, 0.06, 0.01, 0.0, 0.0], dtype=torch.float32, device=x_conceptual.device)
        
            # ______________defining parameters that are specific to this basin, from bmi_config_cfe.json_______
        # soil_params will have tensor elements of size [batch_size, time_steps]. For non-NN parameters, they would be basin-specific
        # but right now it's just 01022500 so these constants are at every cell.
        # In the future, they can be defined differently based on different basins. 
        soil_params = {
            'depth': 2.0 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device), # not sure where they got these values, they don't match CAMELS
            'bb': 8.013513513513514 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device), # exponent on Clapp-Hornberger function, part of calibration
            # satdk is from NH
            'satdk': parameters['satdk'], # saturated hydraulic conductivity [m/hr], part of calibration
            'satpsi': 0.1647076737162162 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device), 
            'slop': 0.08824091635135137 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device), # slope coefficient, part of calibration
            'smcmax': 0.37300223004054056 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device), # maximum soil moisture content [m3/m3], part of calibration
            'wltsmc': 0.04966811960810811* torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device),
            'D': 2.0 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device),
            'mult': 1000.0 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device)
        }
        
        ###  defining soil_reservoir & gw_reservoir constants that are basin specific
        # __________modified from bmi_cfe.py____________
        ## Groundwater/subsurface reservoir (need to be batchsize x timesteps)
        gw_reservoir = {
            'storage_max_m': max_gw_storage,
            # "coeff_primary": self.Cgw, -> this has been changed to dynamic parameter
            'exponent_primary': expon,
            'storage_threshold_primary_m': 0 ,
            # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off.
            'storage_threshold_secondary_m': 0,
            'coeff_secondary': 0,
            'exponent_secondary': 1,
        }
        gw_reservoir['storage_m'] = gw_reservoir['storage_max_m'] * 0.01
        volstart = volstart.add(gw_reservoir["storage_m"])
        vol_in_gw_start = gw_reservoir["storage_m"]

        
        ## Soil Reservoir Configuration
        output_factor_cms = (1/1000) * (catchment_area_km2 * 1000 * 1000) * (1/3600) # this un-normalizes the normalized output area
        # local values to be used in setting up soil reservoir
        trigger_z_m = 0.5 * torch.ones((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32, device=conceptual_inputs.device),
        field_capacity_atm_press_fraction = alpha_fc
        # soil outflux calculation, Eq. 3
        H_water_table_m = field_capacity_atm_press_fraction * atm_press_Pa/unit_weight_water_N_per_m3
        soil_water_content_at_field_capacity = soil_params["smcmax"] * torch.pow(
            H_water_table_m / soil_params["satpsi"], (1.0 / soil_params["bb"]))
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
        assumed_near_channel_water_table_slope = 0.01  # [L/L]
        lateral_flow_threshold_storage_m = field_capacity_storage_threshold_m
        soil_reservoir_storage_deficit_m = torch.tensor([0.0], dtype=torch.float64)
        
        soil_reservoir = {
            'wilting_point_m': soil_params['wltsmc']*soil_params['D'],
            'storage_max_m': soil_params['smcmax']*soil_params['D'],
            'coeff_primary': parameters['satdk'] * soil_params['slop'] * time_step_size #Eq.11
            'exponent_primary': 1, # fixed to 1 based on Eq. 11
            'storage_threshold_primary_m': field_capacity_storage_threshold_m, # place holder for now, this is smcmax * storage_thresh_pow_term*lim_diff
            'coeff_secondary': K_lf,  # Controls lateral flow
            'exponent_secondary': 1,  # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
            'storage_threshold_secondary_m': lateral_flow_threshold_storage_m, ## but this is the same as field_capacity_storage_threshold_m??
        }
        soil_reservoir['storage_m'] = soil_reservoir['storage_max_m'] * 0.6
        volstart = volstart.add(soil_reservoir['storage_m'])
        vol_soil_start = soil_reservoir['storage_m']
        
        
        # ________some other constants_______
        # amount of seconds in 1hr
        time_step_size = 3600
        atm_press_Pa = 101325.0
        unit_weight_water_N_per_m3 = 9810.0
        
        # reset fluxes that can store information at every time-step. This will be #basin x 
        surface_runoff_depth_m = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), 
                                             dtype=torch.float32, device=conceptual_inputs.device)
        infiltration_depth_m = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), 
                                           dtype=torch.float32, device=conceptual_inputs.device)
        actual_et_from_rain_m_per_timestep = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), 
                                                         dtype=torch.float32, device=conceptual_inputs.device)
        actual_et_from_soil_m_per_timestep = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), 
                                                         dtype=torch.float32, device=conceptual_inputs.device)
        primary_flux_m = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), 
                                     dtype=torch.float32, device=conceptual_inputs.device)
        secondary_flux_m = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), 
                                       dtype=torch.float32, device=conceptual_inputs.device)
            
            
        # loop through each timestep
        for j in range(x_conceptual.shape[1]):
            # read forcings for time step
            potential_et_m_per_s = x_conceptual[;,j,1]/1000/3600 # convert PET mm/hr to [m/s]
            timestep_rainfall_input_m = x_conceptual[:,j,0]/1000 # convert precip mm/hr to [m/hr]
        
            # reset volume tracking # part of reset_volume_tracking
            vol_PET = torch.zeros((x_conceptual.shape[0], 1), dtype=torch.float32, 
                                  device=x_conceptual.device)
            
            # reset ET
            reduced_potential_et_m_per_timestep = torch.zeros((x_conceptual.shape[0], 1), dtype=torch.float32, 
                                                              device=x_conceptual.device)
            
            ####__________Rainfall and ET___________####
            
            ### calculate input rainfall and ET
            potential_et_m_per_timestep = potential_et_m_per_s * time_step_size # results in [m/hr]
            vol_PET += potential_et_m_per_timestep
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
                actual_et_from_rain_m_per_timestep[rainfall_mask, j] = actual_et_from_rain
                # should the below be states????? They are not right now.
                timestep_rainfall_input_m[rainfall_mask] = reduced_rainfall # adjusting precip based on evaporation from rainfall
                reduced_potential_et_m_per_timestep[rainfall_mask] = reduced_potential_et # adjusting pet based on evaporation from rainfall
                
                ## And track_volume_from_rainfall (should the following be states as well???)
                vol_et_from_rain += actual_et_from_rain_m_per_timestep[:,j]
                vol_et_to_atm += actual_et_from_rain_m_per_timestep[:,j]
                volout += actual_et_from_rain_m_per_timestep[:,j]
                actual_et_m_per_timestep += actual_et_from_soil_m_per_timestep[:,j]
                
            ### Calculate evaporation from soil ###
            # for this sub-module, check if the soil scheme is "classic" or "ode". If this is "ode", do nothing.
            if self.cfg.soil_scheme == "classic":
                # Creating a mask for elements where excess soil moisture > 0
                excess_sm_for_ET_mask = (soil_reservoir["storage_m"][:,j] > soil_reservoir["wilting_point_m"][:,j]) # this would be one element, Y or N
                et_mask = reduced_potential_et_m_per_timestep > 0
                 # Combine both masks
                combined_mask = et_mask & excess_sm_for_ET_mask
                # If the soil moisture storage is more than wilting point, and PET is not zero, calculate ET from soil
                if torch.any(combined_mask):
                    # calculate et_from_soil: take AET from soil moisture storage & use Budyko curve...
                    storage = soil_reservoir["storage_m"][combined_mask,j]
                    threshold = soil_reservoir["storage_threshold_primary_m"][combined_mask, j]
                    wilting_point = soil_reservoir["wilting_point_m"][combined_mask, j]
                    reduced_pet = reduced_potential_et_m_per_timestep[combined_mask, j]
                    
                    condition1 = storage >= threshold
                    condition2 = (storage > wilting_point) & (storage < threshold)
                    
                    actual_et_from_soil = torch.where(
                        condition1, 
                        torch.minimum(reduced_pet, storage), # If storage is above the FC threshold, AET = PET
                        torch.where(
                            condition2,
                            torch.minimum(
                                (storage - wilting_point)/(threshold - wilting_point) * reduced_pet,
                                storage,
                            ), # If storage is in bewteen the FC and WP threshold, calculate the Budyko type of AET
                            torch.zeros_like(storage) # If storage is less than WP, AET=0
                        )
                    )
                    
                    # update the timestep's et from soil????
                    actual_et_from_soil_m_per_timestep[combined_mask, j] = actual_et_from_soil
                    # adjust soil storage by subtracting et from soil
                    soil_reservoir["storage_m"][combined_mask, j] = soil_reservoir["storage_m"][combined_mask, j] - actual_et_from_soil
                    ###### Should below be batch_size x timestep as well??? This is initialized every timestep right now
                    reduced_potential_et_m_per_timestep[combined_mask] = reduced_potential_et_m_per_timestep[combined_mask] - actual_et_from_soil
                    
                    # and now track_volume_et_from_soil. Should the following be states as well??
                    vol_et_from_soil += actual_et_from_soil_m_per_timestep[:,j]
                    vol_et_to_atm += actual_et_from_soil_m_per_timestep[:,j]
                    volout += actual_et_from_soil_m_per_timestep[:,j]
            
        
            ####__________Infiltration partitioning________###
            
            # check 
                
                
        return {'y_hat': out, 'parameters': parameters, 'internal_states': state}


    #______________________defining states and parameter properties relavent to NH________________
    @property
    def initial_states(self):
        return {'gw_reservoir.storage_m': 0.01,
                'soil_reservoir.storage_m': 0.6} # There are more storage/fluxes but doesn't matter cuz we can just grab whatever I want

    @property
    def parameter_ranges(self):
        return {'satdk': [0.0, 0.000726], # Saturated hydraulic conductivity
                'Cgw': [0.0000018, 0.0018], # Primary groundwater reservoir constant
                } # the only 2 parameters from NN that was from the original dCFE model was Cgw & satdk
    #___________________end of defining states and parameters relevant to NH __________________
