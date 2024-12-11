# original shm packages
import torch
import torch.nn as nn
from typing import Dict, Union
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config

# packages from cfe.py
#import time
import numpy as np
import pandas as pd
#import sys
#import math
# import torch
import torch.nn.functional as F
#from torchdiffeq import odeint

# packages from bmi_cfe.py
import matplotlib.pyplot as plt
from torch import Tensor

class dCFE(BaseConceptualModel):
    """
    This is an attempt to make a dCFE model based on 
    https://github.com/NWC-CUAHSI-Summer-Institute/ngen-aridity/blob/main/Project%20Manuscript_LongForm.pdf
    
    
    Last edited: by Ziyu, 11/05/2024
    
    General outline:
    Takes raw LSTM output, shape them within possible ranges of the Cgw and satdk parameters. 
    Together with other basin-specific parameters and forcings (precip, and srad + tmean for pet) 
    and pass through the CFE for runoff predictions. 
    Spin-up period = warm_up do not have gradient tracking.  The states here are used to run one more time for prediction.
    This model is tailored to basin ID: 01022500 right now, but can be worked on 
    later to train multi-basin. 
    
    The physics is done and forward process & backward processes work so far with this specific basin and time-period of data. 
    
    TODO: 
    Debug/double check for correct physical model & magnitudes
    Improve readability
    """
    
    def __init__(self, cfg: Config):
        super(dCFE, self).__init__(cfg=cfg)
        
        self.cfg = cfg
        
        
    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Perform a forward pass in the hybrid-dCFE model. 

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. hours) that our conceptual
            model is going to be run for. The n_inputs is the number of forcing inputs. 
            x_conceptual[:, warmup, :] will be used to spinup the model. 
        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps - warmup, n_parameters]. This tensor comes from the data-driven model,
            and will be used to obtain the dynamic parameterization of the conceptual model.
            
        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow of size [batch_size, time_steps - warmup]
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model,
                right now it would be satdk and Cgw like the original dCFE proj. 
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model.
                Not currently used in the model.
        """
        # get model params thru baseconceptualmodel.py's function, 
        # this ensure that the output from NN is within the correct range, built into NH
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual, lstm_out=lstm_out)
        
        # initialize basin-specific constants
        self.initialize_basin_constants(x_conceptual)
        
        # Spin up for warm_up amount of time, do not track gradient
        with torch.no_grad():
            for j in range(0, (x_conceptual.shape[1] - lstm_out.shape[1] - 1)):
                # run the CFE model for the time step w/ first pair of params
                self.timestep_CFE(x_conceptual_timestep = x_conceptual[:,j,:], 
                                    satdk_timestep = parameters['satdk'][:, 0],
                                    cgw_timestep = parameters['Cgw'][:, 0])
                
                # store resulting states, right now this doesn't do anything
                states['gw_reservoir_storage_m'][:,j] = self.gw_reservoir['storage_m']
                states['soil_reservoir_storage_m'][:,j] = self.soil_reservoir['storage_m']
                states['first_nash_storage'][:,j] = self.basinCharacteristics['nash_storage'][:,0]
            
        # Run model for prediction for each parameters
        for k in range(lstm_out.shape[1]):
            # run CFE for that time step, w/ time-varying params
            self.timestep_CFE(x_conceptual_timestep = x_conceptual[:,(j + 1 + k),:] ,
                              satdk_timestep = parameters['satdk'][:,k],
                              cgw_timestep = parameters['Cgw'][:,k])
            
            # store states
            states['gw_reservoir_storage_m'][:,(j + 1 + k)] = self.gw_reservoir['storage_m']
            states['soil_reservoir_storage_m'][:,(j + 1 + k)] = self.soil_reservoir['storage_m']
            states['first_nash_storage'][:,(j + 1 + k)] = self.basinCharacteristics['nash_storage'][:,0]
            
            # store runoff for back-prop
            out[:,k,0] = self.flux_Qout_m*1000
            
        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}


    #______________________defining states and parameter properties relavent to NH________________
    @property
    def initial_states(self):
        return {'gw_reservoir_storage_m': 0.5,
                'soil_reservoir_storage_m': 0.6,
                'first_nash_storage': 0.0} # There are more storage/fluxes but doesn't matter cuz we can just grab whatever I want

    @property
    def parameter_ranges(self):
        return {'satdk': [0.0, 0.000726], # Saturated hydraulic conductivity
                'Cgw': [0.0000018, 0.0018], # Primary groundwater reservoir constant
                } # the only 2 parameters from NN that was from the original dCFE model was Cgw & satdk
        
        
    #___________________Defining methods for CFE for each time step__________________
    def timestep_CFE(self, x_conceptual_timestep: torch.Tensor, satdk_timestep: torch.Tensor, cgw_timestep: torch.Tensor):
        """
        INPUT: 
        x_conceptual_timestep = x_conceptual[:,j,:] forcing for every time step. 
        rainfall is x_conceptual_timestep[:,0] and srad is x_conceptual_timestep[:,1]
        """
        
        self.soil_params['satdk'] = satdk_timestep
        self.soil_reservoir['coeff_primary'] = self.soil_params['satdk']*self.soil_params['slop']*self.time_step_size # Eq.11, unit [m/s] * [3600s] now its [m/hr]
        self.basinCharacteristics['Cgw'] = cgw_timestep
        self.gw_reservoir['coeff_primary'] = cgw_timestep
            
        self.initialize_flux_timestep(x_conceptual_timestep)
        
        ####_________________Get forcings________________####
        
        self.get_precip_and_pet(x_conceptual_timestep)
        
        ####_______________Rainfall and ET________________####
        
        self.calculate_input_rainfall_and_ET()
        
        self.calculate_evaporation_from_rainfall()
        
        # TODO: if classic scheme then evaporation from soil:
        self.calculate_evaporation_from_soil()
            
        ####____________Infiltration partitioning__________####
        
        if self.schemes['partition'] == "Schaake":
            self.run_Schaake_subroutine()
        elif self.schemes['partition'] == "Xinanjiang":
            self.run_Xinanjiang_subroutine(x_conceptual_timestep)
        else:
            print(
                "Problem: must specify one of Schaake or Xinanjiang partitioning scheme."
            )
            print("Program terminating.:( \n")
        
        """TODO:
        Ask Andy about c code lines 166-179, where is flux_prec_m from when not previously defined?
        """
        
        self.adjust_and_track_runoff_infiltration()
        
        ####______________Soil moisture reservoir_____________####
        ### Start run_soil_moisture_scheme
        if self.schemes['soil'] == "classic":
           self.run_classic_soil_moisture_subroutine()
        elif self.schemes['soil'] == "ode":
            print(
                "ODE for soil scheme is not yet implemented."
            ) # we can come back and implement this later
            print("Program terminating.:( \n")
        else:
            print(
                "Either a classic or ode soil scheme must be chosen."
            )
            print("Program terminating.:( \n")
        
        self.adjust_from_soil_outflux()
        
        ####_______________groundwater reservoir________________####
  
        self.percolation_and_lateral_flow()
        
        self.calculate_gw_reservoir_flux(x_conceptual_timestep)
        
        # TODO: in c code, it is either GIUH or nash cascade. See lines 239 - 260.
        ####________________surface runoff routing______________####
        
        self.calculate_convolutional_integral_for_GIUH()
        
        ####________________lateral flow routing________________####
        # subsurface scheme:
        self.run_nash_cascade()
        
        # calculate total runoff in meters
        self.flux_Qout_m = self.flux_giuh_runoff_m + self.flux_nash_lateral_runoff_m  + self.flux_from_deep_gw_to_chan_m

    def initialize_basin_constants(self, x_conceptual: torch.Tensor):
         # ________some other constants_______
        # time-related constants
        self.time_step_size = 3600*24 # num of [seconds] per hour, we go by 3600s each time step
        self.timestep_h = self.time_step_size/3600 # time step in [hours]
        self.timestep_d = self.timestep_h/24 # time step in [days]
        # physics constants
        self.atm_press_Pa = 101325.0 # [Pa]
        self.unit_weight_water_N_per_m3 = 9810.0 # [N/m3]
        
        self.schemes = {
            'soil': 'classic', # choose between 'classic' or 'ode', 'ode' not available rn
            'partition': 'Schaake' # choose between 'Schaake' or 'Xinanjiang'
        }
        # TODO: formally soil_scheme, partition_scheme need renamed. Eventually move to config?
        
        # ___________Basin Specific Information___________
        # key for basinChracteristics:
        # - catchment_area_km2: tensor, [batch_size, timestep]. Area of the basin [km2]
        # - redfdk: tensor, [batch_size, timestep]. Runoff partitioning parameter [-] 
        # - max_gw_storage: tensor, [batch_size, timestep]. Max groundwater storage [m]. Part of CFE calibration
        # - expon: tensor, [batch_size, timestep]. A primary groundwater nonlinear reservoir exponential constant [-]. Part of CFE calibration
        # - alpha_fc: tensor, [batch_size, timestep]. 
        # - K_nash: tensor, [batch_size, timestep]. Nash cascade discharge coefficient [-]. Part of CFE calibration
        # - K_lf: tensor, [batch_size, timestep]. Lateral flow coefficient [-]. Part of CFE calibration
        # - nash_storage: tensor, [batch_size, 2]. 2 columns for 2 "buckets" of Nash cascade
        # - giuh_ordinates: tensor, [num_coordinates]. 
        self.basinCharacteristics = {
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
            'giuh_ordinates': torch.tensor([0.93, 0.06, 0.1, 0.0, 0.0], dtype=torch.float32, device=x_conceptual.device)
        } 
        
        self.soil_params = {
            'depth': 2.0 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # not sure where they got these values, they don't match CAMELS, [m]
            'bb': 8.013513513513514 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # exponent on Clapp-Hornberger function, part of calibration
            # satdk is from NH, define this in loop
            #'satdk': parameters['satdk'], # saturated hydraulic conductivity [m/hr], part of calibration
            'satpsi': 0.1647076737162162 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), 
            'slop': 0.08824091635135137 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # slope coefficient, part of calibration
            'smcmax': 0.37300223004054056 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]), # maximum soil moisture content [m3/m3], part of calibration
            'wltsmc': 0.04966811960810811* torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'D': 2.0 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'mult': 1000.0 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
        }
        
        self.gw_reservoir = {
            'storage_max_m': self.basinCharacteristics['max_gw_storage'],
            # "coeff_primary": self.Cgw, -> this has been changed to dynamic parameter, will be defined in loop
            'exponent_primary': self.basinCharacteristics['expon'],
            'storage_threshold_primary_m': 0 ,
            # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off.
            'storage_threshold_secondary_m': 0,
            'coeff_secondary': 0,
            'exponent_secondary': 1,
        }
        self.gw_reservoir['storage_m'] = self.gw_reservoir['storage_max_m'].clone() * 0.9 #0.5 was sweet spot before
        
        ## Soil Reservoir Configuration
        # local values to be used in setting up soil reservoir
        self.trigger_z_m = 0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
        self.field_capacity_atm_press_fraction = self.basinCharacteristics['alpha_fc']
        # soil outflux calculation, Eq. 3
        self.H_water_table_m = self.field_capacity_atm_press_fraction * self.atm_press_Pa/self.unit_weight_water_N_per_m3 # [m]
        self.Omega = self.H_water_table_m - self.trigger_z_m
        # upper & lower limit of the integral in Eq. 4
        self.lower_lim = torch.pow(self.Omega, (1.0 - 1.0 / self.soil_params["bb"])) / (
            1.0 - 1.0 / self.soil_params["bb"])
        self.upper_lim = torch.pow(self.Omega + self.soil_params["D"], (1.0 - 1.0 / self.soil_params["bb"])) / (1.0 - 1.0 / self.soil_params["bb"])
        # integral & power term in Eq.4, Eq.5
        self.storage_thresh_pow_term = torch.pow(1.0 / self.soil_params["satpsi"], (-1.0 / self.soil_params["bb"]))
        self.lim_diff = self.upper_lim - self.lower_lim
        self.field_capacity_storage_threshold_m = (self.soil_params["smcmax"] * self.storage_thresh_pow_term * self.lim_diff)
        # lateral flow function parameters
        self.lateral_flow_threshold_storage_m = self.field_capacity_storage_threshold_m
        
        self.soil_reservoir = {
            'wilting_point_m': self.soil_params['wltsmc']*self.soil_params['D'], #0.049668*2 = 0.09933
            'storage_max_m': self.soil_params['smcmax']*self.soil_params['D'], #0.373*2 = 0.746
            #'coeff_primary': parameters['satdk'] * soil_params['slop'].unsqueeze(1)*time_step_size, #Eq.11, unit [m/s] * [3600s] now its [m/hr]. Define this in loop
            'exponent_primary': 1.0, # fixed to 1 based on Eq. 11
            'storage_threshold_primary_m': self.field_capacity_storage_threshold_m, 
            'coeff_secondary': self.basinCharacteristics['K_lf'],  # Controls lateral flow
            'exponent_secondary': 1.0,  # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
            'storage_threshold_secondary_m': self.lateral_flow_threshold_storage_m, ## but this is the same as field_capacity_storage_threshold_m??
        }
        self.soil_reservoir['storage_m'] = self.soil_reservoir['storage_max_m'].clone() * 0.9 #factor was 0.6 before
        
        self.N = self.basinCharacteristics['giuh_ordinates'].shape[0] # giuh_ordinates are rows x 1 column for each basin, used in routing
        self.runoff_queue_m_per_timestep = torch.ones((x_conceptual.shape[0], self.N + 1), dtype=torch.float32, device=x_conceptual.device) # nash cascade
        self.num_reservoirs = self.basinCharacteristics['nash_storage'].shape[1] # 2 reservoirs
        
        self.vol = {
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
            'out': torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0]),
            'gw_start': self.gw_reservoir["storage_m"],
            'soil_start': self.soil_reservoir['storage_m']
        }
        
        self.flux_perc_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
    
    def initialize_flux_timestep(self, x_conceptual_timestep: torch.Tensor):
        # reset fluxes that can store information at every time-step. This will be #basin x 
        self.surface_runoff_depth_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.infiltration_depth_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.actual_et_from_rain_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.actual_et_from_soil_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.actual_et_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0]) # need to initialize every Epoch?
        # reset ET
        self.reduced_potential_et_m_per_timestep = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        
        self.primary_flux_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.secondary_flux_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        # below are all added later, not in original initialization
        self.primary_flux_from_gw_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.secondary_flux_from_gw_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.flux_giuh_runoff_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.flux_nash_lateral_runoff_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.flux_from_deep_gw_to_chan_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        self.tension_water_m = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0]) # 'Xinanjiang' partition
        
    def get_precip_and_pet(self, x_conceptual_timestep):
        # convert Precip from [mm/day] to [m/day], first conceptual_input must be precip
        self.timestep_rainfall_input_m = x_conceptual_timestep[:,0]/1000 # convert precip mm/day to [m/day]
        
        # calculate PET from shortwave rad and mean temp using jensen_evaporation_2016 "https://github.com/pyet-org/pyet/blob/master/pyet/radiation.py"
        mean_temp = (x_conceptual_timestep[:,1] + x_conceptual_timestep[:, 2])/2 # x_conceptual[:,:,1] is Tmin, x_conceptual[:,:,2] is Tmax
        lambd = 2.501 - 0.002361 * mean_temp
        shortRad = x_conceptual_timestep[:,3]*3600*24/1000000 # convert shortwave radiation [W/m^2] to [MJ/m^2 day]
        pet_m_per_timestep_calc = (0.025 * shortRad * (mean_temp - (-3.0))/lambd)/1000 # convert pet [mm/day] to [m/day]
        pet_m_per_timestep_mask = pet_m_per_timestep_calc < 0 # make mask for negative PET
        self.potential_et_m_per_timestep = torch.where(pet_m_per_timestep_mask, 0, pet_m_per_timestep_calc) # clip negative PET to 0
    
        #__________hourly below__________
        # convert Precip from [mm/hr] to [m/hr], first conceptual_input must be precip
        # self.timestep_rainfall_input_m = x_conceptual_timestep[:,0]/1000 # convert precip mm/hr to [m/hr]
        
        # calculate PET from shortwave rad and mean temp using jensen_evaporation_2016 "https://github.com/pyet-org/pyet/blob/master/pyet/radiation.py"
        # lambd = 2.501 - 0.002361 * x_conceptual_timestep[:,1] # x_conceptual[:,:,1] is mean temp
        # shortRad = x_conceptual_timestep[:,2]*3600/1000000 # convert shortwave radiation [W/m^2] to [MJ/m^2 hr]
        # pet_m_per_timestep_calc = (0.025 * shortRad * (x_conceptual_timestep[:,1] - (-3.0))/lambd)/1000 # convert pet [mm/hr] to [m/hr]
        # pet_m_per_timestep_mask = pet_m_per_timestep_calc < 0 # make mask for negative PET
        # self.potential_et_m_per_timestep = torch.where(pet_m_per_timestep_mask, 0, pet_m_per_timestep_calc) # clip negative PET to 0
    
    def calculate_input_rainfall_and_ET(self):
        """
        Grabs self.potential_et_m_per_timestep and stores them in:
        self.vol['PET']
        self.reduced_potential_et_m_per_timestep
        """
        # potential_et_m_per_timestep = potential_et_m_per_timestep.view(-1,1) # this was not used
        self.vol['PET'] = self.vol['PET'] + self.potential_et_m_per_timestep
        self.reduced_potential_et_m_per_timestep = self.potential_et_m_per_timestep 
    
    def calculate_evaporation_from_rainfall(self):
        """
        This method calculates evapotransporation (ET) from rainfall and track/adjust those fluxes as:
        self.actual_et_from_rain_m_per_timestep
        self.timestep_rainfall_input_m
        self.reduced_potential_et_m_per_timestep
        
        self.vol['et_from_rain']
        self.vol['et_to_atm']
        self.vol['out']
        self.actual_et_m_per_timestep # not actually used anywhere
        """
        ### calculate evaporation from rainfall
        # actual_et_from_rain_m_per_timestep was initialized here to be 0
        # Creating a mask for elements where timestep_rainfall_input_m > 0
        rainfall_mask = self.timestep_rainfall_input_m > 0 
        if torch.any(rainfall_mask): # if there's precip in this batch then
            
            ## Calculate et_from_rainfall
            # apply the mask for when there's rainfall
            rainfall = self.timestep_rainfall_input_m[rainfall_mask]
            pet = self.potential_et_m_per_timestep[rainfall_mask]
            
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
            
            reduced_potential_et = pet - actual_et_from_rain
        
            # storing results back to the states
            self.actual_et_from_rain_m_per_timestep[rainfall_mask] = actual_et_from_rain
            # should the below be states????? They are not right now.
            self.timestep_rainfall_input_m[rainfall_mask] = reduced_rainfall # adjusting precip based on evaporation from rainfall
            self.reduced_potential_et_m_per_timestep[rainfall_mask] = reduced_potential_et # adjusting pet based on evaporation from rainfall
            
            ## And track_volume_from_rainfall (should the following be states as well???)
            self.vol['et_from_rain'] = self.vol['et_from_rain'] + self.actual_et_from_rain_m_per_timestep
            self.vol['et_to_atm'] = self.vol['et_to_atm'] + self.actual_et_from_rain_m_per_timestep
            self.vol['out'] = self.vol['out'] + self.actual_et_from_rain_m_per_timestep
            self.actual_et_m_per_timestep = self.actual_et_m_per_timestep + self.actual_et_from_rain_m_per_timestep # actual_et_m_per_timestep not used anywhere
            
    def calculate_evaporation_from_soil(self):
        """ code modifed by Ziyu
        For this sub-module, check if the soil scheme is "classic" or "ode". If this is "ode", do nothing.
        Fluxes/adjustments as a result are:
        self.soil_reservoir["storage_m"]
        self.reduced_potential_et_m_per_timestep
        self.vol['et_from_soil']
        self.vol['et_to_atm']
        self.vol['out']
        self.actual_et_m_per_timestep
        """
        ### Calculate evaporation from soil ###
        if self.schemes['soil'] == "classic":
            # Creating mask for elements where adjusted PET > 0
            et_mask = (self.reduced_potential_et_m_per_timestep > 0) # adjusted PET
            # Creating mask for elements where excess soil moisture > 0; soil sotrage > wilting point
            excess_sm_for_ET_mask = (self.soil_reservoir["storage_m"] > self.soil_reservoir["wilting_point_m"])
            # Creating mask for elements where soil storage >= soil storage_threshold_primary_m
            excess_sm_primary_threshold = (self.soil_reservoir["storage_m"] >= self.soil_reservoir['storage_threshold_primary_m'])
            # Creating mask for elements where soil storage < soil storage_threshold_primary_m
            # deficit_sm_primary_threshold = ~excess_sm_for_ET_mask
            
            # Combine both masks where PET > 0, and soil storage >= storage_threshold_primary
            combined_excess_primary_mask = et_mask & excess_sm_primary_threshold 
            # Combine both masks where PET > 0, and soil storage > wilting point, and soil storage < storage threshold primary
            combined_excess_ET_and_deficit_primary_mask = et_mask & excess_sm_for_ET_mask & ~excess_sm_primary_threshold
            # If the soil moisture storage is more than wilting point, and PET is not zero, calculate ET from soil
            if torch.any(combined_excess_primary_mask): # if there's PET & storage >= threshold primary
                self.actual_et_from_soil_m_per_timestep[combined_excess_primary_mask] = torch.where(
                    self.reduced_potential_et_m_per_timestep[combined_excess_primary_mask] < self.soil_reservoir["storage_m"][combined_excess_primary_mask],
                    self.reduced_potential_et_m_per_timestep[combined_excess_primary_mask],
                    self.soil_reservoir["storage_m"][combined_excess_primary_mask]
                ) 
            # equivalent to min(reduced_potential_et_m_per_timestep[combined_excess_primary_mask], soil_reservoir["storage_m"])
            if torch.any(combined_excess_ET_and_deficit_primary_mask): #check PET>0, storage>wilting, and storage < primary
                Budyko_numerator = (self.soil_reservoir["storage_m"] - self.soil_reservoir["wilting_point_m"])[combined_excess_ET_and_deficit_primary_mask]
                Budyko_denominator = (self.soil_reservoir["storage_threshold_primary_m"] - self.soil_reservoir["wilting_point_m"])[combined_excess_ET_and_deficit_primary_mask]
                Budyko = Budyko_numerator/Budyko_denominator # calculate the constant
                
                self.actual_et_from_soil_m_per_timestep[combined_excess_ET_and_deficit_primary_mask] = torch.where(
                    Budyko * self.reduced_potential_et_m_per_timestep[combined_excess_ET_and_deficit_primary_mask] < self.soil_reservoir["storage_m"][combined_excess_ET_and_deficit_primary_mask],
                    Budyko * self.reduced_potential_et_m_per_timestep[combined_excess_ET_and_deficit_primary_mask],
                    self.soil_reservoir["storage_m"][combined_excess_ET_and_deficit_primary_mask]
                ) #equivalent to min(Budyko*PET, soil storage)
            
            # adjust soil storage w/ ET from soil
            self.soil_reservoir["storage_m"] = self.soil_reservoir["storage_m"] - self.actual_et_from_soil_m_per_timestep
            # adjust PET w/ ET from soil
            self.reduced_potential_et_m_per_timestep = self.reduced_potential_et_m_per_timestep - self.actual_et_from_soil_m_per_timestep
            """ Original code for above from dCFE
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
            """    
            # and now track_volume_et_from_soil. Should the following be states as well??
            self.vol['et_from_soil'] = self.vol['et_from_soil'] + self.actual_et_from_soil_m_per_timestep
            self.vol['et_to_atm'] = self.vol['et_to_atm'] + self.actual_et_from_soil_m_per_timestep
            self.vol['out'] = self.vol['out'] + self.actual_et_from_soil_m_per_timestep
            self.actual_et_m_per_timestep = self.actual_et_m_per_timestep + self.actual_et_from_soil_m_per_timestep

    def run_Schaake_subroutine(self):
        """ This routine was modified by Ziyu
        Calculate infiltration & runoff based on the Schaake scheme. Modifies:
        self.surface_runoff_depth_m
        self.infiltration_depth_m
        """
        
        """copied from cfe.py
        if (timestep_rainfall_m > 0){
           if(soil_reservoir_storage_deficit_m < 0){
               infilt_excess_m = timestep_rainfall_m
               infilt_depth_m = 0}
           else{ 
               Schaake Eq2, Px = timestep_rainfall_m, infilt_depth_m = (Px*(Ic/(Px+Ic)))
               if (timestep_rainfall_m - infilt_depth_m > 0){
                   infilt_excess_m = timestep_rainfall_m - infilt_depth_m
                }
               else{infilt_excess_m = 0, infilt_depth_m = timestep_rainfall_m - infilt_excess_m}
        else{ infilt_excess_m = 0, infilt_depth_m = 0}
         assume factor = 1, ice_frac_Schaake not > 1e-2 (skip frozen soil module)
         infilt_depth_m = factor * infilt_depth_m
         infilt_excess_m = timestep_rainfall_m - infilt_depth_m
        """
        self.soil_reservoir_storage_deficit_m = (
            self.soil_params["smcmax"] * self.soil_params["D"]
            - self.soil_reservoir["storage_m"]
        )
        self.Schaake_adjusted_magic_constant_by_soil_type = self.basinCharacteristics['refkdt'] * self.soil_params['satdk']/ 2.0e-06 
        
        rainfall_mask = self.timestep_rainfall_input_m > 0
        soil_noDeficit_mask = (self.soil_reservoir_storage_deficit_m < 0) # mark ones w/o deficit
        soil_noDeficit_rain_mask = (rainfall_mask & soil_noDeficit_mask)
        soil_deficit_rain_mask =  (rainfall_mask & ~soil_noDeficit_mask)
        
        if torch.any(rainfall_mask):
            # For soil_reservoir_storage_deficit_m < 0, excess = rain and depth = 0
            self.surface_runoff_depth_m[soil_noDeficit_rain_mask] = self.timestep_rainfall_input_m[soil_noDeficit_rain_mask]
            # Did not put in infiltration_depth_m as they are 0 in this case
            
            # For soil_reservoir_storage_deficit_m >= 0
            Schaake_parenthetical_term = (1 - torch.exp(- self.Schaake_adjusted_magic_constant_by_soil_type * self.timestep_d))
            Ic = self.soil_reservoir_storage_deficit_m * Schaake_parenthetical_term
            Px = self.timestep_rainfall_input_m
            self.infiltration_depth_m[soil_deficit_rain_mask] = (Px*(Ic/(Px + Ic)))[soil_deficit_rain_mask]

            # From vector above, make another condition
            soil_excess_mask = (self.timestep_rainfall_input_m - self.infiltration_depth_m > 0) # mask for if rainfall is more than infilt depth
            combined_soil_excess_mask = (soil_deficit_rain_mask & soil_excess_mask) # mask for above chunk + excess rain
            combined_soil_noExcess_mask = (soil_deficit_rain_mask & ~soil_excess_mask) # mask for above chunk + no excess rain
            self.surface_runoff_depth_m[combined_soil_excess_mask] = (self.timestep_rainfall_input_m - self.infiltration_depth_m)[combined_soil_excess_mask]
            # not written else surface_runoff_depth_m = 0, since initialzied at 0
            self.infiltration_depth_m[combined_soil_noExcess_mask] = (self.timestep_rainfall_input_m - self.surface_runoff_depth_m)[combined_soil_noExcess_mask]
            # not writtien, if no rainfall, surface_runoff_depth_m = 0 and infiltration_depth_m = 0 since initialized at 0
        """ Original notes
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
        """

    def run_Xinanjiang_subroutine(self, x_conceptual_timestep):
        """
        TODO: Need to verify with c code
        Calculate infiltration & runoff based on the Xinanjiang scheme. Modifies:
        self.surface_runoff_depth_m
        self.infiltration_depth_m
        """
        
        """ Original notes
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
        free_water_m = self.soil_reservoir["storage_m"] - self.soil_reservoir["storage_threshold_primary_m"] # this storage was adjusted for ET

        # create mask for when free_water_m > 0
        free_water0_mask = free_water_m > 0.0
        free_water_m[~free_water0_mask] = 0.0  # set 0 or negative values to 0
        if torch.any(free_water0_mask):
            tension_water_m = torch.where(
                free_water0_mask,
                self.soil_reservoir["storage_threshold_primary_m"],
                self.soil_reservoir["storage_m"]
                )
        
        # estimate the maximum free water and tension water available in the soil column
        max_free_water_m = self.soil_reservoir["storage_max_m"] - self.soil_reservoir["storage_threshold_primary_m"]
        max_tension_water_m = self.soil_reservoir["storage_threshold_primary_m"]
        
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
        a_Xinanjiang_inflection_point_parameter = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        b_Xinanjiang_shape_parameter = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        x_Xinanjiang_shape_parameter = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
        
        tension_frac_mask = (tension_water_m / max_tension_water_m) <= (
            0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
            - a_Xinanjiang_inflection_point_parameter
        )
        
        pervious_runoff_m = torch.where(
            tension_frac_mask,
            (self.timestep_rainfall_input_m * torch.pow(
                    (
                        0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                        - a_Xinanjiang_inflection_point_parameter
                    ),
                    (
                    torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                    - b_Xinanjiang_shape_parameter
                    ),
                )
                * torch.pow(
                    (
                        torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                        - (tension_water_m / max_tension_water_m)
                    ),
                    b_Xinanjiang_shape_parameter,
                )
            ),
            self.timestep_rainfall_input_m * (
                torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                - torch.pow(
                    (
                        0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                        + a_Xinanjiang_inflection_point_parameter
                    ),
                    (
                        torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                        - b_Xinanjiang_shape_parameter
                    ),
                )
                * torch.pow(
                    (
                        torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                        - (tension_water_m / max_tension_water_m)
                    ),
                    (b_Xinanjiang_shape_parameter),
                )
            )  
        )
            
        # Separate the surface water from the pervious runoff
        ## NOTE: If impervious runoff is added to this subroutine, impervious runoff should be added to
        ## the surface_runoff_depth_m.
        
        self.surface_runoff_depth_m = pervious_runoff_m * (
            0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
            - torch.pow(
                (
                    0.5 * torch.tensor(1.0, dtype=torch.float32, device=x_conceptual_timestep.device).repeat(x_conceptual_timestep.shape[0])
                    - (free_water_m / max_free_water_m)
                ),
                x_Xinanjiang_shape_parameter,
            )
        )
        
        # The surface runoff depth is bounded by a minimum of 0 and a maximum of the water input depth.
        # Check that the estimated surface runoff is not less than 0.0 and if so, change the value to 0.0.
        surface_runoff0_mask = self.surface_runoff_depth_m < 0.0 # we have a vector instead of scalar
        if torch.any(surface_runoff0_mask):
            self.surface_runoff_depth_m[surface_runoff0_mask] = torch.zeros(
                (torch.sum(surface_runoff0_mask), 1), 
                dtype=torch.float32, device=x_conceptual_timestep.device
            )
        
        # Check that the estimated surface runoff does not exceed the amount of water input to the soil surface.  If it does,
        # change the surface water runoff value to the water input depth.
        surface_runoff_rainfall_mask = self.surface_runoff_depth_m > self.timestep_rainfall_input_m
        if torch.any(surface_runoff_rainfall_mask):
            self.surface_runoff_depth_m[surface_runoff_rainfall_mask] = self.timestep_rainfall_input_m[surface_runoff_rainfall_mask]
            
        # Separate the infiltration from the total water input depth to the soil surface.
        self.infiltration_depth_m = self.timestep_rainfall_input_m - self.surface_runoff_depth_m
        ##### infiltration_depth_m has been initialized at 2 different spots, need to decide where to put them. 
    
    def adjust_and_track_runoff_infiltration(self):
        """ Modified by Ziyu
        This modifies the following based on either soil subroutine:
        self.surface_runoff_depth_m
        self.infiltration_depth_m
        self.soil_reservoir["storage_m"]
        self.soil_reservoir_storage_deficit_m
        self.vol['partition_runoff']
        self.vol['partition_infilt']
        self.vol['to_soil']
        """
        ### adjust_runoff_and_infiltration
        """Calculates saturation excess overland flow (SOF)
        This should be run after calculate_infiltration_excess_overland_flow, then,
        infiltration_depth_m and surface_runoff_depth_m get finalized
        """
        # If the infiltration is more than the soil moisture deficit,
        # additional runoff (SOF) occurs and soil get saturated
        
        # Creating a mask where soil deficit is less than infiltration
        excess_infil_mask = self.soil_reservoir_storage_deficit_m < self.infiltration_depth_m
        
        # If there are any such basins, we apply the conditional logic element-wise
        if torch.any(excess_infil_mask):
            diff = (self.infiltration_depth_m - self.soil_reservoir_storage_deficit_m)[excess_infil_mask]
            # Adjusting the surface runoff and infiltration depths for the specific basins
            self.surface_runoff_depth_m[excess_infil_mask] = self.surface_runoff_depth_m[excess_infil_mask] + diff
            self.infiltration_depth_m[excess_infil_mask] = self.soil_reservoir_storage_deficit_m[excess_infil_mask]
            
            # This was missing from original implementation, added by Ziyu 11/11/24 from c code line 142
            self.soil_reservoir["storage_m"][excess_infil_mask] = self.soil_reservoir["storage_max_m"][excess_infil_mask]
            # Setting the soil reservoir storage deficit to zero for the specific basins
            self.soil_reservoir_storage_deficit_m[excess_infil_mask] = 0.0
        
        ## track_infiltration_and_runoff
        """Tracking runoff & infiltraiton volume with final infiltration & runoff values"""
        self.vol['partition_runoff'] = self.vol['partition_runoff'] + self.surface_runoff_depth_m
        self.vol['partition_infilt'] = self.vol['partition_infilt'] + self.infiltration_depth_m
        self.vol['to_soil'] = self.vol['to_soil'] + self.infiltration_depth_m


    def run_classic_soil_moisture_subroutine(self):
        """ Modified by Ziyu
        Soil moisture scheme using the classic (difference) method. 
        Lines 317 of original author code, equivalent to conceptual_reservoir_flux_calc. It modifies/creates:
        self.soil_reservoir["storage_m"]
        self.primary_flux_m
        self.secondary_flux_m
        """
        # Added on 11/20/2024
        mask_perc_soil = self.flux_perc_m > self.soil_reservoir_storage_deficit_m
        if torch.any(mask_perc_soil):
            diff_perc_soil = self.flux_perc_m[mask_perc_soil] - self.soil_reservoir_storage_deficit_m[mask_perc_soil]
            self.infiltration_depth_m[mask_perc_soil] = self.soil_reservoir_storage_deficit_m[mask_perc_soil]
        #    # not added: lines 170 & 171 send flow back to giuh in self.vol & correct over-prediction of infilt
            self.surface_runoff_depth_m[mask_perc_soil] = self.surface_runoff_depth_m[mask_perc_soil] + diff_perc_soil[mask_perc_soil]
            self.soil_reservoir_storage_deficit_m[mask_perc_soil] = 0
        
        
        # Assumes we don't have a single outlet exponential gw storage...
        # Add infiltration flux and calculate the reservoir flux
        # this is adjusted for ET already (not sure where this is from)
        self.soil_reservoir["storage_m"] = self.soil_reservoir["storage_m"] + self.infiltration_depth_m
        
        ## do soil_conceptual_reservoir_flux_calc
        # Calculate primary flux
        storage_above_threshold_primary = self.soil_reservoir["storage_m"] - self.soil_reservoir["storage_threshold_primary_m"]
        primary_flux_mask = storage_above_threshold_primary > 0.0
        if torch.any(primary_flux_mask):
            storage_diff_primary = self.soil_reservoir["storage_max_m"] - self.soil_reservoir["storage_threshold_primary_m"]
            storage_ratio_primary = storage_above_threshold_primary / storage_diff_primary
            storage_power_primary = torch.pow(storage_ratio_primary, self.soil_reservoir["exponent_primary"]) # "exponent primary" is scalar for now but can try to init as tensor
            primary_flux = self.soil_reservoir["coeff_primary"] * storage_power_primary
            self.primary_flux_m[primary_flux_mask] = torch.where(
                primary_flux < storage_above_threshold_primary, 
                primary_flux, 
                storage_above_threshold_primary)[primary_flux_mask]
            
        # Calculate secondary flux
        storage_above_threshold_secondary = self.soil_reservoir["storage_m"] - self.soil_reservoir["storage_threshold_secondary_m"]
        secondary_flux_mask = storage_above_threshold_secondary > 0.0
        if torch.any(secondary_flux_mask):
            storage_diff_secondary = self.soil_reservoir["storage_max_m"] - self.soil_reservoir["storage_threshold_secondary_m"]
            storage_ratio_secondary = storage_above_threshold_secondary / storage_diff_secondary
            storage_power_secondary = torch.pow(storage_ratio_secondary, self.soil_reservoir["exponent_secondary"]) # "exponent_secondary" is also a scalar for now
            secondary_flux = self.soil_reservoir["coeff_secondary"] * storage_power_secondary
            self.secondary_flux_m[secondary_flux_mask] = torch.where(
                secondary_flux < (storage_above_threshold_secondary - self.primary_flux_m),
                secondary_flux,
                storage_above_threshold_secondary - self.primary_flux_m)[secondary_flux_mask]

# TODO: Check this below. Consider moving soil reservoir storage - flux_perc_m, etc 
# to percolation_and_lateral_flow(self)
    def adjust_from_soil_outflux(self):
        """
        modifies
        self.soil_reservoir['storage_m']
        flux_perc_m
        flux_lat_m
        """
        self.flux_perc_m = self.primary_flux_m  # percolation_flux
        self.flux_lat_m = self.secondary_flux_m # lateral_flux
        
        ## Below is from 208-209
        # If the soil moisture scheme is classic, take out the outflux from soil moisture storage
        # If ODE, outfluxes are already subtracted from the soil moisture storage
        if self.schemes['soil'] == "classic":
            self.soil_reservoir['storage_m'] = self.soil_reservoir['storage_m'] - self.flux_perc_m
            self.soil_reservoir["storage_m"]= self.soil_reservoir["storage_m"]- self.flux_lat_m
            # We can probably combine both above
        elif self.schemes['soil'] == "ode":
            print(
                "ODE for soil scheme is not yet implemented, and storage cannot be stored."
            ) # we can come back and implement this later
            print("Program terminating.:( \n")

# TODO: Check this below. Instead of storage = storage_max, consider doing 
# whole vector storage = storage + flux_perc_m
    def percolation_and_lateral_flow(self):
        """
        Corresponds to lines 194 - 214 in cfe.c
        Calculates soil gw flux from percolation & lateral flux and modifies:
        self.surface_runoff_depth_m
        self.flux_perc_m
        self.gw_reservoir["storage_m"]
        self.vol['partition_runoff']
        self.vol['partition_infilt']
        self.vol['to_gw']
        self.vol['soil_to_gw']
        self.vol['soil_to_lat_flow']
        self.vol['out']
        """
        ### calculate_groundwater_storage_deficit
        gw_reservoir_storage_deficit_m = self.gw_reservoir["storage_max_m"]- self.gw_reservoir["storage_m"]
        ### adjust_precolation_to_gw
        overflow_mask = self.flux_perc_m > gw_reservoir_storage_deficit_m
        
        # When the groundwater storage is full, the overflowing amount goes to direct runoff
        if torch.any(overflow_mask):
            # Calculate the amount of overflow
            diff = (self.flux_perc_m - gw_reservoir_storage_deficit_m)[overflow_mask].clone()
            # there's another variable previously named as diff, maybe we should choose a better name?

            # Overflow goes to surface runoff ## not sure where this is from.
            self.surface_runoff_depth_m[overflow_mask] = self.surface_runoff_depth_m[overflow_mask] + diff

            # Reduce the infiltration (maximum possible flux_perc_m is equal to gw_reservoir_storage_deficit_m)
            self.flux_perc_m[overflow_mask] = gw_reservoir_storage_deficit_m[overflow_mask].clone()

            # Saturate the Groundwater storage # I believe storage + flux_prec_m saturated = storage max
            self.gw_reservoir["storage_m"][overflow_mask] = self.gw_reservoir["storage_max_m"][overflow_mask].clone()
            #^^^^ this is redundant, we could just do storage + flux_perc for whole vector
            gw_reservoir_storage_deficit_m[overflow_mask] = 0.0

            # Track volume
            self.vol['partition_runoff'][overflow_mask] = self.vol['partition_runoff'][overflow_mask] + diff
            self.vol['partition_infilt'][overflow_mask] = self.vol['partition_infilt'][overflow_mask] + diff
            
        # Otherwise all the percolation flux goes to the storage
        # Apply the "otherwise" part of your condition, to all basins where overflow_mask is False
        no_overflow_mask = ~overflow_mask
        if torch.any(no_overflow_mask):
            self.gw_reservoir["storage_m"][no_overflow_mask] = self.gw_reservoir["storage_m"][no_overflow_mask] + self.flux_perc_m[no_overflow_mask].clone()
        
        ### track_volume_from_percolation_and_lateral_flow
        self.vol['to_gw'] = self.vol['to_gw'] + self.flux_perc_m
        self.vol['soil_to_gw'] = self.vol['soil_to_gw'] + self.flux_perc_m
        self.vol['soil_to_lat_flow'] = self.vol['soil_to_lat_flow'] + self.flux_lat_m
        self.vol['out'] = self.vol['out'] + self.flux_lat_m

# TODO: Should be same type calculation as run_classic_soil_moisture_scheme
# Maybe consider reformatting for consistency?
    def calculate_gw_reservoir_flux(self, x_conceptual_timestep: torch.Tensor):
        """
        corresponds to lines 214 - 224
        Calculates flux from conceptual grownd water reservoir, and modifies:
        self.gw_reservoir["storage_m"]
        self.vol['from_gw']
        self.vol['out']
        """
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
            self.gw_reservoir["exponent_primary"] * self.gw_reservoir["storage_m"]/ self.gw_reservoir["storage_max_m"]
            ) - torch.ones((x_conceptual_timestep.shape[0]), dtype=torch.float32, device=x_conceptual_timestep.device)
        self.primary_flux_from_gw_m = torch.where(
            self.basinCharacteristics['Cgw'] * flux_exponential < self.gw_reservoir["storage_m"],
            self.basinCharacteristics['Cgw'] * flux_exponential,
            self.gw_reservoir["storage_m"]
            )
        self.flux_from_deep_gw_to_chan_m = self.primary_flux_from_gw_m + self.secondary_flux_from_gw_m # there's no 2nd flux since exponential
        
        ### track_volume_from_gw
        self.gw_reservoir["storage_m"] = self.gw_reservoir["storage_m"] - self.flux_from_deep_gw_to_chan_m.clone()
        # missing adjustments to flux_from_deep_gw_to_chan_m, maybe not needed but mass balance would be incorrect
        # Mass balance
        self.vol['from_gw'] = self.vol['from_gw'] + self.flux_from_deep_gw_to_chan_m
        self.vol['out'] = self.vol['out'] + self.flux_from_deep_gw_to_chan_m
    
    def calculate_convolutional_integral_for_GIUH(self):
        """
        Calculates runoff from GIUH, modifies:
        self.flux_giuh_runoff_m
        self.runoff_queue_m_per_timestep
        self.vol['out_giuh']
        self.vol['out']
        """
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
        self.runoff_queue_m_per_timestep[:, self.N] = 0.0

        # Add incoming surface runoff to the runoff queue
        self.runoff_queue_m_per_timestep[:, :-1] = self.runoff_queue_m_per_timestep[:, :-1] + (
        self.basinCharacteristics['giuh_ordinates'] * self.surface_runoff_depth_m.expand(self.N, -1).T
        )
        
        # Take the top one in the runoff queue as runoff to channel
        self.flux_giuh_runoff_m = self.runoff_queue_m_per_timestep[:, 0].clone()

        # Shift all the entries forward in preperation for the next timestep
        self.runoff_queue_m_per_timestep[:, :-1] = self.runoff_queue_m_per_timestep[:, 1:].clone()

        ### track_volume_from_giuh
        self.vol['out_giuh'] = self.vol['out_giuh'] + self.flux_giuh_runoff_m
        self.vol['out'] = self.vol['out'] + self.flux_giuh_runoff_m

    def run_nash_cascade(self):
        """
        Runs the nash cascade scheme and modifies:
        self.basinCharacteristics['nash_storage']
        self.flux_nash_lateral_runoff_m
        self.vol['in_nash']
        self.vol['out_nash']
        """
        ### nash_cascade
        """
        Solve for the flow through the Nash cascade to delay the
        arrival of the lateral flow into the channel
        Currently only accepts the same number of nash reservoirs for all watersheds
        """
        nash_storage_timestep = self.basinCharacteristics['nash_storage'].clone() 

        # Calculate the discharge from each Nash storage
        Q = self.basinCharacteristics['K_nash'].unsqueeze(1) * nash_storage_timestep # first pass would be 0

        # Update Nash storage with discharge
        nash_storage_timestep = nash_storage_timestep - Q # first pass would be 0

        # The first storage receives the lateral flow outflux from soil storage
        nash_storage_timestep[:, 0] = nash_storage_timestep[:, 0] + self.flux_lat_m

        # The remaining storage receives the discharge from the upper Nash storage
        if self.num_reservoirs > 1:
            nash_storage_timestep[:, 1:] = nash_storage_timestep[:, 1:]+ Q[:, :-1]
        #    basinCharacteristics['nash_storage'][:, 1:] = basinCharacteristics['nash_storage'][:, 1:] + Q[:, :-1]

        # Update the state
        self.basinCharacteristics['nash_storage'] = nash_storage_timestep.clone()

        # The final discharge at the timestep from Nash cascade is from the lowermost Nash storage
        self.flux_nash_lateral_runoff_m= Q[:, -1].clone()

        ### track_volume_from_nash_cascade
        self.vol['in_nash'] = self.vol['in_nash'] + self.flux_lat_m
        self.vol['out_nash'] = self.vol['out_nash'] + self.flux_nash_lateral_runoff_m


