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
    
    Last edited: by Ziyu, 07/28/2024
    
    General outline:
    The model takes in output from a NN (like a LSTM), and forcings needed for CFE, which are just PET and Precip. 
    
    This starts out as a code-vomit of cfe.py (the physics) & bmi_cfe.py (configuration & some parameter settings) 
    into the NH framework and will follow the general formatting of shm.py + Example 02. The idea is to run this 
    dcfe model via hybrid_model. Classes defined in cfe.py & bmi_cfe.py could be referred to from __init__.
    
    This is still being updated and connected into the NH framework. 
    
    
    """
    
    def __init__(self, cfg: Config):
        super(dCFE, self).__init__(cfg=cfg)
        
        # loading the other classes within this script is not needed since they are all referenced
        # in other classes that is being made into an isntance of BMI_CFE within Forward...
        # self.bmi_cfe = BMI_CFE() # so we can use functions in this class to run CFE, etc
        # self.soil_moisture_flux_ode = soil_moisture_flux_ode() ### This might be already loaded thru CFE
        # self.cfe = CFE() ### This one might be already loaded thru BMI_CFE, need to double check.
        
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
        cfe_state, out =  self._initialize_information(conceptual_inputs=x_conceptual)
        
        # There are some other initializations from bmi_cfe.py for each epoch, maybe can be incorprorated above?
        
        # loop through time-steps
        for j in range(x_conceptual.shape[1]):
            # Read the forcing (from bmi)
            precip = x_conceptual[:, j, 0]
            pet = x_conceptual[:, j, 1]
            
            # initialize the CFE model (from bmi)
            self.cfe_instance = BMI_CFE(
                Cgw=parameters['satdk'][:,j],
                satdk=parameters['Cgw'][:,j],
                cfg=self.cfg,
                cfe_params=self.data.params,
            )
            self.cfe_instance.initialize() # grabs all parameter, fluxes

            # Set precip and PET values in CFE
            self.cfe_instance.set_value(
                "atmosphere_water__time_integral_of_precipitation_mass_flux", precip
                )
            self.cfe_instance.set_value("water_potential_evaporation_flux", pet)
            
            runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm
            # now storing the total outflow
            out[:, j, 0] = runoff
        # loop thru time ends here
        
        return {'y_hat': out, 'parameters': parameters, 'internal_states': cfe_state}

    #______________________defining states and parameter properties________________
    @property
    def initial_states(self):
        return {'ss': 0.0,
                'sf': 1.0,
                'su': 5.0,
                'si': 10.0,
                'sb': 15.0} ## we need to define these differently for CFE still

    @property
    def parameter_ranges(self):
        return {'satdk': [0.0, 0.000726], # Saturated hydraulic conductivity
                'Cgw': [0.0000018, 0.0018], # Primary groundwater reservoir constant
                } # the only 2 parameters from NN that was from the original dCFE model was Cgw & satdk
    #___________________end of defining states and parameters__________________

#____________________beginning of class and methods in bmi_cfe.py__________
class BMI_CFE:
    def __init__(
        self, parameters, cfg=None, cfe_params=None, verbose=False
    ):
        # ________________________________________________
        # Create a Bmi CFE model that is ready for initialization

        super(BMI_CFE, self).__init__()
        self._values = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._start_time = 0.0
        self._end_time = np.finfo("d").max

        # these need to be initialized here as scale_output() called in update()
        self.streamflow_cmh = torch.tensor(0.0)
        # self.streamflow_fms = 0.0
        self.surface_runoff_m = torch.tensor(0.0)

        # ________________________________________________
        # Required, static attributes of the model

        self._att_map = {
            "model_name": "Conceptual Functional Equivalent (CFE)",
            "version": "1.0",
            "author_name": "Jonathan Martin Frame",
            "grid_type": "scalar",
            "time_step_size": 3600,
            "time_units": "1 hour",
        }

        # ________________________________________________
        # Input variable names (CSDMS standard names)

        self._input_var_names = [
            "atmosphere_water__time_integral_of_precipitation_mass_flux",
            "water_potential_evaporation_flux",
        ]

        # ________________________________________________
        # Output variable names (CSDMS standard names)

        self._output_var_names = [
            "land_surface_water__runoff_depth",
            "land_surface_water__runoff_volume_flux",
            "DIRECT_RUNOFF",
            "GIUH_RUNOFF",
            "NASH_LATERAL_RUNOFF",
            "DEEP_GW_TO_CHANNEL_FLUX",
            "SOIL_CONCEPTUAL_STORAGE",
        ]

        # ________________________________________________
        # Create a Python dictionary that maps CSDMS Standard
        # Names to the model's internal variable names.
        # This is going to get long,
        #     since the input variable names could come from any forcing...

        self._var_name_units_map = {
            "land_surface_water__runoff_volume_flux": ["streamflow_cmh", "m3 h-1"],
            "land_surface_water__runoff_depth": ["total_discharge", "m h-1"],
            # --------------   Dynamic inputs --------------------------------
            "atmosphere_water__time_integral_of_precipitation_mass_flux": [
                "timestep_rainfall_input_m",
                "m h-1",
            ],
            "water_potential_evaporation_flux": ["potential_et_m_per_s", "m s-1"],
            "DIRECT_RUNOFF": ["surface_runoff_depth_m", "m"],
            "GIUH_RUNOFF": ["flux_giuh_runoff_m", "m"],
            "NASH_LATERAL_RUNOFF": ["flux_nash_lateral_runoff_m", "m"],
            "DEEP_GW_TO_CHANNEL_FLUX": ["flux_from_deep_gw_to_chan_m", "m"],
            "SOIL_CONCEPTUAL_STORAGE": ["soil_reservoir['storage_m']", "m"],
        }

        # ________________________________________________
        # this is the bmi configuration file
        self.cfe_params = cfe_params
        self.cfg = cfg

        # NN params
        self.Cgw = Cgw  # .unsqueeze(dim=0)
        self.satdk = satdk  # .unsqueeze(dim=0)

        # This takes in the cfg read with Hydra from the yml file
        # self.cfe_cfg = global_params

        self.num_basins = len(self.cfg.data.basin_ids)

        # Verbose
        self.verbose = verbose

        # _________________________________________________
        # nn parameters
        # None

    def reset_internal_attributes(self):
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            self.Schaake_adjusted_magic_constant_by_soil_type.detach()
        )
        self.output_factor_cms = self.output_factor_cms.detach()

        self.gw_reservoir["storage_m"] = self.gw_reservoir["storage_m"].detach()
        self.gw_reservoir["storage_max_m"] = self.gw_reservoir["storage_max_m"].detach()

        self.soil_reservoir["storage_m"] = self.soil_reservoir["storage_m"].detach()
        self.soil_reservoir["wilting_point_m"] = self.soil_reservoir[
            "wilting_point_m"
        ].detach()
        self.soil_reservoir["storage_max_m"] = self.soil_reservoir[
            "storage_max_m"
        ].detach()
        self.soil_reservoir["coeff_primary"] = self.soil_reservoir[
            "coeff_primary"
        ].detach()
        self.soil_reservoir["storage_threshold_primary_m"] = self.soil_reservoir[
            "storage_threshold_primary_m"
        ].detach()
        self.soil_reservoir["storage_threshold_secondary_m"] = self.soil_reservoir[
            "storage_threshold_secondary_m"
        ].detach()

    def load_cfe_params(self):
        for param in self.cfe_params.values():
            if torch.is_tensor(param):
                if param.grad is not None:
                    param.grad = None

        for param in self.cfe_params["soil_params"].values():
            if torch.is_tensor(param):
                if param.grad is not None:
                    param.grad = None

        # GET VALUES FROM Data class.

        # Catchment area
        self.catchment_area_km2 = self.cfe_params["catchment_area_km2"]

        # Soil parameters
        self.alpha_fc = self.cfe_params["alpha_fc"]
        self.soil_params = self.cfe_params["soil_params"]
        self.soil_params["scheme"] = self.cfg.soil_scheme

        # GW paramters
        self.max_gw_storage = self.cfe_params["max_gw_storage"]
        self.expon = self.cfe_params["expon"]
        # self.Cgw = self.cfe_params["Cgw"] -> this has been changed to dynamic parameter

        # Nash storage
        self.K_nash = self.cfe_params["K_nash"]
        self.nash_storage = self.cfe_params["nash_storage"].view(self.num_basins, -1)

        # Lateral flow
        self.K_lf = self.cfe_params["K_lf"]

        # Surface runoff
        self.refkdt = self.cfe_params["refkdt"]
        self.giuh_ordinates = self.cfe_params["giuh_ordinates"].view(
            self.num_basins, -1
        )
        self.surface_partitioning_scheme = self.cfe_params["partition_scheme"]

        # Other
        self.stand_alone = 0

    # __________________________________________________________________
    # __________________________________________________________________
    # BMI: Model Control Function
    def initialize(self, current_time_step=0):
        self.current_time_step = current_time_step

        # ________________________________________________
        # Create some lookup tabels from the long variable names
        self._var_name_map_long_first = {
            long_name: self._var_name_units_map[long_name][0]
            for long_name in self._var_name_units_map.keys()
        }
        self._var_name_map_short_first = {
            self._var_name_units_map[long_name][0]: long_name
            for long_name in self._var_name_units_map.keys()
        }
        self._var_units_map = {
            long_name: self._var_name_units_map[long_name][1]
            for long_name in self._var_name_units_map.keys()
        }

        # ________________________________________________
        # Initalize all the variables
        # so that they'll be picked up with the get functions
        for long_var_name in list(self._var_name_units_map.keys()):
            # All the variables are single values
            # so just set to zero for now
            self._values[long_var_name] = 0
            setattr(self, self.get_var_name(long_var_name), 0)

        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.load_cfe_params()

        # ________________________________________________
        # initialize simulation constants
        self.atm_press_Pa = 101325.0
        self.unit_weight_water_N_per_m3 = 9810.0

        # ________________________________________________
        # Time control
        self.time_step_size = 3600
        self.timestep_h = self.time_step_size / 3600
        self.timestep_d = self.timestep_h / 24.0

        # ________________________________________________________
        # Set these values now that we have the information from the configuration file.
        self.num_giuh_ordinates = self.giuh_ordinates.size(1)
        self.num_lateral_flow_nash_reservoirs = self.nash_storage.size(1)

        # ________________________________________________
        # The configuration should let the BMI know what mode to run in (framework vs standalone)
        # If it is stand alone, then load in the forcing and read the time from the forcig file
        if self.stand_alone == 1:
            self.load_forcing_file()
            try:
                self.current_time = pd.to_datetime(
                    self.forcing_data["time"][self.current_time_step]
                )
            except:
                try:
                    self.current_time = pd.to_datetime(
                        self.forcing_data["date"][self.current_time_step]
                    )
                except:
                    print("Check the column names")
        # ________________________________________________
        # In order to check mass conservation at any time
        self.reset_volume_tracking()
        self.reset_flux_and_states()

        ####################################################################
        # ________________________________________________________________ #
        # ________________________________________________________________ #
        # CREATE AN INSTANCE OF THE CONCEPTUAL FUNCTIONAL EQUIVALENT MODEL #
        self.cfe_model = CFE()
        # ________________________________________________________________ #
        # ________________________________________________________________ #
        ####################################################################

    # ________________________________________________
    # Reset the flux and states to zero for the next epoch in NN
    def reset_flux_and_states(self):
        # ________________________________________________
        # Time control
        self.current_time_step = 0
        self.current_time = pd.Timestamp(year=2007, month=10, day=1, hour=0)

        # ________________________________________________
        # Inputs
        self.timestep_rainfall_input_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.potential_et_m_per_s = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )

        # ________________________________________________
        # flux variables
        self.flux_overland_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # surface runoff that goes through the GIUH convolution process
        self.flux_perc_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # flux from soil to deeper groundwater reservoir
        self.flux_lat_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # lateral flux in the subsurface to the Nash cascade
        self.flux_from_deep_gw_to_chan_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # flux from the deep reservoir into the channels
        self.gw_reservoir_storage_deficit_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # the available space in the conceptual groundwater reservoir
        self.primary_flux_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # temporary vars.
        self.secondary_flux_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # temporary vars.
        self.primary_flux_from_gw_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.secondary_flux_from_gw_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.total_discharge = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.diff_infilt = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.diff_perc = torch.zeros((1, self.num_basins), dtype=torch.float64)

        # ________________________________________________
        # Evapotranspiration
        self.potential_et_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.actual_et_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.reduced_potential_et_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.actual_et_from_rain_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.actual_et_from_soil_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )

        # ________________________________________________
        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s
        self.output_factor_cms = (
            (1 / 1000) * (self.catchment_area_km2 * 1000 * 1000) * (1 / 3600)
        )

        # ________________________________________________
        # ________________________________________________
        # SOIL RESERVOIR CONFIGURATION
        # Local values to be used in setting up soil reservoir
        trigger_z_m = torch.tensor([0.5])
        field_capacity_atm_press_fraction = self.alpha_fc

        # ________________________________________________
        # Soil outflux calculation, Equation 3 in Fred Ogden's document

        H_water_table_m = (
            field_capacity_atm_press_fraction
            * self.atm_press_Pa
            / self.unit_weight_water_N_per_m3
        )

        soil_water_content_at_field_capacity = self.soil_params["smcmax"] * torch.pow(
            H_water_table_m / self.soil_params["satpsi"], (1.0 / self.soil_params["bb"])
        )

        Omega = H_water_table_m - trigger_z_m

        # ________________________________________________
        # Upper & lower limit of the integral in Equation 4 in Fred Ogden's document

        lower_lim = torch.pow(Omega, (1.0 - 1.0 / self.soil_params["bb"])) / (
            1.0 - 1.0 / self.soil_params["bb"]
        )

        upper_lim = torch.pow(
            Omega + self.soil_params["D"], (1.0 - 1.0 / self.soil_params["bb"])
        ) / (1.0 - 1.0 / self.soil_params["bb"])

        # ________________________________________________
        # Integral & power term in Equation 4 & 5 in Fred Ogden's document

        storage_thresh_pow_term = torch.pow(
            1.0 / self.soil_params["satpsi"], (-1.0 / self.soil_params["bb"])
        )

        lim_diff = upper_lim - lower_lim

        field_capacity_storage_threshold_m = (
            self.soil_params["smcmax"] * storage_thresh_pow_term * lim_diff
        )

        # ________________________________________________
        # lateral flow function parameters
        assumed_near_channel_water_table_slope = 0.01  # [L/L]
        lateral_flow_threshold_storage_m = field_capacity_storage_threshold_m
        self.soil_reservoir_storage_deficit_m = torch.tensor([0.0], dtype=torch.float64)

        # ________________________________________________
        # Subsurface reservoirs
        self.gw_reservoir = {
            "is_exponential": True,
            "storage_max_m": self.max_gw_storage,
            # "coeff_primary": self.Cgw, -> this has been changed to dynamic parameter
            "exponent_primary": self.expon,
            "storage_threshold_primary_m": torch.zeros(
                (1, self.num_basins), dtype=torch.float64
            ),
            # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off.
            "storage_threshold_secondary_m": torch.zeros(
                (1, self.num_basins), dtype=torch.float64
            ),
            "coeff_secondary": torch.zeros((1, self.num_basins), dtype=torch.float64),
            "exponent_secondary": torch.ones((1, self.num_basins), dtype=torch.float64),
        }
        self.gw_reservoir["storage_m"] = self.gw_reservoir["storage_max_m"] * 0.01
        self.volstart = self.volstart.add(self.gw_reservoir["storage_m"])
        self.vol_in_gw_start = self.gw_reservoir["storage_m"]

        # TODO: update soil parameter

        self.soil_reservoir = {
            "is_exponential": False,
            "wilting_point_m": self.soil_params["wltsmc"] * self.soil_params["D"],
            "storage_max_m": self.soil_params["smcmax"] * self.soil_params["D"],
            "coeff_primary": self.satdk
            * self.soil_params["slop"]
            * self.time_step_size,  # Controls percolation to GW, Equation 11
            "exponent_primary": torch.ones(
                (1, self.num_basins), dtype=torch.float64
            ),  # Controls percolation to GW, FIXED to 1 based on Equation 11
            "storage_threshold_primary_m": field_capacity_storage_threshold_m,
            "coeff_secondary": self.K_lf,  # Controls lateral flow
            "exponent_secondary": torch.ones(
                (1, self.num_basins), dtype=torch.float64
            ),  # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
            "storage_threshold_secondary_m": lateral_flow_threshold_storage_m,
        }
        self.soil_reservoir["storage_m"] = self.soil_reservoir["storage_max_m"] * 0.6
        self.volstart = self.volstart.add(self.soil_reservoir["storage_m"])
        self.vol_soil_start = self.soil_reservoir["storage_m"]

        # ________________________________________________
        # Schaake partitioning
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            self.refkdt * self.satdk / 2.0e-06
        )
        # print(self.refkdt.grad)
        self.Schaake_output_runoff_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.infiltration_depth_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )

        # ________________________________________________________
        self.runoff_queue_m_per_timestep = torch.zeros(
            self.giuh_ordinates.shape[0], self.num_giuh_ordinates + 1
        )

        # __________________________________________________________
        self.surface_runoff_m = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.streamflow_cmh = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.flux_nash_lateral_runoff_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.flux_giuh_runoff_m = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.flux_Qout_m = torch.zeros((1, self.num_basins), dtype=torch.float64)

    def update_params(self, Cgw, satdk):
        """Update dynamic parameters"""
        self.Cgw = Cgw.unsqueeze(dim=0)
        self.satdk = satdk.unsqueeze(dim=0)

        # Update parameters related to Cgw
        # None

        # Update parameters related to satdk
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            self.refkdt * self.satdk / 2.0e-06
        )
        self.soil_reservoir["coeff_primary"] = (
            self.satdk * self.soil_params["slop"] * self.time_step_size
        )

        if self.verbose:
            print(
                f"Cgw: {self.Cgw:.2f}; satdk: {self.satdk:.5f}; \
                Schaake: {self.Schaake_adjusted_magic_constant_by_soil_type:.3f};\
                Soilcoeff: {self.soil_reservoir['coeff_primary']:.5f}"
            )

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update(self):
        self.volin = self.volin.add(self.timestep_rainfall_input_m)
        self.cfe_model.run_cfe(self)
        self.scale_output()

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update_until(self, until, verbose=True):
        for i in range(self.current_time_step, until):
            self.cfe_model.run_cfe(self)
            self.scale_output()
            if verbose:
                print("total discharge: {}".format(self.total_discharge))
                print("at time: {}".format(self.current_time))

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def finalize(self, print_mass_balance=False):
        self.finalize_mass_balance(verbose=print_mass_balance)
        self.reset_volume_tracking()

        """Finalize model."""
        self.cfe_model = None
        self.cfe_state = None

    # ________________________________________________
    # Mass balance tracking
    def reset_volume_tracking(self):
        self.volstart = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_soil = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_rain = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_partition_runoff = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_partition_infilt = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_out_giuh = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_end_giuh = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_gw = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_gw_start = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_gw_end = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_from_gw = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_in_nash = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_in_nash_end = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_out_nash = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_soil_start = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_soil = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_soil_to_lat_flow = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_soil_to_gw = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_soil_end = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.volin = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.volout = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.volend = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_partition_runoff_IOF = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_partition_runoff_SOF = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_et_to_atm = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_soil = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_rain = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_PET = torch.zeros((1, self.num_basins), dtype=torch.float64)

        self.vol_in_gw_start = torch.zeros((1, self.num_basins), dtype=torch.float64)

        return

    # ________________________________________________________
    def finalize_mass_balance(self, verbose=True):
        self.volend = self.soil_reservoir["storage_m"] + self.gw_reservoir["storage_m"]
        self.vol_in_gw_end = self.gw_reservoir["storage_m"]

        # the GIUH queue might have water in it at the end of the simulation, so sum it up.
        self.vol_end_giuh = torch.sum(self.runoff_queue_m_per_timestep, dim=1)
        self.vol_in_nash_end = torch.sum(self.nash_storage, dim=1)

        self.vol_soil_end = self.soil_reservoir["storage_m"]

        self.global_residual = (
            self.volstart + self.volin - self.volout - self.volend - self.vol_end_giuh
        )
        self.partition_residual = (
            self.volin
            - self.vol_partition_runoff
            - self.vol_partition_infilt
            - self.vol_et_from_rain
        )
        self.giuh_residual = (
            self.vol_partition_runoff - self.vol_out_giuh - self.vol_end_giuh
        )
        self.soil_residual = (
            self.vol_soil_start
            + self.vol_to_soil
            - self.vol_soil_to_lat_flow
            - self.vol_to_gw
            - self.vol_et_from_soil
            - self.vol_soil_end
        )
        self.nash_residual = self.vol_in_nash - self.vol_out_nash - self.vol_in_nash_end
        self.gw_residual = (
            self.vol_in_gw_start
            + self.vol_to_gw
            - self.vol_from_gw
            - self.vol_in_gw_end
        )
        self.AET_residual = (
            self.vol_et_to_atm - self.vol_et_from_rain - self.vol_et_from_soil
        )

        if verbose:
            i = 0
            print(f"\nGLOBAL MASS BALANCE (print for {i}-th basin)")
            print("  initial volume: {:8.4f}".format(self.volstart[0][i].item()))
            print("    volume input: {:8.4f}".format(self.volin[0][i].item()))
            print("   volume output: {:8.4f}".format(self.volout[0][i].item()))
            print("    final volume: {:8.4f}".format(self.volend[0][i].item()))
            print("        residual: {:6.4e}".format(self.global_residual[0][i].item()))

            print("\n AET & PET")
            print("      volume PET: {:8.4f}".format(self.vol_PET[0][i].item()))
            print("      volume AET: {:8.4f}".format(self.vol_et_to_atm[0][i].item()))
            print(
                "ET from rainfall: {:8.4f}".format(self.vol_et_from_rain[0][i].item())
            )
            print(
                "    ET from soil: {:8.4f}".format(self.vol_et_from_soil[0][i].item())
            )
            print("    AET residual: {:6.4e}".format(self.AET_residual[0][i].item()))

            print("\nPARTITION MASS BALANCE")
            print(
                "    surface runoff: {:8.4f}".format(
                    self.vol_partition_runoff[0][i].item()
                )
            )
            print(
                "      infiltration: {:8.4f}".format(
                    self.vol_partition_infilt[0][i].item()
                )
            )
            print(
                " vol. et from rain: {:8.4f}".format(self.vol_et_from_rain[0][i].item())
            )
            print(
                "partition residual: {:6.4e}".format(
                    self.partition_residual[0][i].item()
                )
            )

            print("\nGIUH MASS BALANCE")
            print(
                "  vol. into giuh: {:8.4f}".format(
                    self.vol_partition_runoff[0][i].item()
                )
            )
            print("   vol. out giuh: {:8.4f}".format(self.vol_out_giuh[0][i].item()))
            print(" vol. end giuh q: {:8.4f}".format(self.vol_end_giuh[i].item()))
            print("   giuh residual: {:6.4e}".format(self.giuh_residual[0][i].item()))

            if self.soil_params["scheme"] == "classic":
                print("\nSOIL WATER CONCEPTUAL RESERVOIR MASS BALANCE")
            elif self.soil_params["scheme"] == "ode":
                print("\nSOIL WATER MASS BALANCE")
            print(
                "     init soil vol: {:8.6f}".format(self.vol_soil_start[0][i].item())
            )
            print("    vol. into soil: {:8.6f}".format(self.vol_to_soil[0][i].item()))
            print(
                "  vol.soil2latflow: {:8.6f}".format(
                    self.vol_soil_to_lat_flow[0][i].item()
                )
            )
            print(
                "   vol. soil to gw: {:8.6f}".format(self.vol_soil_to_gw[0][i].item())
            )
            print(
                " vol. et from soil: {:8.6f}".format(self.vol_et_from_soil[0][i].item())
            )
            print("   final vol. soil: {:8.6f}".format(self.vol_soil_end[0][i].item()))
            print("  vol. soil resid.: {:6.6e}".format(self.soil_residual[0][i].item()))

            print("\nNASH CASCADE CONCEPTUAL RESERVOIR MASS BALANCE")
            print("    vol. to nash: {:8.4f}".format(self.vol_in_nash[0][i].item()))
            print("  vol. from nash: {:8.4f}".format(self.vol_out_nash[0][i].item()))
            print(" final vol. nash: {:8.4f}".format(self.vol_in_nash_end[i].item()))
            print("nash casc resid.: {:6.4e}".format(self.nash_residual[0][i].item()))

            print("\nGROUNDWATER CONCEPTUAL RESERVOIR MASS BALANCE")
            print("init gw. storage: {:8.4f}".format(self.vol_in_gw_start[0][i].item()))
            print("       vol to gw: {:8.4f}".format(self.vol_to_gw[0][i].item()))
            print("     vol from gw: {:8.4f}".format(self.vol_from_gw[0][i].item()))
            print("final gw.storage: {:8.4f}".format(self.vol_in_gw_end[0][i].item()))
            print("    gw. residual: {:6.4e}".format(self.gw_residual[0][i].item()))

        return

    # ________________________________________________________
    def load_forcing_file(self):
        self.forcing_data = pd.read_csv(self.forcing_file)

    # ________________________________________________________
    def load_unit_test_data(self):
        self.unit_test_data = pd.read_csv(self.compare_results_file)
        self.cfe_output_data = pd.DataFrame().reindex_like(self.unit_test_data)

    # ------------------------------------------------------------
    def scale_output(self):
        self.surface_runoff_m = self.flux_Qout_m  # self.total_discharge
        self._values["land_surface_water__runoff_depth"] = self.surface_runoff_m
        self.streamflow_cmh = (
            self.total_discharge
        )  # self._values['land_surface_water__runoff_depth'] * self.output_factor_cms

        self._values[
            "land_surface_water__runoff_volume_flux"
        ] = self.streamflow_cmh  # * (1/35.314)

        self._values["DIRECT_RUNOFF"] = self.surface_runoff_depth_m
        self._values["GIUH_RUNOFF"] = self.flux_giuh_runoff_m
        self._values["NASH_LATERAL_RUNOFF"] = self.flux_nash_lateral_runoff_m
        self._values["DEEP_GW_TO_CHANNEL_FLUX"] = self.flux_from_deep_gw_to_chan_m
        # if self.soil_scheme.lower() == 'ode': # Commented out just for debugging, restore later
        self._values["SOIL_CONCEPTUAL_STORAGE"] = self.soil_reservoir["storage_m"]

    # ----------------------------------------------------------------------------
    def initialize_forcings(self):
        for forcing_name in self.cfg_train["dynamic_inputs"]:
            setattr(self, self._var_name_map_short_first[forcing_name], 0)

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # BMI: Model Information Functions
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    def get_attribute(self, att_name):
        try:
            return self._att_map[att_name.lower()]
        except:
            print(" ERROR: Could not find attribute: " + att_name)

    # --------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    # --------------------------------------------------------
    def get_input_var_names(self):
        return self._input_var_names

    def get_output_var_names(self):
        return self._output_var_names

    # ------------------------------------------------------------
    def get_component_name(self):
        """Name of the component."""
        return self.get_attribute("model_name")  # JG Edit

    # ------------------------------------------------------------
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    # ------------------------------------------------------------
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    # ------------------------------------------------------------
    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ptr(var_name)

    def return_runoff(self):
        return self.flux_Qout_m

    def return_storage_states(self):
        return torch.cat(
            (self.gw_reservoir["storage_m"], self.soil_reservoir["storage_m"]), dim=0
        )

    # -------------------------------------------------------------------
    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # BMI: Variable Information Functions
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    def get_var_name(self, long_var_name):
        return self._var_name_map_long_first[long_var_name]

    # -------------------------------------------------------------------
    def get_var_units(self, long_var_name):
        return self._var_units_map[long_var_name]

    # -------------------------------------------------------------------
    def get_var_type(self, long_var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        # JG Edit
        return self.get_value_ptr(long_var_name)  # .dtype

    # ------------------------------------------------------------
    def get_var_grid(self, name):
        # JG Edit
        # all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_grid_id

    # ------------------------------------------------------------
    def get_var_itemsize(self, name):
        #        return np.dtype(self.get_var_type(name)).itemsize
        return np.array(self.get_value(name)).itemsize

    # ------------------------------------------------------------
    def get_var_location(self, name):
        # JG Edit
        # all vars have location node but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    # -------------------------------------------------------------------
    # JG Note: what is this used for?
    def get_var_rank(self, long_var_name):
        return np.int16(0)

    # -------------------------------------------------------------------
    def get_start_time(self):
        return self._start_time  # JG Edit

    # -------------------------------------------------------------------
    def get_end_time(self):
        return self._end_time  # JG Edit

    # -------------------------------------------------------------------
    def get_current_time(self):
        return self.current_time

    # -------------------------------------------------------------------
    def get_time_step(self):
        return self.get_attribute("time_step_size")  # JG: Edit

    # -------------------------------------------------------------------
    def get_time_units(self):
        return self.get_attribute("time_units")

    # -------------------------------------------------------------------
    def set_value(self, var_name, value):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """
        setattr(self, self.get_var_name(var_name), value)
        self._values[var_name] = value

    # ------------------------------------------------------------
    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        # JG Note: TODO confirm this is correct. Get/set values ~=
        #        val = self.get_value_ptr(name)
        #        val.flat[inds] = src

        # JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(name)).flatten().shape[0] == 1:
            self.set_value(name, src)
        else:
            # JMFrame: Need to set the value with the updated array with new index value
            val = self.get_value_ptr(name)
            for i in inds.shape:
                val.flatten()[inds[i]] = src[i]
            self.set_value(name, val)

    # ------------------------------------------------------------
    def get_var_nbytes(self, long_var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        # JMFrame NOTE: Had to import sys for this function
        return sys.getsizeof(self.get_value_ptr(long_var_name))

    # ------------------------------------------------------------
    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        # JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(var_name)).flatten().shape[0] == 1:
            return self.get_value(var_name)
        else:
            val_array = self.get_value(var_name).flatten()
            return np.array([val_array[i] for i in indices])

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html
    # ------------------------------------------------------------
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    # ------------------------------------------------------------
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    # ------------------------------------------------------------
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    # ------------------------------------------------------------
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    # ------------------------------------------------------------
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")

    # ------------------------------------------------------------
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    # ------------------------------------------------------------
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")

    # ------------------------------------------------------------
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin")

    # ------------------------------------------------------------
    def get_grid_rank(self, grid_id):
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    # ------------------------------------------------------------
    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape")

    # ------------------------------------------------------------
    def get_grid_size(self, grid_id):
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    # ------------------------------------------------------------
    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing")

    # ------------------------------------------------------------
    def get_grid_type(self, grid_id=0):
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return "scalar"

    # ------------------------------------------------------------
    def get_grid_x(self):
        raise NotImplementedError("get_grid_x")

    # ------------------------------------------------------------
    def get_grid_y(self):
        raise NotImplementedError("get_grid_y")

    # ------------------------------------------------------------
    def get_grid_z(self):
        raise NotImplementedError("get_grid_z")
#___________________end of methods in bmi_cfe.py___________________________


#___________________beginning of other defined methods in cfe.py___________
class soil_moisture_flux_ode(nn.Module):
    """
    Soil reservoir module that solves ODE
    Using ODE allows simultaneous calculation of outflux, instead of stepwise subtraction of flux which causes overextraction from SM reservoir
    The behavior of soil moisture storage is divided into 3 stages.
    Stage 1: S (Soil moisture storage ) > storage_threshold_primary_m
        Interpretation: When the soil moisture is plenty, AET(=PET), percolation, and lateral flow are all active.
        Equation: dS/dt = Infiltration - PET - (Klf+Kperc) * (S - storage_threshold_primary_m)/(storage_max_m - storage_threshold_primary_m)
    Stage 2: storage_threshold_primary_m > S (Soil moisture storage) > storage_threshold_primary_m - wltsmc
        Interpretation: When the soil moisture is in the medium range, AET is active and proportional to the soil moisture storage ratio. No percolation and lateral flow fluxes.
        Equation: dS/dt = Infiltration - PET * (S - wltsmc)/(storage_threshold_primary_m - wltsmc)
    Stage 3: wltsmc > S (Soil moisture storage)
        Interpretation: When the soil moisture is depleted, no outflux is active
        Equation: dS/dt = Infitlation

    :param t: time
    :param S: Soil moisture storage in meter
    :param storage_threshold_primary_m:
    :param storage_max_m: maximum soil moisture storage, i.e., porosity
    :param coeff_primary: K_perc, percolation coefficient
    :param coeff_secondary: K_lf, lateral flow coefficient
    :param PET: potential evapotranspiration
    :param infilt: infiltration
    :param wilting_point_m: wilting point (in meter)
    :return: dS
    """

    # def __init__(self, i=0, cfe_state=None, reservoir=None):
    def __init__(self, cfe_state=None, reservoir=None):
        super().__init__()

        self.threshold_primary = reservoir["storage_threshold_primary_m"]
        self.storage_max_m = reservoir["storage_max_m"]
        self.wilting_point_m = reservoir["wilting_point_m"]
        self.coeff_primary = reservoir["coeff_primary"]
        self.coeff_secondary = reservoir["coeff_secondary"]
        self.infilt = cfe_state.infiltration_depth_m
        self.PET = cfe_state.reduced_potential_et_m_per_timestep

    def forward(self, t, states):
        S = states

        storage_above_threshold_m = S - self.threshold_primary
        storage_diff = self.storage_max_m - self.threshold_primary
        storage_ratio = torch.clamp(
            storage_above_threshold_m / storage_diff, max=1.0, min=0.0
        )

        storage_above_threshold_m_paw = S - self.wilting_point_m
        storage_diff_paw = self.threshold_primary - self.wilting_point_m
        storage_ratio_paw = torch.clamp(
            storage_above_threshold_m_paw / storage_diff_paw, max=1.0, min=0.0
        )  # Equation 11 (Ogden's document)

        one_vector = torch.ones_like(S)
        dS_dt = (
            self.infilt
            - one_vector * (self.coeff_primary + self.coeff_secondary) * storage_ratio
            - self.PET * storage_ratio_paw
        )

        return dS_dt


class CFE:
    def __init__(self):
        super(CFE, self).__init__()

    def initialize_flux(self, cfe_state):
        """Some fluxses need to be initialized at each timestep"""
        cfe_state.surface_runoff_depth_m = torch.zeros(
            (1, cfe_state.num_basins), dtype=torch.float64
        )

        cfe_state.infiltration_depth_m = torch.zeros(
            (1, cfe_state.num_basins), dtype=torch.float64
        )

        cfe_state.actual_et_from_rain_m_per_timestep = torch.zeros(
            (1, cfe_state.num_basins), dtype=torch.float64
        )

        cfe_state.actual_et_from_soil_m_per_timestep = torch.zeros(
            (1, cfe_state.num_basins), dtype=torch.float64
        )

        cfe_state.primary_flux_m = torch.zeros(
            (1, cfe_state.num_basins), dtype=torch.float64
        )
        cfe_state.secondary_flux_m = torch.zeros(
            (1, cfe_state.num_basins), dtype=torch.float64
        )

    # ____________________________________________________________________________________
    def calculate_input_rainfall_and_PET(self, cfe_state):
        """
        Calculate input rainfall and PET
        """
        cfe_state.potential_et_m_per_timestep = (
            cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        )
        cfe_state.vol_PET += cfe_state.potential_et_m_per_timestep.detach()
        cfe_state.reduced_potential_et_m_per_timestep = (
            cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        )

    # ____________________________________________________________________________________
    def calculate_evaporation_from_rainfall(self, cfe_state):
        """
        Calculate evaporation from rainfall. If it is raining, take PET from rainfall
        """

        # Creating a mask for elements where timestep_rainfall_input_m > 0
        rainfall_mask = cfe_state.timestep_rainfall_input_m > 0

        # If rainfall is NOT present, skip this module
        if not torch.any(rainfall_mask):
            if cfe_state.verbose:
                print(
                    "All rainfall inputs are less than or equal to 0. Function et_from_rainfall will not proceed."
                )
            return

        # If rainfall is present, calculate evaporation from rainfall
        self.et_from_rainfall(cfe_state, rainfall_mask)
        self.track_volume_et_from_rainfall(cfe_state)

    # ____________________________________________________________________________________
    def track_volume_et_from_rainfall(self, cfe_state):
        cfe_state.vol_et_from_rain += (
            cfe_state.actual_et_from_rain_m_per_timestep.detach()
        )
        cfe_state.vol_et_to_atm += cfe_state.actual_et_from_rain_m_per_timestep.detach()
        cfe_state.volout += cfe_state.actual_et_from_rain_m_per_timestep.detach()
        cfe_state.actual_et_m_per_timestep += (
            cfe_state.actual_et_from_rain_m_per_timestep.detach()
        )

    # ____________________________________________________________________________________
    def calculate_evaporation_from_soil(self, cfe_state):
        """
        If the soil moisture calculation scheme is 'classic', calculate the evaporation from the soil
        Elseif the soil moisture calculation scheme is 'ode', do nothing, because evaporation from the soil will be calculated within run_soil_moisture_scheme
        """

        # Creating a mask for elements where excess soil moisture > 0
        excess_sm_for_ET_mask = (
            cfe_state.soil_reservoir["storage_m"]
            > cfe_state.soil_reservoir["wilting_point_m"]
        )
        et_mask = cfe_state.reduced_potential_et_m_per_timestep > 0

        # Combine both masks
        combined_mask = et_mask & excess_sm_for_ET_mask

        if not torch.any(combined_mask):
            if cfe_state.verbose:
                print(
                    "All SM are under wilting point. Function et_from_soil will not proceed."
                )
            return

        # If the soil moisture storage is more than wilting point, and PET is not zero, calculate ET from soil
        self.et_from_soil(cfe_state, combined_mask)
        self.track_volume_et_from_soil(cfe_state)

    # ____________________________________________________________________________________
    def track_volume_et_from_soil(self, cfe_state):
        cfe_state.vol_et_from_soil += (
            cfe_state.actual_et_from_soil_m_per_timestep.detach()
        )
        cfe_state.vol_et_to_atm += cfe_state.actual_et_from_soil_m_per_timestep.detach()
        cfe_state.volout += cfe_state.actual_et_from_soil_m_per_timestep.detach()
        cfe_state.actual_et_m_per_timestep += (
            cfe_state.actual_et_from_soil_m_per_timestep.detach()
        )

    # ____________________________________________________________________________________
    def calculate_the_soil_moisture_deficit(self, cfe_state):
        """Calculate the soil moisture deficit"""
        cfe_state.soil_reservoir_storage_deficit_m = (
            cfe_state.soil_params["smcmax"] * cfe_state.soil_params["D"]
            - cfe_state.soil_reservoir["storage_m"]
        )

    # ____________________________________________________________________________________
    def calculate_infiltration_excess_overland_flow(self, cfe_state):
        """Calculates infiltration excess overland flow
        by running the partitioning scheme based on the choice set in the Configuration file
        """
        rainfall_mask = cfe_state.timestep_rainfall_input_m > 0.0
        schaake_mask = cfe_state.surface_partitioning_scheme == 1  # "Schaake"
        xinanjiang_mask = cfe_state.surface_partitioning_scheme == 2  # "Xinanjiang"

        if torch.any(rainfall_mask):
            if torch.any(schaake_mask):
                combined_mask = rainfall_mask & schaake_mask
                self.Schaake_partitioning_scheme(cfe_state, combined_mask)

            if torch.any(xinanjiang_mask):
                combined_mask = rainfall_mask & xinanjiang_mask
                self.Xinanjiang_partitioning_scheme(cfe_state, combined_mask)

            if not torch.any(schaake_mask | xinanjiang_mask):
                print(
                    "Problem: must specify one of Schaake or Xinanjiang partitioning scheme."
                )
                print("Program terminating.:( \n")
                return

    # __________________________________________________________________________________________________________
    def adjust_runoff_and_infiltration(self, cfe_state):
        """Calculates saturation excess overland flow (SOF)
        This should be run after calculate_infiltration_excess_overland_flow, then,
        infiltration_depth_m and surface_runoff_depth_m get finalized
        """
        # If the infiltration is more than the soil moisture deficit,
        # additional runoff (SOF) occurs and soil get saturated

        # Creating a mask where soil deficit is less than infiltration
        excess_infil_mask = (
            cfe_state.soil_reservoir_storage_deficit_m < cfe_state.infiltration_depth_m
        )

        # If there are any such basins, we apply the conditional logic element-wise
        if torch.any(excess_infil_mask):
            diff = (
                cfe_state.infiltration_depth_m
                - cfe_state.soil_reservoir_storage_deficit_m
            )[excess_infil_mask]

            # Adjusting the surface runoff and infiltration depths for the specific basins
            cfe_state.surface_runoff_depth_m[excess_infil_mask] = (
                cfe_state.surface_runoff_depth_m[excess_infil_mask] + diff
            )
            cfe_state.infiltration_depth_m[excess_infil_mask] = (
                cfe_state.infiltration_depth_m[excess_infil_mask] - diff
            )

            # Setting the soil reservoir storage deficit to zero for the specific basins
            cfe_state.soil_reservoir_storage_deficit_m[excess_infil_mask] = 0.0

        self.track_infiltration_and_runoff(cfe_state)

    # __________________________________________________________________________________________________________
    def track_infiltration_and_runoff(self, cfe_state):
        """Tracking runoff & infiltraiton volume with final infiltration & runoff values"""
        cfe_state.vol_partition_runoff += cfe_state.surface_runoff_depth_m.detach()
        cfe_state.vol_partition_infilt += cfe_state.infiltration_depth_m.detach()
        cfe_state.vol_to_soil += cfe_state.infiltration_depth_m.detach()

    # __________________________________________________________________________________________________________
    def run_soil_moisture_scheme(self, cfe_state):
        """Run the soil moisture scheme based on the choice set in the Configuration file"""
        if cfe_state.soil_params["scheme"].lower() == "classic":
            # Add infiltration flux and calculate the reservoir flux
            cfe_state.soil_reservoir["storage_m"] = (
                cfe_state.soil_reservoir["storage_m"] + cfe_state.infiltration_depth_m
            )
            self.soil_conceptual_reservoir_flux_calc(
                cfe_state=cfe_state, soil_reservoir=cfe_state.soil_reservoir
            )

        elif cfe_state.soil_params["scheme"].lower() == "ode":
            # Infiltration flux is added witin the ODE scheme
            self.soil_moisture_flux_calc_with_ode(
                cfe_state=cfe_state, reservoir=cfe_state.soil_reservoir
            )

    # ________________________________________________________________________________________________________
    def update_outflux_from_soil(self, cfe_state):
        cfe_state.flux_perc_m = cfe_state.primary_flux_m  # percolation_flux
        cfe_state.flux_lat_m = cfe_state.secondary_flux_m  # lateral_flux

        # If the soil moisture scheme is classic, take out the outflux from soil moisture storage
        # If ODE, outfluxes are already subtracted from the soil moisture storage
        if cfe_state.soil_params["scheme"].lower() == "classic":
            cfe_state.soil_reservoir["storage_m"] = (
                cfe_state.soil_reservoir["storage_m"] - cfe_state.flux_perc_m
            )
            cfe_state.soil_reservoir["storage_m"] = (
                cfe_state.soil_reservoir["storage_m"] - cfe_state.flux_lat_m
            )

        # If ODE, track actual ET from soil
        if cfe_state.soil_params["scheme"].lower() == "ode":
            cfe_state.vol_et_from_soil += (
                cfe_state.actual_et_from_soil_m_per_timestep.detach()
            )
            cfe_state.vol_et_to_atm += (
                cfe_state.actual_et_from_soil_m_per_timestep.detach()
            )
            cfe_state.volout += cfe_state.actual_et_from_soil_m_per_timestep.detach()
            cfe_state.actual_et_m_per_timestep += (
                cfe_state.actual_et_from_soil_m_per_timestep.detach()
            )
        elif cfe_state.soil_params["scheme"].lower() == "classic":
            None

    # ________________________________________________________________________________________________________
    def calculate_groundwater_storage_deficit(self, cfe_state):
        cfe_state.gw_reservoir_storage_deficit_m = (
            cfe_state.gw_reservoir["storage_max_m"]
            - cfe_state.gw_reservoir["storage_m"]
        )

    # __________________________________________________________________________________________________________
    def adjust_percolation_to_gw(self, cfe_state):
        overflow_mask = cfe_state.flux_perc_m > cfe_state.gw_reservoir_storage_deficit_m

        # When the groundwater storage is full, the overflowing amount goes to direct runoff
        if torch.any(overflow_mask):
            # Calculate the amount of overflow
            diff = (cfe_state.flux_perc_m - cfe_state.gw_reservoir_storage_deficit_m)[
                overflow_mask
            ].clone()

            # Overflow goes to surface runoff
            cfe_state.surface_runoff_depth_m[overflow_mask] = (
                cfe_state.surface_runoff_depth_m[overflow_mask] + diff
            )

            # Reduce the infiltration (maximum possible flux_perc_m is equal to gw_reservoir_storage_deficit_m)
            cfe_state.flux_perc_m[
                overflow_mask
            ] = cfe_state.gw_reservoir_storage_deficit_m[overflow_mask].clone()

            # Saturate the Groundwater storage
            cfe_state.gw_reservoir["storage_m"][overflow_mask] = cfe_state.gw_reservoir[
                "storage_max_m"
            ][overflow_mask].clone()
            cfe_state.gw_reservoir_storage_deficit_m[overflow_mask] = 0.0

            # Track volume
            cfe_state.vol_partition_runoff[overflow_mask] += diff.detach()
            cfe_state.vol_partition_infilt[overflow_mask] += diff.detach()

        # Otherwise all the percolation flux goes to the storage
        # Apply the "otherwise" part of your condition, to all basins where overflow_mask is False
        no_overflow_mask = ~overflow_mask
        if torch.any(no_overflow_mask):
            cfe_state.gw_reservoir["storage_m"][no_overflow_mask] = (
                cfe_state.gw_reservoir["storage_m"][no_overflow_mask]
                + cfe_state.flux_perc_m[no_overflow_mask].clone()
            )

    # __________________________________________________________________________________________________________
    def track_volume_from_percolation_and_lateral_flow(self, cfe_state):
        cfe_state.vol_to_gw += cfe_state.flux_perc_m.detach()
        cfe_state.vol_soil_to_gw += cfe_state.flux_perc_m.detach()
        cfe_state.vol_soil_to_lat_flow += cfe_state.flux_lat_m.detach()
        cfe_state.volout += cfe_state.flux_lat_m.detach()

    # __________________________________________________________________________________________________________

    def track_volume_from_gw(self, cfe_state):
        cfe_state.gw_reservoir["storage_m"] = (
            cfe_state.gw_reservoir["storage_m"]
            - cfe_state.flux_from_deep_gw_to_chan_m.clone()
        )
        # Mass balance
        cfe_state.vol_from_gw += cfe_state.flux_from_deep_gw_to_chan_m.detach()
        cfe_state.volout += cfe_state.flux_from_deep_gw_to_chan_m.detach()

    # __________________________________________________________________________________________________________
    def track_volume_from_giuh(self, cfe_state):
        cfe_state.vol_out_giuh += cfe_state.flux_giuh_runoff_m.detach()
        cfe_state.volout += cfe_state.flux_giuh_runoff_m.detach()

    # __________________________________________________________________________________________________________
    def track_volume_from_nash_cascade(self, cfe_state):
        cfe_state.vol_in_nash += cfe_state.flux_lat_m.detach()
        cfe_state.vol_out_nash += cfe_state.flux_nash_lateral_runoff_m.detach()

    # __________________________________________________________________________________________________________
    def add_up_total_flux_discharge(self, cfe_state):
        cfe_state.flux_Qout_m = (
            cfe_state.flux_giuh_runoff_m
            + cfe_state.flux_nash_lateral_runoff_m
            + cfe_state.flux_from_deep_gw_to_chan_m
        )
        cfe_state.total_discharge = (
            cfe_state.flux_Qout_m
            * cfe_state.catchment_area_km2
            * 1000000.0
            / cfe_state.time_step_size
        )

    # __________________________________________________________________________________________________________
    def update_current_time(self, cfe_state):
        cfe_state.current_time_step += 1
        cfe_state.current_time += pd.Timedelta(value=cfe_state.time_step_size, unit="s")
        if np.random.random() < 0.001:
            print(
                f"cfe line 462 --- Cgw: {cfe_state.Cgw}, cfe_state.soil_reservoir[coeff_primary]: {cfe_state.soil_reservoir['coeff_primary']}"
            )

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # MAIN MODEL FUNCTION
    def run_cfe(self, cfe_state):
        # Initialize the surface runoff
        self.initialize_flux(cfe_state)

        # Rainfall and ET
        self.calculate_input_rainfall_and_PET(cfe_state)
        self.calculate_evaporation_from_rainfall(cfe_state)

        if cfe_state.soil_params["scheme"].lower() == "classic":
            self.calculate_evaporation_from_soil(cfe_state)

        # Infiltration partitioning
        self.calculate_the_soil_moisture_deficit(cfe_state)
        self.calculate_infiltration_excess_overland_flow(cfe_state)
        self.adjust_runoff_and_infiltration(cfe_state)

        # Soil moisture reservoir
        # if cfe_state.infiltration_depth_m > 0:
        #     print('stop')
        self.run_soil_moisture_scheme(cfe_state)
        self.update_outflux_from_soil(cfe_state)

        # Groundwater reservoir
        self.calculate_groundwater_storage_deficit(cfe_state)
        self.adjust_percolation_to_gw(cfe_state)

        self.track_volume_from_percolation_and_lateral_flow(cfe_state)
        self.gw_conceptual_reservoir_flux_calc(
            cfe_state=cfe_state, gw_reservoir=cfe_state.gw_reservoir
        )
        self.track_volume_from_gw(cfe_state)

        # Surface runoff rounting
        # if cfe_state.surface_runoff_depth_m > 0.0:
        #     print('examine mass balance')
        self.convolution_integral(cfe_state)
        self.track_volume_from_giuh(cfe_state)

        # Lateral flow rounting
        self.nash_cascade(cfe_state)
        self.track_volume_from_nash_cascade(cfe_state)
        self.add_up_total_flux_discharge(cfe_state)

        # Time
        self.update_current_time(cfe_state)

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________

    # __________________________________________________________________________________________________________
    def nash_cascade(self, cfe_state):
        """
        Solve for the flow through the Nash cascade to delay the
        arrival of the lateral flow into the channel
        Currently only accepts the same number of nash reservoirs for all watersheds
        """

        num_reservoirs = cfe_state.num_lateral_flow_nash_reservoirs
        nash_storage = cfe_state.nash_storage.clone()

        # Calculate the discharge from each Nash storage
        Q = cfe_state.K_nash.T * nash_storage

        # Update Nash storage with discharge
        nash_storage = nash_storage - Q

        # The first storage receives the lateral flow outflux from soil storage
        nash_storage[:, 0] = nash_storage[:, 0] + cfe_state.flux_lat_m.squeeze()

        # The remaining storage receives the discharge from the upper Nash storage
        if num_reservoirs > 1:
            nash_storage[:, 1:] = nash_storage[:, 1:] + Q[:, :-1]

        # Update the state
        cfe_state.nash_storage = nash_storage.clone()

        # The final discharge at the timestep from Nash cascade is from the lowermost Nash storage
        cfe_state.flux_nash_lateral_runoff_m = Q[:, -1].clone().unsqueeze(dim=0)

        return

    def convolution_integral(self, cfe_state):
        """
        This function solves the convolution integral involving N GIUH ordinates.

        Inputs:
            Schaake_output_runoff_m
            num_giuh_ordinates
            giuh_ordinates
        Outputs:
            runoff_queue_m_per_timestep
        """
        N = cfe_state.num_giuh_ordinates

        # Set the last element in the runoff queue as zero (runoff_queue[:-1] were pushed forward in the last timestep)
        cfe_state.runoff_queue_m_per_timestep[:, N] = 0.0

        # Add incoming surface runoff to the runoff queue
        cfe_state.runoff_queue_m_per_timestep[
            :, :-1
        ] = cfe_state.runoff_queue_m_per_timestep[:, :-1] + (
            cfe_state.giuh_ordinates * cfe_state.surface_runoff_depth_m.expand(N, -1).T
        )

        # Take the top one in the runoff queue as runoff to channel
        cfe_state.flux_giuh_runoff_m = (
            cfe_state.runoff_queue_m_per_timestep[:, 0].clone().unsqueeze(dim=0)
        )

        # Shift all the entries forward in preperation for the next timestep
        cfe_state.runoff_queue_m_per_timestep[
            :, :-1
        ] = cfe_state.runoff_queue_m_per_timestep[:, 1:].clone()

        return

    # __________________________________________________________________________________________________________
    def et_from_rainfall(self, cfe_state, rainfall_mask):
        """
        iff it is raining, take PET from rainfall first.  Wet veg. is efficient evaporator.
        """

        # Applying the mask
        rainfall = cfe_state.timestep_rainfall_input_m[rainfall_mask]
        pet = cfe_state.potential_et_m_per_timestep[rainfall_mask]

        # If rainfall exceeds PET, actual AET from rainfall is equal to the PET
        # Otherwise, actual ET equals to potential ET
        condition = rainfall > pet

        actual_et_from_rain = torch.where(
            condition,
            pet,  # If P > PET, AET from P is equal to the PET
            rainfall,  # If P < PET, all P gets consumed as AET
        )

        reduced_rainfall = torch.where(
            condition,
            rainfall
            - actual_et_from_rain,  #  # If P > PET, part of P is consumed as AET
            torch.zeros_like(rainfall),  # If P < PET, all P gets consumed as AET
        )

        reduced_potential_et = pet - actual_et_from_rain

        # Storing the results back to the state
        cfe_state.actual_et_from_rain_m_per_timestep[
            rainfall_mask
        ] = actual_et_from_rain
        cfe_state.timestep_rainfall_input_m[rainfall_mask] = reduced_rainfall
        cfe_state.reduced_potential_et_m_per_timestep[
            rainfall_mask
        ] = reduced_potential_et

        return

    # __________________________________________________________________________________________________________
    ########## SINGLE OUTLET EXPONENTIAL RESERVOIR ###############
    ##########                -or-                 ###############
    ##########    TWO OUTLET NONLINEAR RESERVOIR   ###############
    def gw_conceptual_reservoir_flux_calc(self, cfe_state, gw_reservoir):
        """
        This function calculates the flux from a linear, or nonlinear
        conceptual reservoir with one or two outlets, or from an
        exponential nonlinear conceptual reservoir with only one outlet.
        In the non-exponential instance, each outlet can have its own
        activation storage threshold.  Flow from the second outlet is
        turned off by setting the discharge coeff. to 0.0.
        """

        # This is basically only running for GW, so changed the variable name from primary_flux to primary_flux_from_gw_m to avoid confusion
        # if reservoir['is_exponential'] == True:
        flux_exponential = torch.exp(
            gw_reservoir["exponent_primary"]
            * gw_reservoir["storage_m"]
            / gw_reservoir["storage_max_m"]
        ) - torch.ones((1, cfe_state.num_basins), dtype=torch.float64)
        cfe_state.primary_flux_from_gw_m = torch.minimum(
            cfe_state.Cgw * flux_exponential, gw_reservoir["storage_m"]
        ).clone()

        cfe_state.secondary_flux_from_gw_m = torch.zeros(
            (1, cfe_state.num_basins), dtype=torch.float64
        )

        cfe_state.flux_from_deep_gw_to_chan_m = (
            cfe_state.primary_flux_from_gw_m + cfe_state.secondary_flux_from_gw_m
        )

        return

    def soil_conceptual_reservoir_flux_calc(self, cfe_state, soil_reservoir):
        # Calculate primary flux
        storage_above_threshold_primary = (
            soil_reservoir["storage_m"] - soil_reservoir["storage_threshold_primary_m"]
        )
        primary_flux_mask = storage_above_threshold_primary > 0.0

        if torch.any(primary_flux_mask):
            storage_diff_primary = (
                soil_reservoir["storage_max_m"]
                - soil_reservoir["storage_threshold_primary_m"]
            )
            storage_ratio_primary = (
                storage_above_threshold_primary / storage_diff_primary
            )
            storage_power_primary = torch.pow(
                storage_ratio_primary, soil_reservoir["exponent_primary"]
            )
            primary_flux = soil_reservoir["coeff_primary"] * storage_power_primary

            cfe_state.primary_flux_m[primary_flux_mask] = torch.minimum(
                primary_flux, storage_above_threshold_primary
            )[primary_flux_mask]

        # Calculate secondary flux
        storage_above_threshold_secondary = (
            soil_reservoir["storage_m"]
            - soil_reservoir["storage_threshold_secondary_m"]
        )
        secondary_flux_mask = storage_above_threshold_secondary > 0.0

        if torch.any(secondary_flux_mask):
            storage_diff_secondary = (
                soil_reservoir["storage_max_m"]
                - soil_reservoir["storage_threshold_secondary_m"]
            )
            storage_ratio_secondary = (
                storage_above_threshold_secondary / storage_diff_secondary
            )
            storage_power_secondary = torch.pow(
                storage_ratio_secondary, soil_reservoir["exponent_secondary"]
            )
            secondary_flux = soil_reservoir["coeff_secondary"] * storage_power_secondary

            cfe_state.secondary_flux_m[secondary_flux_mask] = torch.minimum(
                secondary_flux,
                storage_above_threshold_secondary - cfe_state.primary_flux_m,
            )[secondary_flux_mask]

        return

    # __________________________________________________________________________________________________________
    #  SCHAAKE RUNOFF PARTITIONING SCHEME
    def Schaake_partitioning_scheme(self, cfe_state, combined_mask):
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

        rainfall = cfe_state.timestep_rainfall_input_m[combined_mask]
        deficit = cfe_state.soil_reservoir_storage_deficit_m[combined_mask]
        magic_const = cfe_state.Schaake_adjusted_magic_constant_by_soil_type[
            combined_mask
        ]
        timestep_d = cfe_state.timestep_d

        exp_term = torch.exp(-magic_const * timestep_d)
        Ic = deficit * (1 - exp_term)
        Px = rainfall
        infilt = Px * (Ic / (Px + Ic))

        # If the rainfall > infiltration, runoff is generated
        # If rainfall < infiltration, no runoff, all of the preciptiation are infiltratied
        runoff = torch.where(
            rainfall - infilt > 0, rainfall - infilt, torch.zeros_like(rainfall)
        )
        infilt = rainfall - runoff

        cfe_state.surface_runoff_depth_m[combined_mask] = runoff
        cfe_state.infiltration_depth_m[combined_mask] = infilt

        return

    # __________________________________________________________________________________________________________
    def Xinanjiang_partitioning_scheme(self, cfe_state):
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
        free_water_m = (
            cfe_state.soil_reservoir["storage_m"]
            - cfe_state.soil_reservoir["storage_threshold_primary_m"]
        )

        if 0.0 < free_water_m:
            tension_water_m = cfe_state.soil_reservoir["storage_threshold_primary_m"]

        else:
            free_water_m = torch.zeros((1, self.num_basins), dtype=torch.float64)
            tension_water_m = cfe_state.soil_reservoir["storage_m"]

        # estimate the maximum free water and tension water available in the soil column
        max_free_water_m = (
            cfe_state.soil_reservoir["storage_max_m"]
            - cfe_state.soil_reservoir["storage_threshold_primary_m"]
        )
        max_tension_water_m = cfe_state.soil_reservoir["storage_threshold_primary_m"]

        # check that the free_water_m and tension_water_m do not exceed the maximum and if so, change to the max value
        if max_free_water_m < free_water_m:
            free_water_m = max_free_water_m

        if max_tension_water_m < tension_water_m:
            tension_water_m = max_tension_water_m

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
        a_Xinanjiang_inflection_point_parameter = torch.ones(
            (1, self.num_basins), dtype=torch.float64
        )
        b_Xinanjiang_shape_parameter = torch.ones(
            (1, self.num_basins), dtype=torch.float64
        )
        x_Xinanjiang_shape_parameter = torch.ones(
            (1, self.num_basins), dtype=torch.float64
        )

        if (tension_water_m / max_tension_water_m) <= (
            0.5 * torch.ones((1, self.num_basins), dtype=torch.float64)
            - a_Xinanjiang_inflection_point_parameter
        ):
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m * (
                torch.pow(
                    (
                        0.5 * torch.ones((1, self.num_basins), dtype=torch.float64)
                        - a_Xinanjiang_inflection_point_parameter
                    ),
                    (
                        torch.ones((1, self.num_basins), dtype=torch.float64)
                        - b_Xinanjiang_shape_parameter
                    ),
                )
                * torch.pow(
                    (
                        torch.ones((1, self.num_basins), dtype=torch.float64)
                        - (tension_water_m / max_tension_water_m)
                    ),
                    b_Xinanjiang_shape_parameter,
                )
            )

        else:
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m * (
                torch.ones((1, self.num_basins), dtype=torch.float64)
                - torch.pow(
                    (
                        0.5 * torch.ones((1, self.num_basins), dtype=torch.float64)
                        + a_Xinanjiang_inflection_point_parameter
                    ),
                    (
                        torch.ones((1, self.num_basins), dtype=torch.float64)
                        - b_Xinanjiang_shape_parameter
                    ),
                )
                * torch.pow(
                    (
                        torch.ones((1, self.num_basins), dtype=torch.float64)
                        - (tension_water_m / max_tension_water_m)
                    ),
                    (b_Xinanjiang_shape_parameter),
                )
            )

        # Separate the surface water from the pervious runoff
        ## NOTE: If impervious runoff is added to this subroutine, impervious runoff should be added to
        ## the surface_runoff_depth_m.

        cfe_state.surface_runoff_depth_m = pervious_runoff_m * (
            0.5 * torch.ones((1, self.num_basins), dtype=torch.float64)
            - torch.pow(
                (
                    0.5 * torch.ones((1, self.num_basins), dtype=torch.float64)
                    - (free_water_m / max_free_water_m)
                ),
                x_Xinanjiang_shape_parameter,
            )
        )

        # The surface runoff depth is bounded by a minimum of 0 and a maximum of the water input depth.
        # Check that the estimated surface runoff is not less than 0.0 and if so, change the value to 0.0.
        if cfe_state.surface_runoff_depth_m < 0.0:
            cfe_state.surface_runoff_depth_m = torch.zeros(
                (1, self.num_basins), dtype=torch.float64
            )

        # Check that the estimated surface runoff does not exceed the amount of water input to the soil surface.  If it does,
        # change the surface water runoff value to the water input depth.
        if cfe_state.surface_runoff_depth_m > cfe_state.timestep_rainfall_input_m:
            cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m

        # Separate the infiltration from the total water input depth to the soil surface.
        cfe_state.infiltration_depth_m = (
            cfe_state.timestep_rainfall_input_m - cfe_state.surface_runoff_depth_m
        )

        return

    # __________________________________________________________________________________________________________
    def et_from_soil(self, cfe_state, combined_mask):
        """
        Take AET from soil moisture storage,
        using Budyko type curve to limit PET if wilting<soilmoist<field_capacity
        """
        storage = cfe_state.soil_reservoir["storage_m"][combined_mask]
        threshold = cfe_state.soil_reservoir["storage_threshold_primary_m"][
            combined_mask
        ]
        wilting_point = cfe_state.soil_reservoir["wilting_point_m"][combined_mask]
        reduced_pet = cfe_state.reduced_potential_et_m_per_timestep[combined_mask]

        condition1 = storage >= threshold
        condition2 = (storage > wilting_point) & (storage < threshold)

        actual_et_from_soil = torch.where(
            condition1,
            torch.minimum(
                reduced_pet, storage
            ),  # If storage is above the FC threshold, AET = PET
            torch.where(
                condition2,
                torch.minimum(
                    (storage - wilting_point)
                    / (threshold - wilting_point)
                    * reduced_pet,
                    storage,
                ),  # If storage is in bewteen the FC and WP threshold, calculate the Budyko type of AET
                torch.zeros_like(storage),  # If storage is less than WP, AET=0
            ),
        )

        cfe_state.actual_et_from_soil_m_per_timestep[
            combined_mask
        ] = actual_et_from_soil
        cfe_state.soil_reservoir["storage_m"][combined_mask] = (
            cfe_state.soil_reservoir["storage_m"][combined_mask] - actual_et_from_soil
        )
        cfe_state.reduced_potential_et_m_per_timestep[combined_mask] = (
            cfe_state.reduced_potential_et_m_per_timestep[combined_mask]
            - actual_et_from_soil
        )

        return

    # # __________________________________________________________________________________________________________
    # # __________________________________________________________________________________________________________
    # def sm_ode_one_basin(self, i, cfe_state, reservoir):
    #     # Initialization

    #     y0 = reservoir["storage_m"][0, i]

    #     t = torch.tensor(
    #         [0, 0.05, 0.15, 0.3, 0.6, 1.0]
    #     )  # ODE time descritization of one time step

    #     # Pass parameters beforehand
    #     func = soil_moisture_flux_ode(i=i, cfe_state=cfe_state, reservoir=reservoir).to(
    #         cfe_state.cfg.device
    #     )

    #     # Solve and ODE
    #     # Use Differentiable ODE package for Torch tensors from here https://github.com/rtqichen/torchdiffeq
    #     sol = odeint(
    #         func,
    #         y0,
    #         t,
    #         # atol=1e-5,
    #         # rtol=1e-5,
    #         # adjoint_params=()
    #     )

    #     # Finalize results
    #     ts_concat = t
    #     ys_concat = sol.squeeze(dim=-1)
    #     t_proportion = torch.diff(ts_concat, dim=0)  # ts_concat[1:] - ts_concat[:-1]

    #     # Create the kernel tensor with torch.ones
    #     kernel = torch.ones(2)

    #     # Get the moving average y values in between the time intervals
    #     convolved = F.conv1d(
    #         ys_concat.float().unsqueeze(dim=0).unsqueeze(dim=0),
    #         kernel.float().unsqueeze(dim=0).unsqueeze(dim=0),
    #         padding=1,
    #     ).squeeze()
    #     # Divide by 2 to match np.convolve
    #     ys_avg_ = convolved.clone() / 2
    #     ys_avg = ys_avg_[1:-1].clone()

    #     return ys_avg, t_proportion, ys_concat

    def soil_moisture_flux_calc_with_ode(self, cfe_state, reservoir):
        """
        This function solves the soil moisture mass balance.
        Inputs:
            reservoir
        Outputs:
            primary_flux_m (percolation)
            secondary_flux_m (lateral flow)
            actual_et_from_soil_m_per_timestep (et_from_soil)
        """

        y0 = reservoir["storage_m"]  # [0, i]

        t = torch.tensor(
            [0, 0.05, 0.15, 0.3, 0.6, 1.0], dtype=torch.float64
        )  # ODE time descritization of one time step

        # Pass parameters beforehand
        func = soil_moisture_flux_ode(cfe_state=cfe_state, reservoir=reservoir).to(
            cfe_state.cfg.device
        )

        # Solve and ODE
        # Use Differentiable ODE package for Torch tensors from here https://github.com/rtqichen/torchdiffeq
        sol = odeint(
            func,
            y0,
            t,
            # atol=1e-5,
            # rtol=1e-5,
            # adjoint_params=()
        )

        # Finalize results
        ts_concat = t
        ys_concat = sol.squeeze(dim=-1).to(torch.float64)
        t_proportion = torch.diff(ts_concat, dim=0)  # ts_concat[1:] - ts_concat[:-1]

        # Create the kernel tensor with torch.ones
        kernel = torch.ones(2)

        # Get the moving average y values in between the time intervals
        ys_concat_2d = ys_concat.squeeze().float()  # No permutation needed
        kernel_1d = kernel.float().squeeze()

        # Applying the convolution separately for each channel and storing the results in a list
        convolved_list = [
            F.conv1d(
                y.unsqueeze(0).unsqueeze(0),
                kernel_1d.unsqueeze(0).unsqueeze(0),
                padding=1,
            )
            for y in ys_concat_2d.T
        ]

        # Stacking the results together to get the final convolved tensor
        convolved = torch.cat(convolved_list, dim=0).squeeze()

        # Divide by 2 to match np.convolve
        ys_avg_ = convolved.clone() / 2
        ys_avg = ys_avg_[:, 1:-1].clone().T

        # Still not sure batch ODE is possible ...
        # # initialize output
        # y0 = reservoir["storage_m"].clone()

        # ys_avg = torch.zeros_like(y0)
        # t_proportion = torch.zeros_like(y0)
        # ys_concat = torch.zeros_like(y0)

        # for i in range(cfe_state.num_basins + 1):
        #     ys_avg, t_proportion, ys_concat = self.sm_ode_one_basin(
        #         i, cfe_state, reservoir
        #     )

        # Get each flux values and scale it

        ## Get parameters
        num_timesteps = len(ys_avg)  # or however you determine the number of timesteps
        batch_threshold_primary = reservoir["storage_threshold_primary_m"].repeat(
            num_timesteps, 1
        )
        batch_storage_max_m = reservoir["storage_max_m"].repeat(num_timesteps, 1)
        batch_coeff_primary = reservoir["coeff_primary"].repeat(num_timesteps, 1)
        batch_coeff_secondary = reservoir["coeff_secondary"].repeat(num_timesteps, 1)
        batch_t_proportion = t_proportion.repeat(cfe_state.num_basins, 1).T
        batch_wilting_point_m = reservoir["coeff_secondary"].repeat(num_timesteps, 1)
        batch_PET = cfe_state.reduced_potential_et_m_per_timestep.repeat(
            num_timesteps, 1
        )
        batch_infilt = torch.tensor(cfe_state.infiltration_depth_m.clone()).repeat(
            num_timesteps, 1
        )

        # Calculate lateral_flux and percolation_flux
        storage_above_threshold_m = ys_avg - batch_threshold_primary
        storage_diff = batch_storage_max_m - batch_threshold_primary
        storage_ratio = torch.clamp(
            storage_above_threshold_m / storage_diff, max=1.0, min=0.0
        )

        lateral_flux = storage_ratio * batch_coeff_secondary
        lateral_flux_frac = lateral_flux * batch_t_proportion

        perc_flux = storage_ratio * batch_coeff_primary
        perc_flux_frac = perc_flux * batch_t_proportion

        # Calculate ET from soil
        storage_above_threshold_m_paw = ys_avg - batch_wilting_point_m
        storage_diff_paw = batch_threshold_primary - batch_wilting_point_m
        storage_ratio_paw = torch.clamp(
            storage_above_threshold_m_paw / storage_diff_paw, max=1.0, min=0.0
        )  # Equation 11 (Ogden's document)

        et_from_soil = batch_PET * storage_ratio_paw
        et_from_soil_frac = et_from_soil * batch_t_proportion

        # Infiltration
        infilt_to_soil_frac = batch_infilt * batch_t_proportion

        # Scale fluxes (Since the sum of all the estimated flux above usually exceed the input flux because of calculation errors, scale it
        # The more finer ODE time descritization you use, the less errors you get, but the more calculation time it takes

        sum_outflux = lateral_flux_frac + perc_flux_frac + et_from_soil_frac

        flux_scale = torch.zeros((cfe_state.num_basins,), dtype=torch.float64)
        nonzero_mask = torch.sum(sum_outflux, dim=0) != 0
        flux_scale[nonzero_mask] = (
            (ys_concat[0] - ys_concat[-1]) + torch.sum(infilt_to_soil_frac, dim=0)
        ).squeeze()[nonzero_mask] / torch.sum(sum_outflux, dim=0)[nonzero_mask]

        # Handle the case when sum_outflux is zero
        zero_mask = ~nonzero_mask
        final_storage_m = torch.zeros((cfe_state.num_basins,), dtype=torch.float64)
        final_storage_m[zero_mask] = (
            y0[zero_mask] + cfe_state.infiltration_depth_m[0][zero_mask]
        )
        final_storage_m[nonzero_mask] = ys_concat[-1][0][nonzero_mask]

        # if torch.sum(sum_outflux) == 0:
        #     flux_scale = torch.zeros((1, self.num_basins), dtype=torch.float64)
        #     if cfe_state.infiltration_depth_m > 0:
        #         # To account for mass balance error by ODE
        #         final_storage_m = y0 + cfe_state.infiltration_depth_m
        #     else:
        #         final_storage_m = y0
        # else:
        #     flux_scale = (
        #         (ys_concat[0] - ys_concat[-1]) + torch.sum(infilt_to_soil_frac)
        #     ) / torch.sum(sum_outflux)
        #     final_storage_m = ys_concat[-1].clone()

        scaled_lateral_flux = lateral_flux_frac * flux_scale
        scaled_perc_flux = perc_flux_frac * flux_scale
        scaled_et_flux = et_from_soil_frac * flux_scale

        # Pass the results
        # ? Do these all gets tracked?
        cfe_state.primary_flux_m = torch.sum(scaled_perc_flux, dim=0)
        cfe_state.secondary_flux_m = torch.sum(scaled_lateral_flux, dim=0)
        cfe_state.actual_et_from_soil_m_per_timestep = torch.sum(scaled_et_flux, dim=0)
        reservoir["storage_m"] = final_storage_m

        sm_mass_balance_timestep = (
            y0
            - final_storage_m
            + cfe_state.infiltration_depth_m
            - cfe_state.primary_flux_m
            - cfe_state.secondary_flux_m
            - cfe_state.actual_et_from_soil_m_per_timestep
        )
        if torch.any(sm_mass_balance_timestep) > 1e-09:
            print("mass balance error")

        # print(f'primary_flux_m: {primary_flux_m}')
        # print(f'secondary_flux_m: {secondary_flux_m}')
        # print(f'actual_et_from_soil_m_per_timestep: {actual_et_from_soil_m_per_timestep}')
        # print(f'reservoir["storage_m"]: {reservoir["storage_m"]}')

        return
# ___________________end of content in cfe.py______________

