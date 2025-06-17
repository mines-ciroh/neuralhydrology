# original shm packages
from typing import Dict, Union

import torch

import neuralhydrology.utils.CFE_modules as cfe_module
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.constants import BASIN_CHARACTERISTIC_KEYS, SOIL_KEYS

# packages from cfe.py
# import time
# import sys
# import math
# import torch
# from torchdiffeq import odeint

# packages from bmi_cfe.py


class dCFE(BaseConceptualModel):
    """
    This is an attempt to make a dCFE model based on
    https://github.com/NWC-CUAHSI-Summer-Institute/ngen-aridity/blob/main/Project%20Manuscript_LongForm.pdf
    General outline:
    Takes raw LSTM output, shape them within possible ranges of the Cgw and satdk parameters.
    Together with other basin-specific parameters and forcings (precip, and srad + tmean for pet)
    and pass through the CFE for runoff predictions.
    Spin-up period = warm_up do not have gradient tracking.  The states here are used to run one more time for prediction.
    This model is tailored to basin ID: 02177000 CHATTOOGA RIVER NEAR CLAYTON, GA right now, but can be worked on
    later to train multi-basin.

    The physics is done and forward process & backward processes work so far with this specific basin and time-period of data.
    There's no snow module.

    TODO:
    Improve readability
    Integrate multi-basin training
    """

    def __init__(self, cfg: Config):
        super(dCFE, self).__init__(cfg=cfg)

        self.cfg = cfg
        # self.temp_soil_params, self.temp_basinCharacteristics = get_dcfe_params(cfg=cfg, device=cfg.device)

    def forward(
        self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor, additional_features: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
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

        # Fetch dcfe params
        # batch_size = x_conceptual.shape[0]
        # TODO: @Ziyu previously these were default_soil_params and default_basinCharacteristics.
        # I think that this was a legacy name from earlier code. Could you confirm droppign the prefix "default_" is okay?
        self.soil_params = {k: additional_features[k] for k in SOIL_KEYS}
        self.basinCharacteristics = {k: additional_features[k] for k in BASIN_CHARACTERISTIC_KEYS}

        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual, lstm_out=lstm_out)

        # use basin params from HydroShare to initialize other constants
        # TODO: Changed the below from default_soil_params and default_basinCharacteristics to self.soil_params and self.basinCharacteristics.
        cfe_calibrated_params = {"soil_params": self.soil_params, "basinCharacteristics": self.basinCharacteristics}

        # TODO: replace the code below with a single function call.
        timestep_spinup_params = {
            "satdk": self.soil_params["satdk"],
            "Cgw": self.basinCharacteristics["Cgw"],
            "bb": self.soil_params["bb"],
            "smcmax": self.soil_params["smcmax"],
            "slop": self.soil_params["slop"],
            "max_gw_storage": self.basinCharacteristics["max_gw_storage"],
            "expon": self.basinCharacteristics["expon"],
            "K_lf": self.basinCharacteristics["K_lf"],
            "K_nash": self.basinCharacteristics["K_nash"],
            "satpsi": self.soil_params["satpsi"],
        }

        constants, cfe_params, gw_reservoir, soil_reservoir, routing_info, flux = cfe_module.initialize_basin_constants(
            cfg=self.cfg, conceptual_forcing=x_conceptual, cfe_params=cfe_calibrated_params, hourly=self.cfg.dcfe_hourly
        )

        # Spin up for warm_up amount of time, do not track gradient
        with torch.no_grad():
            for j in range(0, (lstm_out.shape[1] - self.cfg.spin_up)):
                # run the CFE model for the time step w/ Hydroshare Params
                cfe_params, gw_reservoir, soil_reservoir, routing_info, flux = cfe_module.timestep_CFE_new(
                    x_conceptual_timestep=x_conceptual[:, j, :],
                    cfe_params=cfe_params,
                    timestep_parameters=timestep_spinup_params,
                    constants=constants,
                    gw_reservoir=gw_reservoir,
                    soil_reservoir=soil_reservoir,
                    routing_info=routing_info,
                    flux=flux,
                    hourly=self.cfg.dcfe_hourly,
                )

                out[:, j, 0] = flux["Qout_m"] * 1000  # this will not be included for back prop due to predict_last_n
                # store resulting states, right now this doesn't do anything
                states["gw_reservoir_storage_m"][:, j] = gw_reservoir["storage_m"]
                states["soil_reservoir_storage_m"][:, j] = soil_reservoir["storage_m"]
                states["first_nash_storage"][:, j] = cfe_params["basinCharacteristics"]["nash_storage"][:, 0]

        torch.autograd.set_detect_anomaly(True)
        spin_up_start = self.cfg.spin_up

        # Run model for prediction for each parameters
        # TODO: Is the code below necessary? If yes, can we replace it with a single function call or list comprehension?
        timestep_avg_params = {
            "satdk": parameters["satdk"][:, spin_up_start:].mean(dim=1),
            "Cgw": parameters["Cgw"][:, spin_up_start:].mean(dim=1),
            "bb": parameters["bb"][:, spin_up_start:].mean(dim=1),
            "smcmax": parameters["smcmax"][:, spin_up_start:].mean(dim=1),
            "slop": parameters["slop"][:, spin_up_start:].mean(dim=1),
            "max_gw_storage": parameters["max_gw_storage"][:, spin_up_start:].mean(dim=1),
            "expon": parameters["expon"][:, spin_up_start:].mean(dim=1),
            "K_lf": parameters["K_lf"][:, spin_up_start:].mean(dim=1),
            "K_nash": parameters["K_nash"][:, spin_up_start:].mean(dim=1),
            "satpsi": parameters["satpsi"][:, spin_up_start:].mean(dim=1),
        }

        for k in range(spin_up_start, lstm_out.shape[1]):
            ##### run CFE for that time step, w/ avg params

            cfe_params, gw_reservoir, soil_reservoir, routing_info, flux = cfe_module.timestep_CFE_new(
                x_conceptual_timestep=x_conceptual[:, k, :],
                cfe_params=cfe_params,
                timestep_parameters=timestep_avg_params,
                constants=constants,
                gw_reservoir=gw_reservoir,
                soil_reservoir=soil_reservoir,
                routing_info=routing_info,
                flux=flux,
                hourly=self.cfg.dcfe_hourly,
            )

            # store states
            states["gw_reservoir_storage_m"][:, k] = gw_reservoir["storage_m"]
            states["soil_reservoir_storage_m"][:, k] = soil_reservoir["storage_m"]
            states["first_nash_storage"][:, k] = cfe_params["basinCharacteristics"]["nash_storage"][:, 0]

            # store runoff for back-prop
            out[:, k, 0] = (
                flux["Qout_m"] * 1000
            )  # * self.basinCharacteristics['catchment_area_km2'] * 1000000.0 / self.time_step_size

        return {"y_hat": out, "parameters": parameters, "internal_states": states}

    # ______________________defining states and parameter properties relavent to NH________________
    # TODO: Move these to constants.py
    @property
    def initial_states(self):
        return {
            "gw_reservoir_storage_m": 0.5,
            "soil_reservoir_storage_m": 0.6,
            "first_nash_storage": 0.0,
        }  # There are more storage/fluxes but doesn't matter cuz we can just grab whatever I want

    @property
    def parameter_ranges(self):
        return {
            "satdk": [0.0, 0.000726],  # Saturated hydraulic conductivity [m/hr]
            "Cgw": [0.0000018, 0.0018],  # Primary groundwater reservoir constant [m/hr]
            "bb": [0, 21.94],  # exponent on Clapp-Hornberg functin [-]
            "smcmax": [0.20554, 1],  # Max soil moisture content [m3/hr3]
            "slop": [0, 1],  # slope coefficient [-]
            "max_gw_storage": [0.01, 0.25],  # [m]
            "expon": [1, 8],  # A primary gorundwter nonlinear reservoir exponential constant [-]
            "K_lf": [0, 1],  # Lateral flow coefficient
            "K_nash": [0, 1],  # Nash cascade discharge coefficient
            "satpsi": [0.05, 0.95],
        }
