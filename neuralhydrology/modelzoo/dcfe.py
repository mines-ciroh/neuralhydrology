import torch
import torch.nn as nn
from typing import Dict, Union
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config

class dCFE(BaseConceptualModel):
    """
    This is an attempt to make a dCFE model based on 
    https://github.com/NWC-CUAHSI-Summer-Institute/ngen-aridity/blob/main/Project%20Manuscript_LongForm.pdf
    
    Last edited: by Ziyu, 07/08/2024
    
    General outline:
    The model takes in basin attributes to construct the 9 calibration parameters in the CFE 
    from https://github.com/NWC-CUAHSI-Summer-Institute/ngen-aridity/blob/main/Supplemental_Information.md
    using some kind of NN architecture. This is done because the calibration parameters are basin specific. 
    The calibration parameters as the result of the NN gets passed into the CFE, where together with dynamic 
    forcing data, a runoff is computed and compared to the actual runoff using a loss function. Then, gradients 
    are computed and back-propogated to update the weights in the NN to result in a better better set of the 9 
    predicted calibration parameters. 
    
    """
    
    def __init__(self, cfg: Config):
        super(dCFE, self).__init__(cfg=cfg)
        
        # initialize the 9 parameters
            # bb = ... # Exponent on Clapp-Hornberger (1978) function, unitless
            # smcmax = ... # Maximum soil moisture content, m^3/m^3
            # satdk = ... # Saturated hydraulic conductivity, m/hr
            # slop = ... # Slope coefficient, unit free
            # max_gw_storage = ... # Max groundwater storage, m
            # expon = ... # Primary groundwater nonlinear reservoir exponential constant, unitless
            # Cgw = ... # Primary groundwater nonlinear reservoir constant, m/hr
            # K_lf = ... # Lateral flow coefficient, unitless
            # K_nash = ... # Nash cascade discharge coefficient, unitless
        
        # initialize CFE states somehow...
            # state_groundwater = ...
            # state_...
        
        # initializing the NN component
        self.layer1 = nn.Linear(n_inputs_Attrib, 50) # input basin attributes
        self.layer2 = nn.Linear(50, 9) # outputting calibration parameters, all 9 of them
        
    def forward(self, x_attributes: torch.Tensor, x_dynamic: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Perform a forward pass. 

        Parameters
        ----------
        x_attributes: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs_attrib]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. 
            This will be used in the NN part only to get the 9 calibration parameters.
            
        x_dynamic: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs_dynamic]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. 
            This will be inputed into the CFE part of the model to compute run-off.
            
        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - q_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
                There would be 9 parameters.
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model

        """
        x = torch.relu(self.layer1(x_attributes)) # applyng first layer
        x = self.layer2(x) # applying 2nd layer
        
        # get model params thru baseconceptualmodel.py's function, 
        # this ensure that the output from NN is within the correct range
        parameters = self._get_dynamic_parameters_conceptual(x)
        
        # pass parameters thru CFE to get runoff and the states of CFE
        q_hat, states = CFE(parameters, x_dynamic)
            # Could do cfe_instance = BMI_CFE(
                # Cgw = self.Cgw, satdk = self.satdk, cfg = self. cfg, 
                # cfe_params = self.data.params,
            #)
            # But what about the other 7 calibration parameters? Do those all go in above?
            
            
            
        return {'q_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {'ss': 0.0,
                'sf': 1.0,
                'su': 5.0,
                'si': 10.0,
                'sb': 15.0} ## we need to define these differently for CFE

    @property
    def parameter_ranges(self):
        return {'bb': [0.0, 21.94], # Exponent on Clapp-Hornberger (1978) function
                'smcmax': [0.20554, 1.0], # Maximum soil moisture content
                'satdk': [0.0, 0.000726], # Saturated hydraulic conductivity
                'slop': [0.0, 1.0], # Slope coefficient
                'max_gw_storage': [0.01, 0.25], # Max groundwater storage
                'expon': [1.0, 8.0], # Primary groundwater exponential constant
                'Cgw': [0.0000018, 0.0018], # ^^ reservoir constant
                'K_lf': [0.0, 1.0], # Lateral flow coefficient
                'K_nash': [0.0, 1.0] # Nash cascade discharge coefficient
                }
