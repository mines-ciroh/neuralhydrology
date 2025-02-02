from pathlib import Path
import torch
import re
import os 
from neuralhydrology.utils.config import Config

def get_dcfe_params(cfg, device):
    cfe_param_dir = cfg.param_dir
    basin_id = str(cfg.basin_id)
    if basin_id[0] != '0':
        basin_id = '0' + basin_id
    cfe_param_file_path = cfe_param_dir / (basin_id + '_bmi_config_cfe_pass.txt')
    with open(cfe_param_file_path, 'r') as f:
        content = f.read()
    f.close()
    pattern = r'([\w.]+)\s*=\s*([0-9.eE+-]+(?:,\s*[0-9.eE+-]+)*)'
    matches_list = re.findall(pattern, content)
    matches = {}
    for match in matches_list:
        try:
            matches[match[0]] = float(match[1])
        except:
            matches[match[0]] = [float(x) for x in match[1].split(',')]
    soil_params = {'depth': torch.tensor(matches["soil_params.depth"],device=device,dtype=torch.float32),
                   'bb': torch.tensor(matches["soil_params.b"],device=device,dtype=torch.float32),
                   'satdk': torch.tensor(matches["soil_params.satdk"],device=device,dtype=torch.float32),
                   'satpsi': torch.tensor(matches["soil_params.satpsi"],device=device,dtype=torch.float32),
                   'slop': torch.tensor(matches["soil_params.slop"],device=device,dtype=torch.float32),
                   'smcmax': torch.tensor(matches["soil_params.smcmax"],device=device,dtype=torch.float32),
                   'wltsmc': torch.tensor(matches["soil_params.wltsmc"],device=device,dtype=torch.float32),
                   'D': torch.tensor(2.0, device=device, dtype=torch.float32),
                   'mult': torch.tensor(1.0, device=device, dtype=torch.float32),
                   }
    basinCharacteristics = {'catchment_area_km2': torch.tensor(526.77, device=device, dtype=torch.float32),
                            'refkdt': torch.tensor(matches["refkdt"], device=device, dtype=torch.float32),
                            'max_gw_storage': torch.tensor(matches['max_gw_storage'], device=device, dtype=torch.float32),
                            'expon': torch.tensor(matches['expon'], device=device, dtype=torch.float32),
                            'alpha_fc': torch.tensor(matches['alpha_fc'], device=device, dtype=torch.float32),
                            'K_nash': torch.tensor(matches['K_nash'], device=device, dtype=torch.float32),
                            'K_lf': torch.tensor(matches['K_lf'], device=device, dtype=torch.float32),
                            'nash_storage': torch.tensor(matches['nash_storage'], device=device, dtype=torch.float32),
                            'giuh_ordinates': torch.tensor(matches['giuh_ordinates'], device=device, dtype=torch.float32),
                    }       
    return soil_params, basinCharacteristics


def expand_dcfe_params_along_batch_dim(params, batch_size):
    '''
    Input: params: dict of basin specific paraemters for dcfe.
    Output: new_params: dict of basin specific paramters for dcfe, but expanded along the batch dimension.
    '''
    new_params = {}
    for key in params.keys():
        if key != 'giuh_ordinates':
            new_params[key] = params[key].expand(batch_size, *[-1 for _ in range(len(params[key].shape))])
        elif key == 'giuh_ordinates':
            new_params[key] = params[key] # need to treat this parameter differently
    return new_params