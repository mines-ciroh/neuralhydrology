import json
import re
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from neuralhydrology.datautils import utils


def get_dcfe_params(cfg):
    """This function reads the config file, grabs HydroShare params needed for CFE, and returns a basin-index dataframe
    with the parameters for each basin in the training list.

    These parameters are a combo of default CFE parameters and calibrated parameters from the JSON files.

    Args:
        cfg: configuration
        device: ??

    Returns:
        df: dataframe inexed by basin ids with 2 columns, soil_params and basinCharacteristics, which are each dicts of parameters
    """

    cfe_param_dir = cfg.param_dir
    calibrated_params_dir = cfg.calibrated_params_path
    
    # --- get all the basin ids as strings ---
    basins = utils.load_basin_file(getattr(cfg, "train_basin_file"))
    
    col_keys = keys['soil'] + keys['basin_characteristics']
    df = pd.DataFrame(index=basins, columns=col_keys)

    # --- iterate thru each basin id to get params ---
    for basin_id in basins:
        cfe_param_file_path = cfe_param_dir / (basin_id + "_bmi_config_cfe_pass.txt")
        with open(cfe_param_file_path, "r") as f:
            content = f.read()
        f.close()
        pattern = r"([\w.]+)\s*=\s*([0-9.eE+-]+(?:,\s*[0-9.eE+-]+)*)"
        matches_list = re.findall(pattern, content)
        matches = {}
        for match in matches_list:
            try:
                matches[match[0]] = float(match[1])
            except:
                matches[match[0]] = [float(x) for x in match[1].split(",")]

        soil_params = {}
        for k in keys["soil"]:
            match_key = "b" if k == "bb" else k
            if k == "D":
                soil_params[k] = torch.tensor(2.0, dtype=torch.float32)
            elif k == "mult":
                soil_params[k] = torch.tensor(1.0, dtype=torch.float32)
            else:
                soil_params[k] = torch.tensor(matches[f"soil_params.{match_key}"], dtype=torch.float32)

        basinCharacteristics = {}
        for k in keys["basin_characteristics"]:
            if k == "catchment_area_km2":
                basinCharacteristics[k] = torch.tensor(111.11, dtype=torch.float32)
            else:
                basinCharacteristics[k] = torch.tensor(matches[k], dtype=torch.float32)

        # --- Update parameters from JSON file in calibrated_params_dir ---
        # TODO: Maybe add an option to use the default parameters, instead of the calibrated ones?
        json_file_path = calibrated_params_dir / f"cat_{basin_id}_testrun_results.json"
        if json_file_path.exists():
            with open(json_file_path, "r") as file:
                data = json.load(file)
                best_params = data.get("best_params", {})
                
                for k in best_params.keys():
                    match_key = "refkdt" if k == "scheme" else k
                    if match_key in keys["soil"]:
                        temp_value = best_params.get(match_key, soil_params[match_key])
                        soil_params[match_key] = temp_value.clone().detach() if isinstance(temp_value, torch.Tensor) \
                            else torch.tensor(temp_value, dtype=torch.float32)
                    elif match_key in keys["basin_characteristics"]:
                        temp_value = best_params.get(match_key, basinCharacteristics[match_key])
                        basinCharacteristics[match_key] = temp_value.clone().detach() if isinstance(temp_value, torch.Tensor) \
                            else torch.tensor(temp_value, dtype=torch.float32)
                    else:
                        print(f"[warn] Parameter {k} not recognized in keys, skipping update for basin {basin_id}.")
        else:
            print(f"[warn] JSON file not found for basin {basin_id}, using default parameters.")

        # --- Unpack into DataFrame row ---
        for item in keys['soil']:
            df.at[basin_id, item] = soil_params[item]
        
        for item in keys['basin_characteristics']:
            df.at[basin_id, item] = basinCharacteristics[item]

    return df


def expand_dcfe_params_along_batch_dim(params: dict, batch_size: int) -> dict:
    """
    Input: params: dict of basin specific parameters for dcfe.
    Output: new_params: dict of basin specific paramters for dcfe, but expanded along the batch dimension, that is each value in dict is now of shape (batch_size)
    """
    new_params = {}
    for key in params.keys():
        if key != "giuh_ordinates":
            new_params[key] = params[key].expand(batch_size, *[-1 for _ in range(len(params[key].shape))])
        elif key == "giuh_ordinates":
            new_params[key] = params[key]  # need to treat this parameter differently
    return new_params


def convert_static_conceptual_params_to_batch(
    samples: List[Dict[str, np.ndarray]],
) -> Dict[str, torch.Tensor]:
    """
    Takes a list of samples of static_conceptual_parameters and converts them into a batch.
    The batch is a dictionary. Each key references a static_conceptual_parameter. The value is now a torch.Tensor
    of shape (batch_size, n_parameters), where n_parameters is the number of parameters for that static_conceptual_parameter.
    Note that giuh_ordinates needs to be handled separately, as it is a 1D array of variable length.
    """

    feature = "static_conceptual_params"
    keys = list(samples[0][feature].keys())
    batched_static_conceptual_params = {}

    for key in keys:
        if key != "giuh_ordinates":
            batched_static_conceptual_params[key] = torch.stack([sample[feature][key] for sample in samples])
        else:
            pad_target = max([len(sample[feature]["giuh_ordinates"]) for sample in samples])
            pad_lengths = [pad_target - len(sample[feature]["giuh_ordinates"]) for sample in samples]
            batched_static_conceptual_params[key] = torch.stack(
                [
                    F.pad(
                        sample[feature]["giuh_ordinates"],
                        (0, pad_length),
                        "constant",
                        0,
                    )
                    for sample, pad_length in zip(samples, pad_lengths)
                ]
            )
    return batched_static_conceptual_params


def move_data_to_device(
    data: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], device: torch.device
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    for key in data.keys():
        if key == "static_conceptual_params":
            # the value associated to 'static_conceptual_params' is a dictionary.
            # Need to move each value in the dictionary to the device individually.
            for static_conceptual_param_name in data[key].keys():
                data[key][static_conceptual_param_name] = data[key][static_conceptual_param_name].to(device)
        elif not key.startswith("date"):
            data[key] = data[key].to(device)
    return data

def identify_basins_with_low_snow(
    cfg,
    basin_file_path: str,
    yearly_max_snow_days: int = 5,
) -> list[str]:
    """
    Identify basins with low snow days based on NLDAS forcing data.
    Args:
        cfg: Configuration object containing data directory.
        basin_file_path: Path to the basin file to look through.
        yearly_max_snow_days: Maximum number of snow days per year to consider a basin as having low snow.
    Returns:
        List of basin IDs that have low snow days.
    """
    # Grab config file so we can see if the code works
    basins = utils.load_basin_file(basin_file_path)
    camels_dir = cfg.data_dir
    nldas_dir = camels_dir / "basin_mean_forcing" / "nldas"

    selected_basins = []

    for basin in basins:
        forcing_path = None
        for huc_id in range(1, 19):
            huc_folder = nldas_dir / f"{huc_id:02d}"
            candidate = huc_folder / f"{basin}_lump_nldas_forcing_leap.txt"
            if candidate.exists():
                forcing_path = candidate
                break

        if forcing_path is None:
            print(f"[warn] Forcing file not found for {basin}")
            continue

        try:
            df = pd.read_csv(forcing_path, sep=r"\s+", header=3)
            df["Tavg(C)"] = 0.5 * (df["Tmax(C)"] + df["Tmin(C)"])

            snow_days = (df["PRCP(mm/day)"] > 0) & (df["Tavg(C)"] < 0)

            snow_day_count = snow_days.sum()
            years = df["Year"].nunique()
            snow_days_per_year = snow_day_count / years

            if snow_days_per_year <= yearly_max_snow_days:
                selected_basins.append(basin)

        except FileNotFoundError:
                print(f"[warn] Forcing file missing: {forcing_path}")

    return selected_basins


def filter_basins_all_param_files(
    cfg,
    basins: List[str],
) -> Dict[List[str], List[str]]:
    """
    Check whether each basin has both the CFE config and calibrated JSON files.

    Args:
        basins: List of basin IDs as strings.
        cfe_param_dir: Path to directory containing *_bmi_config_cfe_pass.txt files.
        calibrated_params_dir: Path to directory containing cat_*_testrun_results.json files.

    Returns:
        A tuple:
            - valid_basins: list of basin IDs where both files exist
            - missing_basins: list of basin IDs where one or both files are missing
    """
    cfe_param_dir = cfg.param_dir
    calibrated_param_dir = cfg.calibrated_params_path
    
    valid_basins = []
    missing_basins = []

    for basin in basins:
        cfe_file = cfe_param_dir / f"{basin}_bmi_config_cfe_pass.txt"
        json_file = calibrated_param_dir / f"cat_{basin}_testrun_results.json"

        if cfe_file.exists() and json_file.exists():
            valid_basins.append(basin)
        else:
            missing_basins.append(basin)
            if not cfe_file.exists():
                print(f"[missing] CFE file not found for {basin}")
            if not json_file.exists():
                print(f"[missing] JSON file not found for {basin}")

    return {"valid_basins": valid_basins, "missing_basins": missing_basins}

keys = {
    "basin_characteristics": [
        "catchment_area_km2",
        "refkdt",
        "max_gw_storage",
        "expon",
        "Cgw",
        "alpha_fc",
        "K_nash",
        "K_lf",
        "nash_storage",
        "giuh_ordinates",
        ],
    "soil": [
        "depth",
        "bb",
        "satdk",
        "satpsi",
        "slop",
        "smcmax",
        "wltsmc",
        "D",
        "mult",
        ],
    "study_calibrated_params": [
        "bb",
        "smcmax",
        "satdk",
        "slop",
        "max_gw_storage",
        "expon",
        "Cgw",
        "K_lf",
        "K_nash",
        "refkdt",
        ],
    }

# used in CFE modules
physics_constants = {
        "atm_press_Pa": 101325.0,  # [Pa]
        "unit_weight_water_N_per_m3": 9810.0,  # [N/m3]
    }