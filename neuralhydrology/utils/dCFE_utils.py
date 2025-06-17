import json
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from neuralhydrology.datautils import utils
from neuralhydrology.utils.constants import BASIN_CHARACTERISTIC_KEYS, SOIL_KEYS


def get_dcfe_params_test(cfg):
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
    # get all the basin ids as strings, now only from train_basin_file but in other code it was {period}_basin_file
    # where period is train, val, test etc
    basins = utils.load_basin_file(getattr(cfg, "train_basin_file"))
    # empty dataframe to store the parameters, basin ids as index
    # df = pd.DataFrame(index=basins, columns=["soil_params", "basinCharacteristics"])

    col_keys = SOIL_KEYS + BASIN_CHARACTERISTIC_KEYS
    df = pd.DataFrame(index=basins, columns=col_keys)

    ####

    # not sure why below is here, comment out for now
    # if basin_id[0] != "0":
    #    basin_id = "0" + basin_id

    # iterate thru each basin id
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

        # TODO: Write the code below as a single function call.
        soil_params = {
            "depth": torch.tensor(matches["soil_params.depth"], dtype=torch.float32),
            "bb": torch.tensor(matches["soil_params.b"], dtype=torch.float32),
            "satdk": torch.tensor(matches["soil_params.satdk"], dtype=torch.float32),
            "satpsi": torch.tensor(matches["soil_params.satpsi"], dtype=torch.float32),
            "slop": torch.tensor(matches["soil_params.slop"], dtype=torch.float32),
            "smcmax": torch.tensor(matches["soil_params.smcmax"], dtype=torch.float32),
            "wltsmc": torch.tensor(matches["soil_params.wltsmc"], dtype=torch.float32),
            "D": torch.tensor(2.0, dtype=torch.float32),
            "mult": torch.tensor(1.0, dtype=torch.float32),
        }
        # TODO: Write the code below as a single function call.
        basinCharacteristics = {
            "catchment_area_km2": torch.tensor(111.11, dtype=torch.float32),
            "refkdt": torch.tensor(matches["refkdt"], dtype=torch.float32),
            "max_gw_storage": torch.tensor(matches["max_gw_storage"], dtype=torch.float32),
            "expon": torch.tensor(matches["expon"], dtype=torch.float32),
            "Cgw": torch.tensor(matches["Cgw"], dtype=torch.float32),
            "alpha_fc": torch.tensor(matches["alpha_fc"], dtype=torch.float32),
            "K_nash": torch.tensor(matches["K_nash"], dtype=torch.float32),
            "K_lf": torch.tensor(matches["K_lf"], dtype=torch.float32),
            "nash_storage": torch.tensor(matches["nash_storage"], dtype=torch.float32),
            "giuh_ordinates": torch.tensor(matches["giuh_ordinates"], dtype=torch.float32),
        }

        # Update parameters from JSON file in calibrated_params_dir

        # TODO: Ensure that all basins have corresponding json files. Add a try/except block to handle missing files.
        json_file_path = calibrated_params_dir / f"cat_{basin_id}_testrun_results.json"
        if json_file_path.exists():
            with open(json_file_path, "r") as file:
                data = json.load(file)
                best_params = data.get("best_params", {})

                # Update the parameters in soil_params and basinCharacteristics
                soil_params["bb"] = torch.tensor(
                    best_params.get("bb", soil_params["bb"].item()),
                    dtype=torch.float32,
                )
                soil_params["smcmax"] = torch.tensor(
                    best_params.get("smcmax", soil_params["smcmax"].item()),
                    dtype=torch.float32,
                )
                soil_params["satdk"] = torch.tensor(
                    best_params.get("satdk", soil_params["satdk"].item()),
                    dtype=torch.float32,
                )
                soil_params["slop"] = torch.tensor(
                    best_params.get("slop", soil_params["slop"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["max_gw_storage"] = torch.tensor(
                    best_params.get("max_gw_storage", basinCharacteristics["max_gw_storage"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["expon"] = torch.tensor(
                    best_params.get("expon", basinCharacteristics["expon"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["Cgw"] = torch.tensor(
                    best_params.get("Cgw", basinCharacteristics["Cgw"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["K_lf"] = torch.tensor(
                    best_params.get("K_lf", basinCharacteristics["K_lf"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["K_nash"] = torch.tensor(
                    best_params.get("K_nash", basinCharacteristics["K_nash"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["refkdt"] = torch.tensor(
                    best_params.get("scheme", basinCharacteristics["refkdt"].item()),
                    dtype=torch.float32,
                )

        # --- 4) Unpack into DataFrame row ---
        # soil ["depth","bb","satdk","satpsi","slop","smcmax","wltsmc","D","mult"]
        # bc_keys   = ["catchment_area_km2","refkdt","max_gw_storage","expon",
        #         "Cgw","alpha_fc","K_nash","K_lf","nash_storage","giuh_ordinates"]
        # TODO: Write this as a single function call.
        df.at[basin_id, "depth"] = soil_params["depth"]
        df.at[basin_id, "bb"] = soil_params["bb"]
        df.at[basin_id, "satdk"] = soil_params["satdk"]
        df.at[basin_id, "satpsi"] = soil_params["satpsi"]
        df.at[basin_id, "slop"] = soil_params["slop"]
        df.at[basin_id, "smcmax"] = soil_params["smcmax"]
        df.at[basin_id, "wltsmc"] = soil_params["wltsmc"]
        df.at[basin_id, "D"] = soil_params["D"]
        df.at[basin_id, "mult"] = soil_params["mult"]
        df.at[basin_id, "catchment_area_km2"] = basinCharacteristics["catchment_area_km2"]
        df.at[basin_id, "refkdt"] = basinCharacteristics["refkdt"]
        df.at[basin_id, "max_gw_storage"] = basinCharacteristics["max_gw_storage"]
        df.at[basin_id, "expon"] = basinCharacteristics["expon"]
        df.at[basin_id, "Cgw"] = basinCharacteristics["Cgw"]
        df.at[basin_id, "alpha_fc"] = basinCharacteristics["alpha_fc"]
        df.at[basin_id, "K_nash"] = basinCharacteristics["K_nash"]
        df.at[basin_id, "K_lf"] = basinCharacteristics["K_lf"]
        df.at[basin_id, "nash_storage"] = basinCharacteristics["nash_storage"]
        df.at[basin_id, "giuh_ordinates"] = basinCharacteristics["giuh_ordinates"]

    return df


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
    # get all the basin ids as strings, now only from train_basin_file but in other code it was {period}_basin_file
    # where period is train, val, test etc
    basins = utils.load_basin_file(getattr(cfg, "train_basin_file"))
    # empty dataframe to store the parameters, basin ids as index
    df = pd.DataFrame(index=basins, columns=["soil_params", "basinCharacteristics"])

    # not sure why below is here, comment out for now
    # if basin_id[0] != "0":
    #    basin_id = "0" + basin_id

    # iterate thru each basin id
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
        soil_params = {
            "depth": torch.tensor(matches["soil_params.depth"], dtype=torch.float32),
            "bb": torch.tensor(matches["soil_params.b"], dtype=torch.float32),
            "satdk": torch.tensor(matches["soil_params.satdk"], dtype=torch.float32),
            "satpsi": torch.tensor(matches["soil_params.satpsi"], dtype=torch.float32),
            "slop": torch.tensor(matches["soil_params.slop"], dtype=torch.float32),
            "smcmax": torch.tensor(matches["soil_params.smcmax"], dtype=torch.float32),
            "wltsmc": torch.tensor(matches["soil_params.wltsmc"], dtype=torch.float32),
            "D": torch.tensor(2.0, dtype=torch.float32),
            "mult": torch.tensor(1.0, dtype=torch.float32),
        }
        basinCharacteristics = {
            "catchment_area_km2": torch.tensor(111.11, dtype=torch.float32),
            "refkdt": torch.tensor(matches["refkdt"], dtype=torch.float32),
            "max_gw_storage": torch.tensor(matches["max_gw_storage"], dtype=torch.float32),
            "expon": torch.tensor(matches["expon"], dtype=torch.float32),
            "Cgw": torch.tensor(matches["Cgw"], dtype=torch.float32),
            "alpha_fc": torch.tensor(matches["alpha_fc"], dtype=torch.float32),
            "K_nash": torch.tensor(matches["K_nash"], dtype=torch.float32),
            "K_lf": torch.tensor(matches["K_lf"], dtype=torch.float32),
            "nash_storage": torch.tensor(matches["nash_storage"], dtype=torch.float32),
            "giuh_ordinates": torch.tensor(matches["giuh_ordinates"], dtype=torch.float32),
        }

        # Update parameters from JSON file in calibrated_params_dir

        # TODO: Ensure that all basins have corresponding json files. Add a try/except block to handle missing files.
        json_file_path = calibrated_params_dir / f"cat_{basin_id}_testrun_results.json"
        if json_file_path.exists():
            with open(json_file_path, "r") as file:
                data = json.load(file)
                best_params = data.get("best_params", {})

                # Update the parameters in soil_params and basinCharacteristics
                soil_params["bb"] = torch.tensor(
                    best_params.get("bb", soil_params["bb"].item()),
                    dtype=torch.float32,
                )
                soil_params["smcmax"] = torch.tensor(
                    best_params.get("smcmax", soil_params["smcmax"].item()),
                    dtype=torch.float32,
                )
                soil_params["satdk"] = torch.tensor(
                    best_params.get("satdk", soil_params["satdk"].item()),
                    dtype=torch.float32,
                )
                soil_params["slop"] = torch.tensor(
                    best_params.get("slop", soil_params["slop"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["max_gw_storage"] = torch.tensor(
                    best_params.get("max_gw_storage", basinCharacteristics["max_gw_storage"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["expon"] = torch.tensor(
                    best_params.get("expon", basinCharacteristics["expon"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["Cgw"] = torch.tensor(
                    best_params.get("Cgw", basinCharacteristics["Cgw"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["K_lf"] = torch.tensor(
                    best_params.get("K_lf", basinCharacteristics["K_lf"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["K_nash"] = torch.tensor(
                    best_params.get("K_nash", basinCharacteristics["K_nash"].item()),
                    dtype=torch.float32,
                )
                basinCharacteristics["refkdt"] = torch.tensor(
                    best_params.get("scheme", basinCharacteristics["refkdt"].item()),
                    dtype=torch.float32,
                )
        df.at[basin_id, "soil_params"] = soil_params
        df.at[basin_id, "basinCharacteristics"] = basinCharacteristics

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
