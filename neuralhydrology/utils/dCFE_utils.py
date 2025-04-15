import json
import re

import torch


def get_dcfe_params(cfg, device):
    cfe_param_dir = cfg.param_dir
    calibrated_params_dir = cfg.calibrated_params_path
    basin_id = str(cfg.basin_id)
    if basin_id[0] != "0":
        basin_id = "0" + basin_id
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
        "depth": torch.tensor(
            matches["soil_params.depth"], device=device, dtype=torch.float32
        ),
        "bb": torch.tensor(
            matches["soil_params.b"], device=device, dtype=torch.float32
        ),
        "satdk": torch.tensor(
            matches["soil_params.satdk"], device=device, dtype=torch.float32
        ),
        "satpsi": torch.tensor(
            matches["soil_params.satpsi"], device=device, dtype=torch.float32
        ),
        "slop": torch.tensor(
            matches["soil_params.slop"], device=device, dtype=torch.float32
        ),
        "smcmax": torch.tensor(
            matches["soil_params.smcmax"], device=device, dtype=torch.float32
        ),
        "wltsmc": torch.tensor(
            matches["soil_params.wltsmc"], device=device, dtype=torch.float32
        ),
        "D": torch.tensor(2.0, device=device, dtype=torch.float32),
        "mult": torch.tensor(1.0, device=device, dtype=torch.float32),
    }
    basinCharacteristics = {
        "catchment_area_km2": torch.tensor(526.77, device=device, dtype=torch.float32),
        "refkdt": torch.tensor(matches["refkdt"], device=device, dtype=torch.float32),
        "max_gw_storage": torch.tensor(
            matches["max_gw_storage"], device=device, dtype=torch.float32
        ),
        "expon": torch.tensor(matches["expon"], device=device, dtype=torch.float32),
        "Cgw": torch.tensor(matches["Cgw"], device=device, dtype=torch.float32),
        "alpha_fc": torch.tensor(
            matches["alpha_fc"], device=device, dtype=torch.float32
        ),
        "K_nash": torch.tensor(matches["K_nash"], device=device, dtype=torch.float32),
        "K_lf": torch.tensor(matches["K_lf"], device=device, dtype=torch.float32),
        "nash_storage": torch.tensor(
            matches["nash_storage"], device=device, dtype=torch.float32
        ),
        "giuh_ordinates": torch.tensor(
            matches["giuh_ordinates"], device=device, dtype=torch.float32
        ),
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
                device=device,
                dtype=torch.float32,
            )
            soil_params["smcmax"] = torch.tensor(
                best_params.get("smcmax", soil_params["smcmax"].item()),
                device=device,
                dtype=torch.float32,
            )
            soil_params["satdk"] = torch.tensor(
                best_params.get("satdk", soil_params["satdk"].item()),
                device=device,
                dtype=torch.float32,
            )
            soil_params["slop"] = torch.tensor(
                best_params.get("slop", soil_params["slop"].item()),
                device=device,
                dtype=torch.float32,
            )
            basinCharacteristics["max_gw_storage"] = torch.tensor(
                best_params.get(
                    "max_gw_storage", basinCharacteristics["max_gw_storage"].item()
                ),
                device=device,
                dtype=torch.float32,
            )
            basinCharacteristics["expon"] = torch.tensor(
                best_params.get("expon", basinCharacteristics["expon"].item()),
                device=device,
                dtype=torch.float32,
            )
            basinCharacteristics["Cgw"] = torch.tensor(
                best_params.get("Cgw", basinCharacteristics["Cgw"].item()),
                device=device,
                dtype=torch.float32,
            )
            basinCharacteristics["K_lf"] = torch.tensor(
                best_params.get("K_lf", basinCharacteristics["K_lf"].item()),
                device=device,
                dtype=torch.float32,
            )
            basinCharacteristics["K_nash"] = torch.tensor(
                best_params.get("K_nash", basinCharacteristics["K_nash"].item()),
                device=device,
                dtype=torch.float32,
            )
            basinCharacteristics["refkdt"] = torch.tensor(
                best_params.get("scheme", basinCharacteristics["refkdt"].item()),
                device=device,
                dtype=torch.float32,
            )

    return soil_params, basinCharacteristics


def expand_dcfe_params_along_batch_dim(params: dict, batch_size: int) -> dict:
    """
    Input: params: dict of basin specific parameters for dcfe.
    Output: new_params: dict of basin specific paramters for dcfe, but expanded along the batch dimension, that is each value in dict is now of shape (batch_size)
    """
    new_params = {}
    for key in params.keys():
        if key != "giuh_ordinates":
            new_params[key] = params[key].expand(
                batch_size, *[-1 for _ in range(len(params[key].shape))]
            )
        elif key == "giuh_ordinates":
            new_params[key] = params[key]  # need to treat this parameter differently
    return new_params
