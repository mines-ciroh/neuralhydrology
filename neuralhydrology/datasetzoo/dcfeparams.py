from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.dCFE_utils import get_dcfe_params


class dcfeParameters(BaseDataset):
    """Data set class for the CAMELS US data set by [#]_ and [#]_.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(dcfeParameters, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def load_dcfe_attributes(param_dir: Path, basins: List[str] = []) -> pd.DataFrame:
        """Load dCFE conceptual parameters for each basin.

        Parameters
        ----------
        param_dir : Path
            Path to the directory containing dCFE conceptual parameters.
        basins : List[str], optional
            If passed, return only attributes for the basins specified in this list. Otherwise, return attributes for all basins.

        Returns
        -------
        pandas.DataFrame
            Basin-indexed DataFrame containing the conceptual parameters.
        """
        params_path = param_dir / 'dcfe_conceptual_params'

        if not attributes_path.exists():
        raise RuntimeError(f"Conceptual parameters folder not found at {attributes_path}")

        # Load all parameter files in the directory
        param_files = attributes_path.glob('*.csv')

        dfs = []
        for param_file in param_files:
            df_temp = pd.read_csv(param_file, index_col=0)  # Assuming basin ID is the index
            dfs.append(df_temp)

        # Combine all parameter files into a single DataFrame
        df = pd.concat(dfs, axis=0)

        if basins:
            if any(b not in df.index for b in basins):
                raise ValueError('Some basins are missing conceptual parameters.')
            df = df.loc[basins]

        return df
