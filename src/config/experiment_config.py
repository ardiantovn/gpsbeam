import itertools
import datetime

from dataclasses import dataclass, field
from loguru import logger
from typing import List, Tuple, Dict

@dataclass
class ExperimentConfig:
    exp_folder_name: str = '00_drone_rnn_var_input'

    exp_dict: Dict[str, List[Tuple]] = field(default_factory=lambda: {
        # data variation
        # "scenario_num": [8],
        "seednum": [0],
        # "sampling_method": ["sequential"],

        # model variation
        "num_classes": [64],
        "seq_len": [8],
        "out_len": [3],
        # "dropout": [0.5],

        "shuffle_train_dset": [True],
        "model_input_column_list": [
                                    ["unit2_loc_minmax_norm"],
                                    ["unit2to1_vector"],
                                    ],
    })

    # metrics
    top_n_list: List[int] = field(default_factory=lambda: [1, 3, 5])
    confidence_threshold_list: List[float] = field(default_factory=lambda: [0.8, 0.85, 0.9, 0.95, 0.99])
    power_loss_db_threshold_list: List[float] = field(default_factory=lambda: [1, 3, 6])

    def __post_init__(self):
        self.day_time = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
        self.exp_folder_name = f'{self.exp_folder_name}_{self.day_time}'
        
        # Generate all possible combinations
        combinations = list(itertools.product(*self.exp_dict.values()))
        self.combinations = [dict(zip(self.exp_dict.keys(), combo)) for combo in combinations]

    def show_config(self):
        config_str = "\n".join([f"\t {key}={value}" for key, value in self.__dict__.items()])
        logger.info(f"\n\tExperimentConfig initialized with: \n{config_str}")