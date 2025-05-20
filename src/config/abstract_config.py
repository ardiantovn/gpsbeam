from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal
from loguru import logger

@dataclass
class AbstractConfig:
    model_arch_name: str = 'abstract-model'
    
    def __post_init__(self):
        # Group features by their length/dimension
        self.feature_groups = {
            1: ["unit2_speed", "unit2_altitude", "unit2_distance", "unit2_height", 
                "unit2_x-speed", "unit2_y-speed", "unit2_z-speed", "unit2_pitch", 
                "unit2_roll", 'sample_idx', "unit2_height_log", "unit2_height_minmax_norm"],
            2: ["unit2_loc_minmax_norm", 'unit2_loc',],
            3: ["unit2to1_vector", "unit1_loc_vector", "unit2_loc_vector"],
            8: ["unit1_beam_idx_8"],
            16: ["unit1_beam_idx_16"],
            32: ["unit1_beam_idx_32"],
            64: ["unit1_beam_idx_64"],
            216: ["unit1_lidar_scr", "unit1_lidar"]
        }

        # Set label column name based on number of classes
        self.label_column_name = f'unit1_beam_idx_{self.num_classes}'
        if self.dl_task_type == 'base_beam_tracking' or self.dl_task_type == 'base_beam_prediction':
            self.pred_num = self.out_len
        else:
            self.pred_num = self.out_len + 1

        # Calculate total feature length based on input columns
        self.feature_len = 0
        for model_input_column in self.model_input_column_list:
            feature_len = next((length for length, features in self.feature_groups.items() 
                              if model_input_column in features), None)
            if feature_len is None:
                raise ValueError(f"Invalid model_input_column: {model_input_column}")
            self.feature_len += feature_len

    def show_config(self):
        config_str = "\n".join([f"\t {key}={value}" for key, value in self.__dict__.items()])
        logger.info(f"\n\tModelConfig initialized with: \n{config_str}")
