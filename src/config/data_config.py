from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal

from loguru import logger

@dataclass
class DataConfig:
    scenario_num: int = 1
    seednum: int = 42
    splitting_method: Literal["sequential", "adjusted"] = "adjusted"
    shuffle_sequential: bool = False
    num_classes: int = 64

    train_val_test_split_frac: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    portion_percentage: int = 100 # 1-100

    gps_calibrated_scenario_nums: List[int] = field(default_factory=lambda: [8, 9])
    lidar_scr_scenario_nums: List[int] = field(default_factory=lambda: [8, 9])
    lidar_scenario_nums: List[int] = field(default_factory=lambda: [31, 32, 33])
    radar_scenario_nums: List[int] = field(default_factory=lambda: [9, 31, 32, 33])

    def __post_init__(self):
        self.label_column_name = f"unit1_beam_idx_{self.num_classes}"
        if len(self.train_val_test_split_frac) == 2:
            self.train_frac = self.train_val_test_split_frac[0] 
            self.val_frac = 0
            self.test_frac = self.train_val_test_split_frac[1]
        else:
            self.train_frac = self.train_val_test_split_frac[0]
            self.val_frac = self.train_val_test_split_frac[1]
            self.test_frac = self.train_val_test_split_frac[2]
    
    def show_config(self):
        config_str = "\n".join([f"\t {key}={value}" for key, value in self.__dict__.items()])
        logger.info(f"\n\tDataConfig initialized with: \n{config_str}")
    