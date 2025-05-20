import math
import os
import random
from typing import List, Tuple, Dict, Any
from numpy.typing import NDArray

import hickle as hkl
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

from src.preprocessor.gpsprep import GpsPrep

class DataPrep:

    def __init__(self,
                 data_config: Any,
                 ) -> None:
        """
        Initializes the DataPrep class.

        Class that contains method to prepare ai ready dataset.
        Returns:
            None
        """
        self.data_config = data_config
        self._set_seed(self.data_config.seednum)
        self._setup_dir()

        # special keys (the value will be converted from np.array into list of tuple)
        self.special_keys = ['unit1_loc', 'unit2_loc', 'unit2_loc_minmax_norm',
                             'unit1_loc_vector', 'unit2_loc_vector',
                             'unit1_pwr_values_8',
                             'unit1_pwr_values_16', 'unit1_pwr_values_32',
                             'unit1_pwr_values_64', "unit2to1_vector",
                             ]



    def _setup_dir(self):
        # Get the current working directory
        self.main_folder = os.getcwd().replace('\\', '/').split("/")[:-2]
        self.main_folder = '/'.join(self.main_folder)

        # Define the scenario and set up directory paths
        self.raw_data_folder = os.path.join(self.main_folder, 'data/raw')
        self.processed_data_folder = os.path.join(self.main_folder, 'data/processed')

        if not os.path.exists(self.processed_data_folder):
            os.makedirs(self.processed_data_folder)
        if self.data_config.splitting_method == 'sequential':
            split_str = f'splitting_method_{self.data_config.splitting_method}_shuffle_{self.data_config.shuffle_sequential}_'
        else:
            split_str = f'splitting_method_{self.data_config.splitting_method}'

        self.name_str = (
                        f'seednum{self.data_config.seednum}_'
                        f'train{self.data_config.train_frac}_'
                        f'test{self.data_config.test_frac}_'
                        f'portion{self.data_config.portion_percentage}_'
                        f'beam{self.data_config.num_classes}_'
                        f'{split_str}'
                        )

    def _check_path_availability(self, path: str) -> None:
        """
        A method to check if a file/folder is available in the specified path.

        Args:
            path (str): The path to the file/path to be checked.

        Returns:
            None
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File/Folder {path} does not exist.")
        else:
            logger.debug(f"File/Folder {path} is available.")
            return True

    def _set_seed(self, seednum: int = 42) -> None:
        """Sets the seed for reproducibility."""
        random.seed(seednum)
        np.random.seed(seednum)
        torch.manual_seed(seednum)
        torch.cuda.manual_seed(seednum)
        torch.cuda.manual_seed_all(seednum)  # if using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __get_unit1_pwr(self, n_beams: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get maximum power in each array -> get its index|
        # The idx will be inside [0, 1, 2, ... , 63] (64 beams)
        self.unit1_beam_idx_arr = np.argmax(self.unit1_pwr_values_arr, axis=1)

        # Assign beam values (and automatically downsample if n_beams != 64)
        n_beams_list = [8, 16, 32, 64]
        max_beams = self.unit1_pwr_values_arr[0].shape[-1]
        if n_beams not in n_beams_list:
            raise ValueError(f'n_beams should be in this list {n_beams_list}')

        divider = max_beams // n_beams
        # EX: divider=2 and max_beam=64 -> create ([0,2,4,..., 60]) (array's len=32)
        beam_idxs = np.arange(0, max_beams, divider)

        # resampling all unit1_pwr_values . Ex: array with len=64 become array with len=32
        # and the idx will be in range [0, 1, 2, ... , 31] (32 beams)
        downsampled_unit1_pwr_values_arr = self.unit1_pwr_values_arr[:,beam_idxs]

        # Get maximum power in each downsampled array -> get its index
        downsampled_unit1_beam_idx_arr = np.argmax(downsampled_unit1_pwr_values_arr, axis=1)
        return downsampled_unit1_pwr_values_arr, downsampled_unit1_beam_idx_arr

    def _process_unit1_pwr(self, n_beams_list: List[int] = [8, 16, 32]):
        # EXTRACT VALUES FROM RAW DATASET
        # get unit1_pwr_values, unit1_pwr_idx,
        # downsampled_unit1_pwr_values, downsampled_unit1_pwr_idx
        self._update_unit1_pwr_path()
        for i_beam in n_beams_list:
            downsampled_unit1_pwr_values_arr, downsampled_unit1_beam_idx_arr = self.__get_unit1_pwr(n_beams=i_beam)
            # self.data_dict[f"unit1_pwr_values_{i_beam}"] = downsampled_unit1_pwr_values_arr
            self.data_dict[f"unit1_beam_idx_{i_beam}"] = downsampled_unit1_beam_idx_arr
        # self.data_dict["unit1_pwr_values_64"] = [tuple(i) for i in self.unit1_pwr_values_arr]
        self.data_dict["unit1_beam_idx_64"] = self.unit1_beam_idx_arr

    def _update_lidar_path(self):
        lidar_path = []
        key = None
        self.lidar_values_arr = ["None"]*self.data_len
        if self.data_config.scenario_num in self.data_config.lidar_scenario_nums:
            key = 'unit1_lidar'
        elif self.data_config.scenario_num in self.data_config.lidar_scr_scenario_nums:
            key = 'unit1_lidar_scr'
        if key is not None:
            lidar_path = self.raw_df[key].values
            self.lidar_values_arr = np.array([f'{self.raw_scenario_folder}/{i[1:]}'
                                                for i in lidar_path])
            self.data_dict[key] = self.lidar_values_arr
        else:
            # create lidar column with zeros numpy array
            self.data_dict['unit1_lidar'] = self.lidar_values_arr
            self.data_dict['unit1_lidar_scr'] = self.lidar_values_arr


    def _update_radar_path(self):
        radar_path = []
        self.radar_values_arr = ['None']*self.data_len
        if self.data_config.scenario_num in self.data_config.radar_scenario_nums:
            radar_path = self.raw_df['unit1_radar'].values
        if len(radar_path) > 0:
            self.radar_values_arr = np.array([f'{self.raw_scenario_folder}/{i[1:]}'
                                            for i in radar_path])
        self.data_dict['unit1_radar'] = self.radar_values_arr

    def _update_camera_path(self):
        self.camera_values_arr = ['None']*self.data_len
        camera_path = self.raw_df['unit1_rgb'].values
        self.camera_values_arr = np.array([f'{self.raw_scenario_folder}/{i[1:]}'
                                            for i in camera_path])
        self.data_dict['unit1_rgb'] = self.camera_values_arr

    def _update_unit1_pwr_path(self):
        self.unit1_pwr_path_arr = ['None']*self.data_len
        unit1_pwr_path = self.raw_df['unit1_pwr_60ghz'].values
        self.unit1_pwr_path_arr = np.array([f'{self.raw_scenario_folder}/{i[1:]}'
                                            for i in unit1_pwr_path])
        self.unit1_pwr_values_arr = np.array([np.loadtxt(i)
                                              for i in self.unit1_pwr_path_arr])
        self.data_dict['unit1_pwr_60ghz'] = self.unit1_pwr_path_arr

    def _update_unit2_other_val(self, col_name: str):
        unit2_val_path = self.raw_df[col_name].values
        self.unit2_val_path_arr = np.array([f'{self.raw_scenario_folder}/{i[1:]}'
                                            for i in unit2_val_path])
        self.unit2_val_arr = np.array([np.loadtxt(i)
                                              for i in self.unit2_val_path_arr])
        self.data_dict[col_name] = self.unit2_val_arr

    def _run_all_gps_preprocessing(self) -> None:
        self.gpsprep = GpsPrep()

        # get unit1_loc
        unit1_loc_path_arr = self.raw_df['unit1_loc'].values
        unit1_loc_values_arr = [self.gpsprep.read(self.raw_scenario_folder, i) for i in unit1_loc_path_arr]

        # get unit2_loc
        if self.data_config.scenario_num in self.data_config.gps_calibrated_scenario_nums:
            self.unit2_loc_col_name = 'unit2_loc_cal'
        else:
            self.unit2_loc_col_name = 'unit2_loc'
        unit2_loc_path_arr = self.raw_df[self.unit2_loc_col_name].values
        unit2_loc_values_arr = [self.gpsprep.read(self.raw_scenario_folder, i) for i in unit2_loc_path_arr]
        if "unit2_hdop" and "unit2_pdop" in self.raw_df.columns:
            self.data_dict["unit2_hdop"] = self.raw_df["unit2_hdop"].values
            self.data_dict["unit2_pdop"] = self.raw_df["unit2_pdop"].values
        else:
            # create zeros numpy array
            self.data_dict["unit2_hdop"] = np.zeros(len(self.data_dict["sample_idx"]))
            self.data_dict["unit2_pdop"] = np.zeros(len(self.data_dict["sample_idx"]))

        # minmax_norm
        self.data_dict["unit1_loc"] = unit1_loc_values_arr
        self.data_dict["unit2_loc"] = unit2_loc_values_arr

        self.data_dict['unit2_loc_minmax_norm'] = self.gpsprep.minmax_norm(latlon_arr=unit2_loc_values_arr)

        # get other values, this is for drone scenario(scenario 23)
        other_val_list = ['unit2_speed', 'unit2_altitude', 'unit2_distance', 'unit2_height',
                          'unit2_x-speed', 'unit2_y-speed', 'unit2_z-speed',
                          'unit2_pitch', 'unit2_roll']
        for col_name in other_val_list:
            if col_name in self.raw_df.columns:
                self._update_unit2_other_val(col_name=col_name)

        # height natural log
        self.data_dict['unit2_height_log'] = np.log(self.data_dict['unit2_height'] + 1)

        # height minmax_norm
        self.data_dict['unit2_height_minmax_norm'] = self.gpsprep.minmax_norm(latlon_arr=self.data_dict['unit2_height'],
                                                                              min_val=0,
                                                                              max_val=np.max(self.data_dict['unit2_height']))

        unit1_height_arr = np.zeros(len(unit1_loc_values_arr))
        if "unit2_height" in self.data_dict.keys():
            unit2_height_arr = self.data_dict["unit2_height"]
        else:
            unit2_height_arr = np.zeros(len(unit2_loc_values_arr))
        self.data_dict['unit2to1_vector'] = self.gpsprep.unit2_to_unit1_vector(
                                                    unit1_latlon_arr=unit1_loc_values_arr,
                                                    unit2_latlon_arr=unit2_loc_values_arr,
                                                    unit1_height_arr=unit1_height_arr,
                                                    unit2_height_arr=unit2_height_arr)
        self.data_dict["unit1_loc_vector"] = self.gpsprep.unitx_vector(
                                                    unitx_latlon_arr=unit1_loc_values_arr,
                                                    unitx_height_arr=unit1_height_arr)
        self.data_dict["unit2_loc_vector"] = self.gpsprep.unitx_vector(
                                                    unitx_latlon_arr=unit2_loc_values_arr,
                                                    unitx_height_arr=unit2_height_arr)

    def _preprocess(self) -> None:
        self.raw_scenario_folder = f'{self.raw_data_folder}/Scenario{self.data_config.scenario_num}'
        raw_data_file = f'{self.raw_scenario_folder}/scenario{self.data_config.scenario_num}.csv'

        # Check if the path exists or not
        self._check_path_availability(raw_data_file)

        # Read the initial dataset
        self.raw_df = pd.read_csv(raw_data_file)
        self.raw_df.columns = map(str.lower, self.raw_df.columns)

        self.data_dict = {}
        self.data_dict["sample_idx"] = self.raw_df["index"].values
        self.data_len = self.data_dict["sample_idx"].shape[0]
        self.data_dict['from_scenario'] = np.array([self.data_config.scenario_num] * self.data_len)
        self.data_dict['seednum'] = np.array([self.data_config.seednum] * self.data_len)
        self.data_dict["seq_index"] = self.raw_df["seq_index"].values

        self._process_unit1_pwr()
        self._update_lidar_path()
        self._update_radar_path()
        self._update_camera_path()
        self._run_all_gps_preprocessing()

    def _filter_dict_by_sample_idx(self, dummy_dict: dict) -> dict:
        # Find the indices of matching sample_idx
        matching_mask = np.isin(self.data_dict['sample_idx'], dummy_dict['sample_idx'])

        # Create the filtered dictionary
        filtered_dict = {}
        for k, v in self.data_dict.items():
            if isinstance(v, np.ndarray):
                filtered_dict[k] = v[matching_mask]
            elif isinstance(v, list):
                filtered_dict[k] = np.array([item for item, mask in zip(v, matching_mask) if mask])
        self.dummy_dict = dummy_dict
        return filtered_dict

    def get_train_val_test_dataset(self) -> Tuple[pd.DataFrame]:
        self.processed_folder = os.path.join(self.main_folder,
                                        f'data/processed/Scenario{self.data_config.scenario_num}')

        # get from processed folder if dataset exists
        self.hickle_fname = (
                            f'{self.processed_folder}/'
                            f'dset_'
                            f'scenario{self.data_config.scenario_num}_'
                            f'{self.name_str}'
                            '.hkl'
                            )

        if os.path.exists(self.hickle_fname):
            try:
                self.dataset_dict = hkl.load(self.hickle_fname)
                logger.info(f"\nDataset is LOADED from {self.hickle_fname}")
            except Exception as e:
                logger.error(f"""Error loading dataset from {self.hickle_fname}.
                                Try to delete this file and run again: {e}""")
                raise
        else:
            if not os.path.exists(self.processed_folder):
                os.makedirs(self.processed_folder)
            self._generate_train_val_test_dataset()

        train_num = len(self.dataset_dict['train']['sample_idx'])
        val_num = len(self.dataset_dict['val']['sample_idx']) if 'val' in self.dataset_dict else 0
        test_num = len(self.dataset_dict['test']['sample_idx'])
        total_num = train_num + val_num + test_num

        logger.info(f"""
                    RAW DATASET INFO
                    ------------------------------
                    Scenario Num                                    : {self.data_config.scenario_num},
                    Splitting Method                                : {self.data_config.splitting_method}
                    Portion Percentage                              : {self.data_config.portion_percentage}
                    Training                                        : {train_num} samples [{train_num/total_num*100:.2f}%]
                    Validation                                      : {val_num} samples [{val_num/total_num*100:.2f}%]
                    Testing                                         : {test_num} samples [{test_num/total_num*100:.2f}%]
                    Total                                           : {total_num} samples\n""")
        logger.info("---"*50)

    def _generate_train_val_test_dataset(self) -> Tuple[pd.DataFrame]:
        # Validate split fractions
        split_fractions = [self.data_config.train_frac]
        if hasattr(self.data_config, 'val_frac') and self.data_config.val_frac > 0:
            split_fractions.append(self.data_config.val_frac)
        split_fractions.append(self.data_config.test_frac)

        assert abs(sum(split_fractions) - 1.0) < 1e-10, "Split sizes must sum to 1"

        self._preprocess()

        # Log dataset generation information
        logger.info("Generating train, validation and test datasets...")

        # Create dummy DataFrame for splitting
        dummy_df = pd.DataFrame({
            'sample_idx': self.data_dict['sample_idx'],
            'seq_index': self.data_dict['seq_index'],
            self.data_config.label_column_name: self.data_dict[self.data_config.label_column_name]
        })

        if hasattr(self.data_config, 'val_frac') and self.data_config.val_frac > 0:
            train_df, val_df, test_df = self._split_general_case(dummy_df)
        else:
            train_df, test_df = self._split_general_case(dummy_df)
            val_df = None

        # Take portion of datasets if specified
        if hasattr(self.data_config, 'portion_percentage'):
            if val_df is not None:
                train_df, val_df, test_df = self._take_portion_of_datasets(
                    train_df, val_df, test_df, self.data_config.portion_percentage
                )
            else:
                train_df, test_df = self._take_portion_of_datasets(
                    train_df, test_df, self.data_config.portion_percentage
                )

        # Create dataset dictionaries
        self.dataset_dict = self._create_dataset_dict(train_df, val_df, test_df)

        # Save to hickle
        self._save_to_hickle(self.dataset_dict)

    def _split_general_case(self, dummy_df):
        if self.data_config.splitting_method == 'sequential':
            total_samples = len(dummy_df)
            train_size = int(self.data_config.train_frac * total_samples)

            if hasattr(self.data_config, 'val_frac') and self.data_config.val_frac > 0:
                val_size = int(self.data_config.val_frac * total_samples)
            else:
                val_size = 0

            if self.data_config.shuffle_sequential:
                # Shuffle using the configured seed
                dummy_df = dummy_df.sample(frac=1, random_state=self.data_config.seednum)

            train_df = dummy_df.iloc[:train_size]
            if val_size > 0:
                val_df = dummy_df.iloc[train_size:train_size+val_size]
                test_df = dummy_df.iloc[train_size+val_size:]
                return train_df, val_df, test_df
            else:
                test_df = dummy_df.iloc[train_size:]
                return train_df, test_df

        elif self.data_config.splitting_method in ['adjusted-without-label-adjustment',
                                                   'adjusted']:
            total_samples = len(dummy_df)
            train_size = int(self.data_config.train_frac * total_samples)
            if hasattr(self.data_config, 'val_frac') and self.data_config.val_frac > 0:
                val_size = int(self.data_config.val_frac * total_samples)
            else:
                val_size = 0
            test_size = total_samples - train_size - val_size

            # Calculate overall class distribution
            class_distribution = dummy_df[self.data_config.label_column_name].value_counts(normalize=True)

            # Create a list of chunk sizes based on percentages of total_samples
            chunk_sizes = [int(total_samples * p) for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 1.0]]

            # get the number of samples for each seq_index and then get the minimum value
            seq_index_counts = dummy_df['seq_index'].value_counts()
            min_samples = seq_index_counts.min()

            # filter the chunk_sizes with min_samples
            chunk_sizes = [i for i in chunk_sizes if i > min_samples]

            best_chunk_size = 0
            best_distribution_score = float('inf')
            best_splits = None

            for chunk_size in chunk_sizes:
                train_df = pd.DataFrame()
                val_df = pd.DataFrame() if val_size > 0 else None
                test_df = pd.DataFrame()

                # Chunk the dummy_df
                chunks = []
                for i in range(0, total_samples, chunk_size):
                    chunk = dummy_df.iloc[i:i+chunk_size].copy()
                    chunks.append(chunk)

                for chunk in chunks:
                    chunk_train_size = int(self.data_config.train_frac * len(chunk))
                    if val_size > 0:
                        chunk_val_size = int(self.data_config.val_frac * len(chunk))

                    # Split chunk into train, val and test
                    chunk_train = chunk.iloc[:chunk_train_size]
                    if val_size > 0:
                        chunk_val = chunk.iloc[chunk_train_size:chunk_train_size+chunk_val_size]
                        chunk_test = chunk.iloc[chunk_train_size+chunk_val_size:]
                    else:
                        chunk_test = chunk.iloc[chunk_train_size:]

                    # Append to respective dataframes
                    train_df = pd.concat([train_df, chunk_train])
                    if val_size > 0:
                        val_df = pd.concat([val_df, chunk_val])
                    test_df = pd.concat([test_df, chunk_test])

                # Calculate distribution score
                train_dist = train_df[self.data_config.label_column_name].value_counts(normalize=True)
                if val_size > 0:
                    val_dist = val_df[self.data_config.label_column_name].value_counts(normalize=True)
                test_dist = test_df[self.data_config.label_column_name].value_counts(normalize=True)

                # lower is better because it is the sum of absolute difference
                dist_score = (
                    (class_distribution - train_dist).abs().sum() +
                    ((class_distribution - val_dist).abs().sum() if val_size > 0 else 0) +
                    (class_distribution - test_dist).abs().sum()
                )
                logger.info(f"""Chunk size: {chunk_size} ({chunk_size/total_samples*100:.2f}%),
                            Distribution score: {dist_score}""")

                if dist_score < best_distribution_score:
                    best_distribution_score = dist_score
                    best_chunk_size = chunk_size
                    best_splits = (train_df, val_df, test_df) if val_size > 0 else (train_df, test_df)

            # Use the best splits found
            if val_size > 0:
                train_df, val_df, test_df = best_splits
            else:
                train_df, test_df = best_splits

            if self.data_config.splitting_method == 'adjusted':
                # Check and adjust class ratios
                for class_label in class_distribution.index:
                    train_class = train_df[train_df[self.data_config.label_column_name] == class_label]
                    if val_size > 0:
                        val_class = val_df[val_df[self.data_config.label_column_name] == class_label]
                    test_class = test_df[test_df[self.data_config.label_column_name] == class_label]

                    # Concatenate data for this class
                    if val_size > 0:
                        class_data = pd.concat([train_class, val_class, test_class])
                    else:
                        class_data = pd.concat([train_class, test_class])

                    # Resplit the data
                    class_train_size = int(self.data_config.train_frac * len(class_data))
                    if val_size > 0:
                        class_val_size = int(self.data_config.val_frac * len(class_data))

                    new_train_class = class_data.iloc[:class_train_size]
                    if val_size > 0:
                        new_val_class = class_data.iloc[class_train_size:class_train_size+class_val_size]
                        new_test_class = class_data.iloc[class_train_size+class_val_size:]
                    else:
                        new_test_class = class_data.iloc[class_train_size:]

                    # Replace the data for this class
                    train_df = train_df[train_df[self.data_config.label_column_name] != class_label]
                    if val_size > 0:
                        val_df = val_df[val_df[self.data_config.label_column_name] != class_label]
                    test_df = test_df[test_df[self.data_config.label_column_name] != class_label]

                    train_df = pd.concat([train_df, new_train_class])
                    if val_size > 0:
                        val_df = pd.concat([val_df, new_val_class])
                    test_df = pd.concat([test_df, new_test_class])

            # Calculate distribution similarity
            train_dist = train_df[self.data_config.label_column_name].value_counts(normalize=True)
            if val_size > 0:
                val_dist = val_df[self.data_config.label_column_name].value_counts(normalize=True)
            test_dist = test_df[self.data_config.label_column_name].value_counts(normalize=True)

            train_similarity = 1 - (class_distribution - train_dist).abs().sum() / 2
            if val_size > 0:
                val_similarity = 1 - (class_distribution - val_dist).abs().sum() / 2
            test_similarity = 1 - (class_distribution - test_dist).abs().sum() / 2

            logger.info(f"Best chunk percentages: {best_chunk_size / total_samples * 100:.2f}%")
            if val_size > 0:
                logger.info(f"""Distribution similarity -
                            Train: {train_similarity:.4f},
                            Val: {val_similarity:.4f},
                            Test: {test_similarity:.4f}""")
            else:
                logger.info(f"""Distribution similarity -
                            Train: {train_similarity:.4f},
                            Test: {test_similarity:.4f}""")

        else:
            raise ValueError(f"""Unknown splitting method:
                             {self.data_config.splitting_method}""")

        # sort train, val and test by sample_idx, ascending
        train_df = train_df.sort_values(by='sample_idx', ascending=True)
        if val_size > 0:
            val_df = val_df.sort_values(by='sample_idx', ascending=True)
        test_df = test_df.sort_values(by='sample_idx', ascending=True)

        if val_size > 0:
            return train_df, val_df, test_df
        return train_df, test_df

    def _take_portion_of_datasets(self,
                                  train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  portion_percentage: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Take the first xx% of each dataset.

        Args:
            train_df (pd.DataFrame): Training dataset
            val_df (pd.DataFrame): Validation dataset (can be None)
            test_df (pd.DataFrame): Test dataset
            portion_percentage (int): Percentage of data to keep (1-100)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Portioned train, validation and test datasets
        """
        if not 1 <= portion_percentage <= 100:
            raise ValueError("portion_percentage must be between 1 and 100")

        portion = portion_percentage / 100

        train_portion = train_df.iloc[:int(len(train_df) * portion)]
        if val_df is not None:
            val_portion = val_df.iloc[:int(len(val_df) * portion)]
        test_portion = test_df.iloc[:int(len(test_df) * portion)]

        if val_df is not None:
            return train_portion, val_portion, test_portion
        return train_portion, test_portion

    def _create_dataset_dict(self, train_df, val_df, test_df):
        dataset_dict = {
            'train': self._filter_dict_by_sample_idx(train_df.to_dict(orient='list')),
            'test': self._filter_dict_by_sample_idx(test_df.to_dict(orient='list')),
        }
        if val_df is not None:
            dataset_dict['val'] = self._filter_dict_by_sample_idx(val_df.to_dict(orient='list'))
        return dataset_dict

    def _save_to_hickle(self, dataset_dict):
        hkl.dump(dataset_dict, self.hickle_fname, mode='w')
        logger.info(f"\nTrain-Test dataset is SAVED as {self.hickle_fname}")

    def convert_to_df(self, any_dict: dict):
        processed_dict = {}
        for key, value in any_dict.items():
            if key in self.special_keys:
                try:
                    processed_dict[key] = [tuple(i) for i in value]
                except:
                    print(key)
                    raise
            else:
                processed_dict[key] = value if isinstance(value, (list, np.ndarray)) else [value]
        df = pd.DataFrame(processed_dict)
        return df
