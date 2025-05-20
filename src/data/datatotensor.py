import numpy as np
import torch
from loguru import logger
from typing import Dict, List
from numpy.typing import NDArray
from scipy.io import loadmat
from src.preprocessor.gpsprep import GpsPrep
class DataToTensor(torch.utils.data.Dataset):
    def __init__(self, 
             data_dict: Dict[str, NDArray],
             model_input_column_list: List[str],
             label_column_name: str = '',
             seq_len: int = 8,
             out_len: int = 3,
             zero_pad_nonconsecutive: bool = False,
             ends_input_with_out_len_zeros: bool = False
             ):
        super(DataToTensor, self).__init__()
        
        # Initialize attributes
        self.data_dict = data_dict
        self.model_input_column_list = model_input_column_list
        self.label_column_name = label_column_name
        self.seq_len = seq_len  
        self.out_len = out_len
        self.input_len = self.seq_len + self.out_len
        self.zero_pad_nonconsecutive = zero_pad_nonconsecutive
        self.ends_input_with_out_len_zeros = ends_input_with_out_len_zeros
        self.gps_prep_obj = GpsPrep()
        # Group features by length
        self.feature_groups = {
            1: ["unit2_speed", "unit2_altitude", "unit2_distance", "unit2_height", 
                "unit2_x-speed", "unit2_y-speed", "unit2_z-speed", "unit2_pitch", 
                "unit2_roll", 'sample_idx', "unit2_height_log"],
            2: ["unit2_loc_minmax_norm", 'unit2_loc'],
            3: ["unit2to1_vector", "unit1_loc_vector", "unit2_loc_vector"],
            8: ["unit1_beam_idx_8"],
            16: ["unit1_beam_idx_16"],
            32: ["unit1_beam_idx_32"],
            64: ["unit1_beam_idx_64"],
            216: ["unit1_lidar_scr", "unit1_lidar"]
        }

        # create beam embedding
        beam_size = int(self.label_column_name.split('_')[-1])
        self.beam_embedding_dict = {}
        for i in range(0, beam_size):
            beam_embedding = self.generate_beam_vector(i, beam_size)
            self.beam_embedding_dict[i] = beam_embedding

        # Group data by sequence index
        self.all_data = self._group_data_by_seq_index()

        # Create samples
        self.samples = self._create_samples()

        # Process samples and create tensor dictionaries
        self.tensor_dict_list = self._process_samples()

    def _group_data_by_seq_index(self):
        all_data = {}
        for idx, seq_idx in enumerate(self.data_dict['seq_index']):
            if seq_idx not in all_data:
                all_data[seq_idx] = []
            
            sample_data = {key: value[idx] for key, value in self.data_dict.items()}
            all_data[seq_idx].append(sample_data)
        return all_data

    def _create_samples(self):
        samples = []
        for seq_idx, seq_data in self.all_data.items():
            for start in range(len(seq_data) - self.seq_len - self.out_len + 1):
                end = start + self.seq_len
                label_len = self.out_len + 1
                # we generate possible sequence of sample_idx
                # Ex: [1,2,3,4,5,6,7,8,9,10,11], self.seq_len = 8, self.out_len = 3
                # Later, we will use the first 8 sample_idx as input, and the the last 4 sample_idx as output.
                # Ex: input: [1,2,3,4,5,6,7,8], output: [8,9,10,11]
                # Note: sample_idx 8 is the current sample_idx, and sample_idx 9,10,11 is the future sample_idx.
                possible_sequence = [item['sample_idx'] for item in seq_data[start:end-1+label_len]]
                
                if np.all(np.diff(possible_sequence) == 1):
                    idx_list = [(seq_idx, i) for i in range(start, end-1+label_len)]
                    samples.append(idx_list)
                else:
                    # if the data splitting method is sequential-adjusted,
                    # there is a chance that the sequence is not consecutive.
                    # In this case, we need to find the longest sequence.
                    longest_sequence = self._generate_longest_sequence(possible_sequence)
                    # we remove any sequence that is shorter than self.out_len + 1,
                    # because we have to make sure that the output_sequence is consecutive.
                    if len(longest_sequence) >= self.out_len + 1:
                        idx_list = [i for i, item in enumerate(seq_data) if item['sample_idx'] in longest_sequence]
                        # We handle the case where the longest_sequence length is 
                        # shorter than self.seq_len + self.out_len.
                        # Ex: longest_sequence = [4,5,6,7,8,9,10,11], self.seq_len = 8, self.out_len = 3
                        # we will pad the longest_sequence with the first element of the longest_sequence
                        # so the final longest_sequence will be [4,4,4,4,5,6,7,8,9,10,11].
                        # or if self.zero_pad_nonconsecutive is True, we will pad the first element with 0
                        # so the final longest_sequence will be [0,0,0,0,4,5,6,7,8,9,10,11].
                        idx_list = self._pad_non_consecutive_sequence(idx_list)
                        idx_list = [(seq_idx, idx) for idx in idx_list]
                        samples.append(idx_list)
        return np.array(samples)

    def _process_samples(self):
        tensor_dict_list = []
        for sample_seq in self.samples:
            # split the sequence [1,2,3,4,5,6,7,8,9,10,11] into input and output
            # input: [1,2,3,4,5,6,7,8], output: [8,9,10,11]
            input_seq_data = [self.all_data[seq_idx][i] for seq_idx, i in sample_seq[:self.seq_len]]
            output_seq_data = [self.all_data[seq_idx][i] for seq_idx, i in sample_seq[self.seq_len-1:]]
            
            tensor_dict = self._create_tensor_dict(input_seq_data, output_seq_data)
            tensor_dict_list.append(tensor_dict)
        return np.array(tensor_dict_list)

    def _create_tensor_dict(self, input_seq_data, output_seq_data):
        tensor_dict = {}
        concat_feature_list = []
        input_sample_idx = torch.zeros(self.seq_len, dtype=torch.long)
        input_seq_idx = torch.zeros(self.seq_len, dtype=torch.long)
        input_from_scenario = torch.zeros(self.seq_len, dtype=torch.long)
        input_speed = torch.zeros(self.seq_len, dtype=torch.float)
        input_height = torch.zeros(self.seq_len, dtype=torch.float)
        input_beam_idx = torch.zeros(self.seq_len, dtype=torch.long)
        input_pitch = torch.zeros(self.seq_len, dtype=torch.float)
        input_roll = torch.zeros(self.seq_len, dtype=torch.float)

        for model_input_column in self.model_input_column_list:
            feature_len = next(len for len, features in self.feature_groups.items() if model_input_column in features)
            if self.ends_input_with_out_len_zeros:
                feature_list = torch.zeros((self.seq_len+self.out_len, feature_len))
            else:
                feature_list = torch.zeros((self.seq_len, feature_len))
            
            pad_count = self._count_padding(input_seq_data)

            for i, item in enumerate(input_seq_data):
                
                feature_list[i] = self._process_feature(model_input_column, item)
                input_sample_idx[i] = item['sample_idx']
                input_seq_idx[i] = item['seq_index']
                input_from_scenario[i] = item['from_scenario']
                input_speed[i] = item['unit2_speed']
                input_height[i] = item['unit2_height']
                input_beam_idx[i] = item[self.label_column_name]
                input_pitch[i] = item['unit2_pitch']
                input_roll[i] = item['unit2_roll']

                if self.zero_pad_nonconsecutive:
                    for i in range(0, pad_count+1):
                        feature_list[i] = torch.zeros_like(feature_list[i])
                        input_sample_idx[i] = 0
                        input_seq_idx[i] = 0
                        input_from_scenario[i] = 0
                        input_speed[i] = 0
                        input_height[i] = 0
                        input_beam_idx[i] = 0
                        input_pitch[i] = 0
                        input_roll[i] = 0


            concat_feature_list.append(feature_list)

        feature_list = torch.cat(concat_feature_list, dim=1)

        out_beam = torch.tensor([item[self.label_column_name] for item in output_seq_data], dtype=torch.long)
        future_sample_idx = torch.tensor([item['sample_idx'] for item in output_seq_data], dtype=torch.long)


        tensor_dict.update({
            'input_from_scenario': input_from_scenario,
            'input_seq_idx': input_seq_idx,
            'input_speed': input_speed,
            'input_height': input_height,
            'input_value': feature_list,
            'input_beam_idx': input_beam_idx,
            'input_pitch': input_pitch,
            'input_roll': input_roll,
            'label': out_beam,
            'future_sample_idx': future_sample_idx,
            'input_sample_idx': input_sample_idx
        })

        return tensor_dict

    def _count_padding(self, arr:List[int]):
        padding_count = 0
        for num in arr:
            if num == arr[0]:
                padding_count += 1
            else:
                break
        return padding_count-1

    def _process_feature(self, model_input_column, item):
        if 'lidar' in model_input_column:
            try:
                lidar_data = loadmat(item['unit1_lidar_scr'])['data'][:, 0] / 10
            except:
                lidar_data = loadmat(item['unit1_lidar'])['data'][:, 0] / 10
            return torch.tensor(lidar_data)
        elif 'unit1_beam_idx' in model_input_column:
            beam_idx = int(item[model_input_column])
            beam_vector = self.beam_embedding_dict[beam_idx]
            return torch.tensor(beam_vector)
        else:
            return torch.tensor(item[model_input_column])

    def _generate_longest_sequence(self, numbers: List[int]):
        sequences = []
        current_sequence = [numbers[0]]
        
        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i-1] + 1:
                current_sequence.append(numbers[i])
            else:
                sequences.append(current_sequence)
                current_sequence = [numbers[i]]
        
        sequences.append(current_sequence)
        longest_sequence = max(sequences, key=len)
        
        return longest_sequence

    def _pad_non_consecutive_sequence(self, longest_sequence:List[int]):
        if self.zero_pad_nonconsecutive:
            padding = [-1] * (self.seq_len + self.out_len - len(longest_sequence))
        else:
            padding = [longest_sequence[0]] * (self.seq_len + self.out_len - len(longest_sequence))
        padded_sequence = padding + longest_sequence
        return padded_sequence
    
    def generate_beam_vector(self, beam_idx: int, beam_size: int) -> np.ndarray:
        """Generate a random beam vector based on beam index.
           the vector is drawn from a Gaussian distribution with 
           zeros mean and unity standard deviation.
           The length of the vector is the same as the beam size.
        
        Args:
            beam_idx (int): Beam index
            
        Returns:
            numpy.ndarray: Random vector drawn from N(0,1) distribution
        """
            
        # Set random seed for reproducibility
        np.random.seed(42 + beam_idx)  # Different seed for each beam
        
        # Generate random vector from normal distribution(mean=0, std=1)
        random_vector = np.random.normal(loc=0.0, scale=1.0, size=beam_size)
        
        return random_vector
    
    def __len__(self):
        return len(self.tensor_dict_list)
    
    def __getitem__(self, idx: int):
        return self.tensor_dict_list[idx]
    