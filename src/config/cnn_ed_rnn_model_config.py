from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal
from loguru import logger
from src.config.abstract_config import AbstractConfig

@dataclass
class ModelConfig(AbstractConfig):
    model_arch_name: str = 'cnn-ed-rnn-model'
    train_epoch: int = 20
    train_batch_size: int = 32
    val_batch_size: int = 1024
    test_batch_size: int = 1024
    num_worker: int = 0

    optimizer_name: Literal["adam"] = 'adam'
    adam_learning_rate: float = 1e-3
    adam_weight_decay: float = 0
    adam_opt_milestone_list: List[int] = field(default_factory=lambda: [10, 15])
    adam_opt_gamma: float = 0.1 #  # reduces learning rate by X-times at each milestone

    loss_func_name: str = 'cross-entropy-loss'
    weight_balancing_method: Literal["max", "median", "mean", "none"] = "none"

    use_early_stopping: bool = False
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-3

    device: Literal["mps", "cpu", "cuda"] = 'cpu'

    # all input column inside this will be
    # concatenated an be used as model input
    model_input_column_list: List[str] = field(default_factory=lambda: ['unit1_lidar_scr'])

    #CNN
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64])
    
    # RNN
    num_classes: int = 64
    rnn_num_layers: int = 1
    rnn_hidden_size: float = 64
    rnn_dropout: float = 0
    cnn_dropout: float = 0
    
    # MLP
    mlp_layer_sizes: List[int] = None

    shuffle_train_dset: bool = True

    seq_len: int = 8
    out_len: int = 3

    zero_pad_nonconsecutive: bool = False
    ends_input_with_out_len_zeros: bool = False
    
    dl_task_type: Literal["base_beam_tracking", "beam_tracking"] = 'beam_tracking'

    def __post_init__(self):
        super().__post_init__()
    
    def show_config(self):
        super().show_config()
