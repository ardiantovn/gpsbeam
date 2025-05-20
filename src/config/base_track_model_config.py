from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal
from loguru import logger
from src.config.abstract_config import AbstractConfig

@dataclass
class ModelConfig(AbstractConfig):
    model_arch_name: str = 'base-track-model'
    train_epoch: int = 200
    train_batch_size: int = 512
    val_batch_size: int = 1024
    test_batch_size: int = 1024
    num_worker: int = 0

    optimizer_name: Literal["adam"] = 'adam'
    adam_learning_rate: float = 0.01
    adam_weight_decay: float = 1e-4
    adam_opt_milestone_list: List[int] = field(default_factory=lambda: [40, 120])
    adam_opt_gamma: float = 0.1 # learning rate reduction factor

    loss_func_name: str = 'cross-entropy-loss'

    use_early_stopping: bool = False
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-3

    device: Literal["mps", "cpu", "cuda"] = 'cpu'

    # all input column inside this will be
    # concatenated an be used as model input
    model_input_column_list: List[str] = field(default_factory=lambda: ['unit2_loc_minmax_norm'])

     # RNN
    num_classes: int = 32
    rnn_num_layers: int = 2
    rnn_hidden_size: float = 128
    rnn_dropout: float = 0.5

    shuffle_train_dset: bool = True

    seq_len: int = 1
    out_len: int = 1
    dl_task_type: str = 'base_beam_tracking'

    zero_pad_nonconsecutive: bool = False
    ends_input_with_out_len_zeros: bool = False

    def __post_init__(self):
        super().__post_init__()
    
    def show_config(self):
        super().show_config()
