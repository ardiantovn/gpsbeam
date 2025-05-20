import pandas as pd

from src.data.dataprep import DataPrep
from src.data.dataplot import DataPlot
from src.data.datarecap import DataRecap
from src.modelprep import ModelPrep
from src.config.data_config import DataConfig
from src.config.cnn_ed_rnn_model_config import ModelConfig
from src.config.experiment_config import ExperimentConfig

from loguru import logger
from typing import List, Dict, Any
import numpy as np

class RunExperiment:

    def __init__(self, 
                 data_config: DataConfig, 
                 model_config: ModelConfig, 
                 experiment_config: ExperimentConfig):
        self.data_config_obj = data_config
        self.model_config_obj = model_config
        self.exp_config_obj = experiment_config

    def run(self):  
        recap_list = []
        prev_data_config = None
        for combo in self.exp_config_obj.combinations:
            if 'model_input_column_list' not in combo.keys():
                combo['model_input_column_list'] = self.model_config_obj.model_input_column_list
            for key, value in combo.items():
                if key in self.data_config_obj.__dict__.keys():
                    self.data_config_obj.__dict__[key] = value
                    self.data_config_obj.__post_init__()
                    
                    logger.debug(f"updated data_config |{key} to {value}")
                if key in self.model_config_obj.__dict__.keys():
                    self.model_config_obj.__dict__[key] = value
                    self.model_config_obj.__post_init__()
                    logger.debug(f"updated model_config | {key} to {value}")
            
            if prev_data_config is None or self.data_config_obj.__dict__ != prev_data_config:
                self.dataprep_obj = DataPrep(self.data_config_obj)
                self.dataprep_obj.get_train_val_test_dataset()
                prev_data_config = self.data_config_obj.__dict__.copy()
            
            self.modelprep_obj = ModelPrep(experiment_config=self.exp_config_obj,
                                    data_config=self.data_config_obj,
                                    model_config=self.model_config_obj,
                                    dataset_dict=self.dataprep_obj.dataset_dict)
            self.modelprep_obj.start_logging()  
            self.modelprep_obj.main_training_loop()
            test_results = self.modelprep_obj.test(best_model_fname=self.modelprep_obj.best_model_fname)
            self.modelprep_obj.stop_logging()

            result_metrics = test_results['metrics']
            del test_results['metrics']
            recap = {**combo, **test_results, **result_metrics}
            recap_list.append(recap)

        df_recap = pd.DataFrame(recap_list)
        df_recap.to_csv(f'{self.modelprep_obj.model_recap_dir}/exp_test_result_recap.csv', index=False)

        dataplot_obj = DataPlot()
        datarecap_obj = DataRecap()
    
        for key, value in self.exp_config_obj.exp_dict.items():
            if key != 'seednum':
                if len(value) > 1:
                    for num_classes in self.exp_config_obj.exp_dict['num_classes']:
                        select_df = df_recap[df_recap['num_classes'] == num_classes]
                        dataplot_obj.plot_boxplot(df=select_df, 
                                                 num_classes=num_classes,
                                                 folder_name=self.modelprep_obj.model_recap_dir,
                                                 selected_column=key)
                    datarecap_obj.aggregate_metrics(df=df_recap,
                                                    folder_name=self.modelprep_obj.model_recap_dir,
                                                    selected_column=key)
        logger.info(f"Model recap saved at {self.modelprep_obj.model_recap_dir}")

    def _run_single_experiment(self, config: Dict[str, int]) -> float:
        self.model_config_obj.embed_size = config['embed_size']
        self.model_config_obj.hidden_size = config['hidden_size']
        self.model_config_obj.num_layers = config['num_layers']

        self.dataprep_obj = DataPrep(self.data_config_obj)
        self.dataprep_obj.get_train_val_test_dataset()
        
        self.modelprep_obj = ModelPrep(experiment_config=self.exp_config_obj,
                                  data_config=self.data_config_obj,
                                  model_config=self.model_config_obj,
                                  dataset_dict=self.dataprep_obj.dataset_dict)
        self.modelprep_obj.start_logging()
        self.modelprep_obj.main_training_loop()
        test_results = self.modelprep_obj.test(best_model_fname=self.modelprep_obj.best_model_fname)
        self.modelprep_obj.stop_logging()

        # Assuming the main metric for accuracy is 'pred0_top1_test_acc_percent'
        accuracy = test_results['metrics'].get('pred0_top1_test_acc_percent', 0)
        return accuracy
