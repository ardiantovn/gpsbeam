import os
import copy
import datetime
import inspect
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import DataLoader
from loguru import logger
from typing import Dict, List, Tuple, Any, Literal
from tqdm import tqdm

from src.utils.metric_handler import MetricHandler
from src.data.dataplot import DataPlot
from src.data.datatotensor import DataToTensor
from src.data.base_datatotensor import BaseDataToTensor
from src.utils.early_stop import EarlyStopping
import src.arch as ModelClasses

class ModelPrep:
    def __init__(self,
                 experiment_config: Any,
                 data_config: Any,
                 model_config: Any,
                 dataset_dict: Dict[str, Any],
                 is_use_combined_dataset: bool = False,
                 scenario_num_list: List[int] = None,
                 **kwargs):
        torch.manual_seed(data_config.seednum)
        self.data_config = data_config
        self.exp_config = experiment_config
        self.model_config = model_config
        self.dataset_dict = dataset_dict
        self.is_use_combined_dataset = is_use_combined_dataset
        self.scenario_num_list = scenario_num_list
        self.__dict__.update(kwargs)

        self._setup_paths()
        self.metric_handler = MetricHandler(self.exp_config, self.model_config)  # Initialize metric_handler here


    def _setup_paths(self):
        self.main_folder = '/'.join(os.getcwd().replace('\\', '/').split("/")[:-2])
        self.day_time = datetime.datetime.now().strftime('%m-%d-%Y_%H_%M_%S')
        self.exp_folder_name = f'data/experiment_result/{self.exp_config.exp_folder_name}'
        self.model_recap_dir = f"data/experiment_result/{self.exp_config.exp_folder_name}/model_recap"
        self.model_recap_dir = os.path.join(self.main_folder,self.model_recap_dir)
        logger.info(f"model_recap_dir Name: {self.model_recap_dir}")
        self.exp_folder_name = os.path.join(self.main_folder, self.exp_folder_name, self.day_time)
        self.dl_code = os.path.join(self.exp_folder_name, 'dl_generated')
        self.checkpoint_dir = f'{self.dl_code}/model_checkpoint'
        self.model_inference_result_dir = f'{self.dl_code}/model_inference_result'
        self.model_measurement_dir = f'{self.dl_code}/model_measurement'

        self.name_str = (
                        f"arch_{self.model_config.model_arch_name}_"
                        f"nclass_{self.model_config.num_classes}_"
                        )

        self.best_model_fname = (
                        f"{self.checkpoint_dir}/"
                        f"{self.name_str}"
                        ".pth"
                        )
        self.onnx_model_fname = (
                        f"{self.checkpoint_dir}/"
                        f"{self.name_str}"
                        ".onnx"
                        )

        self._create_directories()

    def _create_directories(self):
        directories = [self.exp_folder_name, self.model_recap_dir,
                       self.dl_code, self.checkpoint_dir,
                       self.model_inference_result_dir,
                       self.model_measurement_dir,
                       ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _setup_model(self):
        self.model = self._get_model()
        self.model.to(self.model_config.device)

    def _get_model(self) -> nn.Module:
        if self.model_config.num_classes is None:
            raise ValueError("num_classes must be specified")

        model_class = None
        for _, cls in ModelClasses.__dict__.items():
            if inspect.isclass(cls) and hasattr(cls, 'ARCH_NAME') and \
                self.model_config.model_arch_name == cls.ARCH_NAME:
                model_class = cls
                break

        if model_class is None:
            raise ValueError(f"Unknown model_arch_name: {self.model_config.model_arch_name}")

        model_params: Dict[str, Any] = {"num_classes": self.model_config.num_classes}

        
        if self.model_config.model_arch_name in ['cnn-ed-gru-model']:
            model_params.update({
                "cnn_channels": self.model_config.cnn_channels,
                "feature_len": self.model_config.feature_len,
                "num_classes": self.model_config.num_classes,
                "rnn_num_layers": self.model_config.rnn_num_layers,
                "rnn_hidden_size": self.model_config.rnn_hidden_size,
                "rnn_dropout": self.model_config.rnn_dropout,
                "cnn_dropout": self.model_config.cnn_dropout,
                "mlp_layer_sizes": self.model_config.mlp_layer_sizes,
                "out_len": self.model_config.out_len
            })
        elif self.model_config.model_arch_name in ['base-pred-model']:
            model_params.update({
                "feature_len": self.model_config.feature_len,
                "num_classes": self.model_config.num_classes,
                "mlp_hidden_layer_sizes": self.model_config.mlp_hidden_layer_sizes
            })
        elif self.model_config.model_arch_name in ['base-track-model']:
            model_params.update({
                "feature_len": self.model_config.feature_len,
                "num_classes": self.model_config.num_classes,
                "rnn_num_layers": self.model_config.rnn_num_layers,
                "rnn_hidden_size": self.model_config.rnn_hidden_size,
                "rnn_dropout": self.model_config.rnn_dropout,
                "out_len": self.model_config.out_len
            })
        else:
            raise ValueError(f"Unknown model_arch_name: {self.model_config.model_arch_name}")

        return model_class(**model_params)

    def _setup_data_loaders(self):

        train_num = len(self.dataset_dict['train']['sample_idx'])
        if 'val' in self.dataset_dict:
            val_num = len(self.dataset_dict['val']['sample_idx'])
        else:
            val_num = 0
        test_num = len(self.dataset_dict['test']['sample_idx'])
        total_num = train_num + val_num + test_num
        
        counts = np.bincount(self.dataset_dict['train'][self.model_config.label_column_name], 
                                minlength=self.model_config.num_classes)
           
        # Compute train dataset entropy
        prob = counts/np.sum(counts) # count probability
        prob = prob[prob > 0] # Remove zeros to avoid log(0)
        self.train_entropy = -np.sum(prob * np.log(prob)) # compute entropy

        logger.info(f"""
                    RAW DATASET INFO
                    ------------------------------
                    Scenario Num                                    : {self.data_config.scenario_num},
                    SplittingMethod                                 : {self.data_config.splitting_method}
                    Portion Percentage                              : {self.data_config.portion_percentage}
                    Training                                        : {train_num} samples [{train_num/total_num*100:.2f}%]
                    Validation                                      : {val_num} samples [{val_num/total_num*100:.2f}%]
                    Testing                                         : {test_num} samples [{test_num/total_num*100:.2f}%]
                    Train Dataset Entropy                           : {self.train_entropy:.2f}
                    Total                                           : {total_num} samples\n""")
        
        if self.model_config.dl_task_type == 'base_beam_prediction' or self.model_config.dl_task_type == 'base_beam_tracking':
            train_dataset = self._create_base_pred_tensor_dataset(data_dict=self.dataset_dict['train'])
            if 'val' in self.dataset_dict:
                val_dataset = self._create_base_pred_tensor_dataset(data_dict=self.dataset_dict['val'])
            else:
                val_dataset = None
            test_dataset = self._create_base_pred_tensor_dataset(data_dict=self.dataset_dict['test'])
        else:
            train_dataset = self._create_tensor_dataset(data_dict=self.dataset_dict['train'])
            if 'val' in self.dataset_dict:
                val_dataset = self._create_tensor_dataset(data_dict=self.dataset_dict['val'])
            else:
                val_dataset = None
            test_dataset = self._create_tensor_dataset(data_dict=self.dataset_dict['test'])

        train_num = len(train_dataset.samples)
        if val_dataset is not None:
            val_num = len(val_dataset.samples)
        else:
            val_num = 0
        test_num = len(test_dataset.samples)
        total_num = train_num + val_num + test_num
        logger.info(f"""
                    FILTERED DATASET INFO
                    ------------------------------
                    Scenario Num                                    : {self.data_config.scenario_num},
                    SplittingMethod                                 : {self.data_config.splitting_method}
                    Portion Percentage                              : {self.data_config.portion_percentage}
                    Training                                        : {train_num} samples [{train_num/total_num*100:.2f}%]
                    Validation                                      : {val_num} samples [{val_num/total_num*100:.2f}%]
                    Testing                                         : {test_num} samples [{test_num/total_num*100:.2f}%]
                    Total                                           : {total_num} samples\n""")
        logger.info("---"*50)

        self.train_loader = self._create_dataloader(dataset=train_dataset,
                            batch_size=self.model_config.train_batch_size,
                            shuffle=self.model_config.shuffle_train_dset)
        if val_dataset is not None:
            self.val_loader = self._create_dataloader(dataset=val_dataset,
                            batch_size=self.model_config.val_batch_size,
                            shuffle=False)
        else:
            self.val_loader = None
        self.test_loader = self._create_dataloader(dataset=test_dataset,
                            batch_size=self.model_config.test_batch_size,
                            shuffle=False)

    def _create_tensor_dataset(self, data_dict: Dict):
        return DataToTensor(
            data_dict=data_dict,
            model_input_column_list=self.model_config.model_input_column_list,
            label_column_name=self.model_config.label_column_name,
            seq_len=self.model_config.seq_len,
            out_len=self.model_config.out_len,
            zero_pad_nonconsecutive=self.model_config.zero_pad_nonconsecutive,
            ends_input_with_out_len_zeros=self.model_config.ends_input_with_out_len_zeros
        )
    
    def _create_base_pred_tensor_dataset(self, data_dict: Dict):
        return BaseDataToTensor(
            data_dict=data_dict,
            model_input_column_list=self.model_config.model_input_column_list,
            label_column_name=self.model_config.label_column_name,
            seq_len=self.model_config.seq_len,
            out_len=self.model_config.out_len
        )

    def _create_dataloader(self,
                           dataset: torch.utils.data.Dataset,
                           batch_size: int,
                           shuffle: bool
                           ):
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.model_config.num_worker,
            pin_memory=True,
        )

    def _setup_training(self):
        self._setup_loss_function()
        self._setup_optimizer_and_scheduler()
        self.early_stopping = self._setup_early_stopping()

    def _setup_loss_function(self):
        if self.model_config.loss_func_name == 'cross-entropy-loss':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.model_config.loss_func_name}")

    def _setup_optimizer_and_scheduler(self):
        if self.model_config.optimizer_name == 'adam':
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=self.model_config.adam_learning_rate,
                weight_decay=self.model_config.adam_weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.opt,
                milestones=self.model_config.adam_opt_milestone_list,
                gamma=self.model_config.adam_opt_gamma
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.model_config.optimizer_name}")

    def start_logging(self):
        self.logfile = f'{self.exp_folder_name}/model_experiment.log'
        if not os.path.exists(self.logfile):
            self.modelprep_log = logger.add(self.logfile,
                                            level="INFO",
                                            colorize=False,
                                            backtrace=True,
                                            diagnose=True)
        else:
            self.modelprep_log = False
        self.exp_config.show_config()
        self.data_config.show_config()
        self.model_config.show_config()

    def stop_logging(self):
        if self.modelprep_log:
            logger.remove(self.modelprep_log)

    def _train(self,
                data_loader: DataLoader,
                callbacks=None) -> Dict[str, Any]:
        self.model.train()
        if self.model_config.optimizer_name == 'adam-free':
            self.opt.train()

        running_loss = 0
        batch_count = 0

        start_time = time.time()
        for data in tqdm(data_loader, desc="Training"):
            self.opt.zero_grad(set_to_none=True)
            inference_result = self.inference(data, func_mode='train')
            outputs, labels = inference_result['outputs'], inference_result['labels']
            # outputs form -> unnormalized logits
            # labels form -> label indices
            loss = self.criterion(outputs.reshape(-1, self.model_config.num_classes),
                                      labels.flatten())
            loss.backward()
            self.opt.step()
            running_loss += loss.item()
            batch_count += 1
        end_time = time.time()
        avg_loss = running_loss / batch_count
        if self.is_use_combined_dataset:
            results_dict = {
                'scenario_num': self.scenario_num_list,
                'avg_loss': avg_loss,
                'training_duration_ms': round((end_time - start_time) * 1000, 3),
            }
        else:
            results_dict = {
                'scenario_num': self.data_config.scenario_num,
                'avg_loss': avg_loss,
                'training_duration_ms': round((end_time - start_time) * 1000, 3),
            }

        if callbacks is not None:
            for callback in callbacks:
                callback()

        if self.is_use_combined_dataset:
            results_dict = {
                'scenario_num': self.scenario_num_list,
                'avg_loss': avg_loss,
                'training_duration_ms': round((end_time - start_time) * 1000, 3),
            }
        else:
            results_dict = {
                'scenario_num': self.data_config.scenario_num,
                'avg_loss': avg_loss,
                'training_duration_ms': round((end_time - start_time) * 1000, 3),
            }

        return results_dict

    def inference(self,
                tensor_dict: Dict[str, torch.Tensor],
                func_mode: str = 'test') -> Dict[str, Any]:

        inputs = tensor_dict['input_value'].to(self.model_config.device, non_blocking=True)
        labels = tensor_dict['label'].to(self.model_config.device, non_blocking=True)
        if self.model_config.model_arch_name in ['base-pred-model']:
            h = None
        else:
            h = self.model.init_hidden(inputs.shape[0])

        if 'test' in func_mode:
            with torch.no_grad():
                if self.model_config.model_arch_name in ['base-pred-model']:
                    outputs = self.model(inputs)
                else:
                    outputs, h = self.model(inputs, h)
        else:
            if self.model_config.model_arch_name in ['base-pred-model']:
                outputs = self.model(inputs)
            else:
                outputs, h = self.model(inputs, h)

        outputs = outputs[:,-self.model_config.pred_num:,:]

        result = {
            'input_from_scenario': tensor_dict['input_from_scenario'],
            'input_seq_idx': tensor_dict['input_seq_idx'],
            'input_speed': tensor_dict['input_speed'],
            'input_height': tensor_dict['input_height'],
            'input_pitch': tensor_dict['input_pitch'],
            'input_roll': tensor_dict['input_roll'],
            'labels': labels,
            'outputs': outputs,
            'input_sample_idx': tensor_dict['input_sample_idx'],
            'future_sample_idx': tensor_dict['future_sample_idx']
        }
        return result

    def _validate(self,
                  data_loader: DataLoader,
                  func_mode: Literal['val', 'test'],
                  verbose: bool = True) -> Dict[str, Any]:
        self.model.eval()
        if self.model_config.optimizer_name == 'adam-free':
            self.opt.eval()

        running_loss = 0
        batch_results_list = []

        self.metric_handler.reset()  # Reset the metric handler

        with torch.no_grad():
            for tensor_dict in tqdm(data_loader, desc=func_mode):
                batch_results = self.inference(tensor_dict, func_mode)
                outputs, labels = batch_results['outputs'], batch_results['labels']
                # outputs form -> unnormalized logits
                # labels form -> label indices
                loss = self.criterion(outputs.reshape(-1, self.model_config.num_classes),
                                      labels.flatten())
                running_loss += loss.item()

                batch_results['loss'] = loss.item()
                batch_results_list.append(batch_results)

                self.metric_handler.update_metrics_batch_first(outputs, labels, func_mode)

        avg_loss = round(running_loss / len(data_loader), 3)
        metrics = self.metric_handler.calculate_final_metrics_batch_first(func_mode, verbose)

        # Process each sample outside the loop
        inference_results = []
        for batch_results in batch_results_list:
            for i in range(len(batch_results['labels'])):
                sample_result = {key: value[i] if isinstance(value, torch.Tensor) else value
                                 for key, value in batch_results.items()}
                inference_results.append(sample_result)

        if self.is_use_combined_dataset:
            results_dict = {
                'scenario_num': self.scenario_num_list,
                'avg_loss': avg_loss,
                'metrics': metrics,
                'inference_results': batch_results_list
            }
        else:
            results_dict = {
                'scenario_num': self.data_config.scenario_num,
                'avg_loss': avg_loss,
                'metrics': metrics,
                'inference_results': inference_results
            }
        return results_dict

    def test(self,
             best_model_fname: str,
             verbose: bool = True,
             func_mode: Literal['test'] = 'test') -> Dict[str, Any]:
        epoch = 0
        self._load_best_checkpoint(best_model_fname)

        # Get model information
        model_info = self._get_model_info()

        # Run test
        self.metric_handler.reset()
        test_results = self._validate(data_loader=self.test_loader, func_mode=func_mode, verbose=verbose)

        recap_metrics = self._save_inference_detail(results_dict=test_results, epoch=epoch, func_mode=func_mode)
        test_results['model_size_MiB'] = model_info['model_size_MiB']
        
        for metrics_name, metrics_value in recap_metrics.items():
            for pred in range(len(metrics_value)):
                if "mean_power_loss_db" in metrics_name:
                    if self.model_config.dl_task_type == 'base_beam_tracking':
                        test_results[f'pred{pred+1}_top1_{metrics_name}'] = round(metrics_value[pred], 3)
                    else:
                        test_results[f'pred{pred}_top1_{metrics_name}'] = round(metrics_value[pred], 3)
                else:
                    test_results[f'pred{pred}_top1_{metrics_name}'] = round(metrics_value[pred], 3)
        
        # for pred in range(len(mean_power_loss_db)):
        #     test_results[f'pred{pred}_top1_mean_power_loss_db'] = round(mean_power_loss_db[pred], 3)

        # Add model information to test results
        results_dict = {"day_time": self.day_time}
        results_dict.update(test_results)
        results_dict.update(model_info)

        self._save_epoch_vs_loss(results_list=[{"epoch":epoch,
                                                "avg_loss": test_results['avg_loss']}],
                                                func_mode=func_mode)
        self._save_metrics_result(results_list=[test_results], func_mode=func_mode)
        
        # save onnx
        sample_input = next(iter(self.test_loader))['input_value']
        sample_input_cpu = sample_input.cpu()
        model_cpu = self.model.cpu()
        
        torch.onnx.export(model_cpu, 
                        sample_input_cpu, 
                        self.onnx_model_fname,
                        export_params=True,
                        do_constant_folding=True,
                        opset_version=10,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
        
        # Move model back to original device
        self.model.to(self.model_config.device)
        logger.info(f"Saved ONNX model to {self.onnx_model_fname}")

        logger.info("TEST RESULTS")

        for k, v in test_results['metrics'].items():
            logger.info(f"{k}: {v}")
        
        for k, v in test_results.items():
            if k not in ['metrics', 'inference_results']:
                logger.info(f"{k}: {v}")

        # Print model architecture
        logger.info("Model Architecture:")
        logger.info(self.model)

        del test_results['inference_results']
        return test_results

    def _get_model_info(self) -> Dict[str, Any]:
        # Get input size from the first batch of the test loader
        sample_input = next(iter(self.test_loader))['input_value']
        input_size = sample_input.shape

        # Get model summary
        self.model.to(self.model_config.device)
        # Generate model summary with input and hidden state
        model_summary = torchinfo.summary(self.model,
                                        input_size=input_size,
                                        device=self.model_config.device)

        # Calculate model size in MiB
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_MiB = (param_size + buffer_size) / 1024**2

        return {
            'model_macs': model_summary.total_mult_adds,
            'model_num_param': model_summary.total_params,
            'model_size_MiB': round(model_size_MiB, 2)
        }

    def _load_best_checkpoint(self,
                            best_model_fname: str):

        # Assuming the best model's name is preserved in `best_model_fname`, load the best model
        logger.debug(f"""Get Model: {self.model_config.model_arch_name}
                         NUM CLASSES: {self.model_config.num_classes}""")
        self._setup_model()
        logger.info(f"Model loaded from {best_model_fname}")
        self.model.load_state_dict(torch.load(best_model_fname))
        self.model.to(self.model_config.device)  # Ensure model is on the correct device

    def _save_epoch_vs_loss(self,
                            results_list: List[Dict[str, Any]],
                            func_mode: Literal['train','val', 'test'] = 'val',
                            is_finetune: bool = False):
        df = pd.DataFrame(results_list)
        drop_col = ['metrics', 'inference_results']
        for i in drop_col:
            if i in df.columns:
                df = df.drop(i, axis=1)

        #rename column avg_loss
        df = df.rename(columns={'avg_loss': f'{func_mode}_avg_loss'})
        if is_finetune:
            fname = f'{self.model_measurement_dir}/finetune_{func_mode}_epoch_vs_loss.csv'
        else:
            fname = f'{self.model_measurement_dir}/{func_mode}_epoch_vs_loss.csv'
        df.to_csv(fname, index=False)
        return df
        # logger.info(f'Saved {func_mode} epoch vs loss to {fname}')
    
    # Process each prediction in the batch
    def process_outputs(self, x: List[List[float]]) -> List[List[float]]:
        result = []
        for i in range(len(x)):
            # Sort probabilities in descending order for each prediction
            result.append(sorted(x[i], reverse=True))
        return result

    def _save_inference_detail(self,
                               epoch: int,
                               results_dict: Dict[str, Any],
                               func_mode: Literal['train', 'val', 'test'] = 'val'):
        processed_data = results_dict['inference_results']
        # Get top 5 indices for all outputs at once
        top5_indices = torch.topk(torch.stack([item['outputs'] for item in processed_data]), 5, dim=-1).indices.tolist()
        
        # Assign top5_indices to each item in processed_data
        for item, indices in zip(processed_data, top5_indices):
            item['top5_preds'] = indices

        df = pd.DataFrame(processed_data)
        
        # Apply softmax to the outputs column to convert logits to probabilities, 
        # then sort the probabilities in descending order
        df['outputs'] = df['outputs'].apply(lambda x: torch.nn.functional.softmax(x, dim=1).cpu().numpy())
        df['outputs'] = df['outputs'].apply(self.process_outputs)
        
        # Calculate how many beams needed to achieve each confidence threshold
        for threshold in self.exp_config.confidence_threshold_list:
            col_name = f'n_beams_for_{int(threshold*100)}pct_conf'
            df[col_name] = df['outputs'].apply(
                lambda x: [min(np.argmax(np.cumsum(pred) > threshold) + 1, len(pred)) for pred in x]
            )

        # drop outputs
        df = df.drop(columns=['outputs'])
        # convert all tensor to integer or list
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
        
        # move column 'labels' to the end
        cols = df.columns.tolist()
        cols.remove('labels')
        cols.append('labels')
        df = df[cols]

        # round avg_loss to 3 decimal places
        df['loss'] = df['loss'].round(3)
        df['train_entropy'] = self.train_entropy

        # get power array
        if 'test' in func_mode:
            unit1_pwr_values_arr = self.dataset_dict['test']['unit1_pwr_60ghz']
        else:
            unit1_pwr_values_arr = self.dataset_dict[func_mode]['unit1_pwr_60ghz']
        unit1_pwr_values_arr = np.array([np.loadtxt(i) 
                                              for i in unit1_pwr_values_arr])
        
        # resampling power_values depends on num_classes
        max_beams = unit1_pwr_values_arr[0].shape[-1]
        divider = max_beams // self.model_config.num_classes
        beam_idxs = np.arange(0, max_beams, divider)
        unit1_pwr_values_arr = unit1_pwr_values_arr[:,beam_idxs]
        if 'test' in func_mode:
            sample_idx_arr = self.dataset_dict['test']['sample_idx']
        else:
            sample_idx_arr = self.dataset_dict[func_mode]['sample_idx']
        temp_df = pd.DataFrame({'sample_idx': sample_idx_arr,
                                'unit1_pwr_60ghz': unit1_pwr_values_arr.tolist()})
        df[['true_power', 'pred_power', 'power_loss_ratio', 'power_loss_db', 
            'noise_power', 'n_beams_within_tresh', 'avg_power']] = df.apply(lambda x: 
                                                                pd.Series(self.get_power_loss_ratio(temp_df, 
                                                                              x['future_sample_idx'], 
                                                                              x['labels'], 
                                                                              x['top5_preds'])), axis=1)
        # Calculate reliability metrics for different thresholds (1dB, 3dB, 6dB)
        for thresh in [1, 3, 6]:
            col_name = f'is_power_loss_leq_{thresh}db'
            df[col_name] = df['power_loss_db'].apply(lambda x: [val <= thresh for val in x])
            
        if epoch < 10:
            epoch = f"0{epoch}"
        fname = f'{self.model_inference_result_dir}/{func_mode}_epoch_{epoch}_inference_detail.csv'
        df.to_csv(fname, index=False)
        logger.info(f"Saved inference detail to {fname}")
        
        recap_metrics = {}
        
        # Calculate mean power loss db
        power_loss_ratios = np.array(df['power_loss_ratio'].tolist())
        mean_power_loss_db = 10 * np.log10(np.mean(power_loss_ratios, axis=0))
        recap_metrics['mean_power_loss_db'] = mean_power_loss_db
        
        # Calculate mean number of beams needed to achieve each confidence threshold
        # Calculate overhead saving for each confidence threshold
        for conf in self.exp_config.confidence_threshold_list:
            conf_pct = int(conf*100)
            beam_col = f'n_beams_for_{conf_pct}pct_conf'
            overhead_saving_col = f'overhead_saving_for_{conf_pct}pct_conf'
            if beam_col in df.columns:
                beams_list = np.array(df[beam_col].tolist())
                recap_metrics[beam_col] = np.mean(beams_list, axis=0)
                recap_metrics[overhead_saving_col] = (1 - (np.mean(beams_list, axis=0) / self.model_config.num_classes))*100
        
        # Calculate reliability for each threshold
        for thresh in self.exp_config.power_loss_db_threshold_list:
            col_name = f'is_power_loss_leq_{thresh}db'
            is_power_loss_leq = np.array(df[col_name].tolist())
            reliability = np.mean(is_power_loss_leq, axis=0) * 100
            recap_metrics[f'reliability_power_loss_leq_{thresh}db_pct'] = reliability
        return recap_metrics
    
    def get_power_loss_ratio(self, 
                             temp_df: pd.DataFrame, 
                             future_sample_idx: List[int], 
                             labels: List[int], 
                             top5_preds: List[List[int]]):
        avg_power_list = []
        true_power_list = []
        pred_power_list = []
        noise_power_list = []
        power_loss_ratio_list = []
        power_loss_db_list = []
        n_beams_within_tresh_list = []
        for i, item in enumerate(future_sample_idx):
            sample_idx_power = temp_df.loc[temp_df['sample_idx'] == item, 'unit1_pwr_60ghz'].values[0]
            avg_power = np.mean(sample_idx_power)
            avg_power_list.append(avg_power)
            label = labels[i]
            pred = top5_preds[i][0]
            true_power = sample_idx_power[label]
            pred_power = sample_idx_power[pred]
            power_noise = np.min(sample_idx_power)
            noise_power_list.append(power_noise)
            power_loss_ratio = (true_power-0.5*power_noise)/(pred_power-0.5*power_noise)
            true_power_list.append(true_power)
            pred_power_list.append(pred_power)
            power_loss_ratio_list.append(power_loss_ratio)
            power_loss_db = 10*np.log10(power_loss_ratio)
            power_loss_db_list.append(power_loss_db)
            tresh = 0.7
            n_beams_within_tresh = np.sum(np.array(sample_idx_power) > tresh*true_power)
            n_beams_within_tresh_list.append(n_beams_within_tresh)
        # to numpy array
        true_power_list = np.array(true_power_list)
        pred_power_list = np.array(pred_power_list)
        power_loss_ratio_list = np.array(power_loss_ratio_list)
        noise_power_list = np.array(noise_power_list)
        n_beams_within_tresh_list = np.array(n_beams_within_tresh_list)
        avg_power_list = np.array(avg_power_list)
        return true_power_list, pred_power_list, power_loss_ratio_list, power_loss_db_list, noise_power_list, n_beams_within_tresh_list, avg_power_list

    def _save_metrics_result(self,
                             results_list: List[Dict[str, Any]],
                             func_mode: Literal['val', 'test'],
                             is_finetune: bool = False):


        df = pd.DataFrame(results_list)
        df = pd.concat([df.drop(columns=['metrics', 'inference_results'], axis=1),
                        df['metrics'].apply(pd.Series)], axis=1)
        if self.is_use_combined_dataset:
            scenario_num_str = '_'.join(map(str, self.scenario_num_list))
        else:
            scenario_num_str = self.data_config.scenario_num
        if is_finetune:
            fname = f'{self.model_measurement_dir}/finetune_{func_mode}_metrics_result_scenario_{scenario_num_str}.csv'
        else:
            fname = f'{self.model_measurement_dir}/{func_mode}_metrics_result_scenario_{scenario_num_str}.csv'
        df.to_csv(fname, index=False)

    def _save_best_model(self,
                         top1_acc_percent: float,
                         best_top1_acc_percent: float,
                         top1_key: str,
                         epoch: int,
                         is_finetune: bool = False):
        if top1_acc_percent > best_top1_acc_percent:
            if is_finetune:
                self.best_model_fname = (
                            f"{self.checkpoint_dir}/"
                            f"{self.name_str}"
                            f"_finetune"
                            f"_epoch_{epoch}.pth"
                            )
            torch.save(self.model.state_dict(), self.best_model_fname)
            # logger.debug(f'Saving best model to {self.best_model_fname}')
            best_top1_acc_percent = top1_acc_percent
        logger.debug(f"\nEPOCH {epoch+1} |updated best {top1_key}: {best_top1_acc_percent} %")
        return best_top1_acc_percent

    def _add_epoch_to_dict(self,
                            any_dict: Dict,
                            epoch: int):
        key = "epoch"
        any_dict.update({key: epoch})
        any_dict = {key: any_dict[key]} | {k: v for k, v in any_dict.items() if k != key}
        return any_dict

    def main_training_loop(self) -> Tuple[str, str]:
        self._setup_data_loaders()
        self._setup_model()
        self._setup_training()

        best_top1_acc_percent = -1
        func_mode = 'val'
        train_res_list = []
        val_res_list = []
        for epoch in range(self.model_config.train_epoch):
            # Reset metric_handler for each epoch
            self.metric_handler.reset()

            train_results = self._train(data_loader=self.train_loader)
            if self.val_loader is not None:
                val_results = self._validate(data_loader=self.val_loader, func_mode=func_mode)
            else:
                val_results = self._validate(data_loader=self.test_loader, func_mode=func_mode)

            train_results = self._add_epoch_to_dict(train_results, epoch)
            val_results = self._add_epoch_to_dict(val_results, epoch)

            train_res_list.append(train_results)
            val_res_list.append(val_results)

            if self.model_config.dl_task_type == 'base_beam_tracking':
                top1_key = f'pred1_top1_{func_mode}_acc_percent'
            else:
                top1_key = f'pred0_top1_{func_mode}_acc_percent'
            top1_acc_percent = val_results['metrics'][top1_key]
            best_top1_acc_percent = self._save_best_model(top1_acc_percent,
                                                          best_top1_acc_percent,
                                                          top1_key,
                                                          epoch)


            if self.model_config.optimizer_name == 'adam':
                self.scheduler.step()

            val_avg_loss = val_results['avg_loss']
            if self.__should_stop_early(early_stopping_obj=self.early_stopping,
                                        val_avg_loss=val_avg_loss,
                                        epoch=epoch):
                break

        train_epoch_df = self._save_epoch_vs_loss(results_list=train_res_list, func_mode='train')
        val_epoch_df = self._save_epoch_vs_loss(results_list=val_res_list, func_mode=func_mode)
        # merge df on column epoch
        epoch_df = train_epoch_df.merge(val_epoch_df, on='epoch')
        epoch_df['train_entropy'] = self.train_entropy
        dataplot_obj = DataPlot()
        if self.is_use_combined_dataset:
            scenario_num_str = '_'.join(map(str, self.scenario_num_list))
        else:
            scenario_num_str = self.data_config.scenario_num
        dataplot_obj.plot_label_distribution(dataset_dict=self.dataset_dict,
                                    folder_name=self.model_measurement_dir,
                                    label_column_name=self.model_config.label_column_name,
                                    scenario_num_str=scenario_num_str,
                                    splitting_method=self.data_config.splitting_method)
        dataplot_obj.plot_loss_vs_epochs(df=epoch_df,
                            folder_name=self.model_measurement_dir)
        epoch_df.to_csv(f'{self.model_measurement_dir}/train_val_epoch_vs_loss.csv', index=False)
        self._save_metrics_result(results_list=val_res_list, func_mode=func_mode)

    def _setup_early_stopping(self):
        if self.model_config.use_early_stopping:
            early_stopping = EarlyStopping(
                                    patience=self.model_config.early_stop_patience,
                                    min_delta=self.model_config.early_stop_min_delta
                                    )
            return early_stopping
        else:
            return None

    def __should_stop_early(self,
                            early_stopping_obj: Any,
                            val_avg_loss: float,
                            epoch: int):
        if self.model_config.use_early_stopping:
            early_stopping_obj(val_loss=val_avg_loss)
            if early_stopping_obj.early_stop:
                logger.info(f"""Early stopping at epoch: {epoch + 1} |
                                Val Avg loss: {val_avg_loss}""")
                return True
        return False
    
    def get_dense_model_info(self,
                             test_results: Dict[str, Any]):
        if self.model_config.dl_task_type == 'base_beam_tracking':
            dense_model_accuracy = test_results['metrics']['pred1_top1_test_acc_percent']
        else:
            dense_model_accuracy = test_results['metrics']['pred0_top1_test_acc_percent']
        dense_model_accuracy = test_results['metrics']['pred0_top1_test_acc_percent']
        model_info = self._get_model_info()
        logger.info(f"Dense model accuracy              : {dense_model_accuracy:.2f}%")
        logger.info(f"Dense Model MACs                  : {model_info['model_macs']}")
        logger.info(f"Dense Model number of parameters  : {model_info['model_num_param']}")
        logger.info(f"Dense Model size                  : {model_info['model_size_MiB']} MiB")
    