import torch
import numpy as np
from typing import Any, Dict, List
from loguru import logger

class MetricHandler:
    def __init__(self, exp_config: Any, model_config: Any):
        self.exp_config = exp_config
        self.model_config = model_config
        self.accumulated_metrics = {}
        self.all_outputs = []
        self.all_labels = []
        self.total = np.zeros((self.model_config.pred_num,))

    def reset(self):
        self.accumulated_metrics = {}
        self.all_outputs = []
        self.all_labels = []
        self.total = np.zeros((self.model_config.pred_num,))

    def update_metrics(self, 
                       outputs: torch.Tensor, 
                       labels: torch.Tensor, 
                       func_mode: str):
        self.all_outputs.append(outputs)
        self.all_labels.append(labels)
        self.total += torch.sum(labels != -100, dim=-1).cpu().numpy()

    def update_metrics_batch_first(self, 
                               outputs: torch.Tensor, 
                               labels: torch.Tensor, 
                               func_mode: str):
        self.all_outputs.append(outputs)
        self.all_labels.append(labels)
        self.total += torch.sum(labels != -100, dim=0).cpu().numpy()

    def calculate_final_metrics(self, func_mode: str) -> Dict[str, float]:
        all_outputs = torch.cat(self.all_outputs, dim=1)
        all_labels = torch.cat(self.all_labels, dim=1)
        
        _, idx = torch.topk(all_outputs, max(self.exp_config.top_n_list), dim=-1)
        idx = idx.cpu().numpy()
        label = all_labels.cpu().numpy()
        
        top_k_correct = {k: np.zeros((self.model_config.pred_num,)) for k in self.exp_config.top_n_list}
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                for k in self.exp_config.top_n_list:
                    top_k_correct[k][i] += np.isin(label[i, j], idx[i, j, :k]).sum()
        
        val_acc = {f'top{k}': np.round(top_k_correct[k] / self.total, 3) for k in self.exp_config.top_n_list}
        
        logger.debug("\n"+str(np.stack([val_acc[f'top{k}'] for k in self.exp_config.top_n_list], 0)), flush=True)

        metrics = {}
        for n in range(self.model_config.pred_num):
            for k in self.exp_config.top_n_list:
                if self.model_config.dl_task_type == 'base_beam_tracking':
                    key = f'pred{n+1}_top{k}_{func_mode}_acc_percent'
                else:
                    key = f'pred{n}_top{k}_{func_mode}_acc_percent'
                metrics[key] = round(val_acc[f'top{k}'][n] * 100, 3)  # Convert to percentage and round to 3 decimal places

        return metrics

    def calculate_final_metrics_batch_first(self, func_mode: str, verbose: bool = True) -> Dict[str, float]:
        all_outputs = torch.cat(self.all_outputs, dim=0)  # [batch_size, seq_len, num_classes]
        all_labels = torch.cat(self.all_labels, dim=0)    # [batch_size, seq_len]
        
        _, idx = torch.topk(all_outputs, max(self.exp_config.top_n_list), dim=-1)
        idx = idx.cpu().numpy()
        label = all_labels.cpu().numpy()
        
        top_k_correct = {k: np.zeros((self.model_config.pred_num,)) for k in self.exp_config.top_n_list}
        for k in self.exp_config.top_n_list:
            top_k_correct[k] = np.sum(np.any(idx[:, :, :k] == label[:, :, None], axis=-1), axis=0)
        
        val_acc = {f'top{k}': np.round(top_k_correct[k] / self.total, 3) for k in self.exp_config.top_n_list}
        
        if verbose:
            logger.debug("\n"+str(np.stack([val_acc[f'top{k}'] for k in self.exp_config.top_n_list], 0)), flush=True)

        metrics = {}
        for n in range(self.model_config.pred_num):
            for k in self.exp_config.top_n_list:
                if self.model_config.dl_task_type == 'base_beam_tracking':
                    key = f'pred{n+1}_top{k}_{func_mode}_acc_percent'
                else:
                    key = f'pred{n}_top{k}_{func_mode}_acc_percent'
                metrics[key] = round(val_acc[f'top{k}'][n] * 100, 3)  # Convert to percentage and round to 3 decimal places

        return metrics