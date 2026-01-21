"""
Datalogger class for NN training
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Literal, Dict, Tuple, List
import os
import h5py

class DataLogger:
    """
    Logger module to track model statistics:

    Args:
        metrics (Dict[str, Dict[str, list]]): metrics file (e.g. from previous training points)
        class_names (List[str]): List of class names in case of multiclass-segmentation

    """
    def __init__(self, metrics: Dict[str, Dict[str, list]] = None, class_names: List[str] = None):
        if class_names:
            self.num_classes = len(class_names)
            self.class_names = class_names
        else:
            self.num_classes = 1
            self.class_names = None
        if metrics:
            self.metrics = metrics
        else:
            self.metrics = {
                phase: self._init_phase_metrics()
                for phase in ['Train', 'Val']
            }
    
    def _init_phase_metrics(self):
        metrics_dict = {
            "loss": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "iou": [],
        }

        return metrics_dict

    def _get_metrics_from_logits(self,
        conf_metrics: Tuple,
        loss: float = 'NaN',
    ) -> dict:
        """
        Compute the metrics for the predictions and annotations of raw probability predictions and binary
        annotations.
        """

        tp, fp, fn, tn = conf_metrics
        
        eps = 1e-12 # avoid division by 0

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        loss = np.array(loss)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "loss": loss
            }

        return metrics

    def log(self, 
            stats: Tuple[np.ndarray, np.ndarray],
            phase: Literal['Train', 'Val'] = 'Train') -> None:
        """
        Log the metrics into the logger.

        Args:
            stats (Tuple[np.ndarray, np.ndarray]): Tuple containing (confusion_metrics, loss_value) 
            phase (Literal['Train', 'Val']): indicates whether training or validation stats are being logged
        """
        metrics = self._get_metrics_from_logits(stats[0], stats[1])
        if phase not in self.metrics:
            raise KeyError(f"Phase '{phase}' not found in logger. Please use 'Train' or 'Val'.")
        
        for key, value in metrics.items():
            if key in self.metrics[phase]:
                self.metrics[phase][key].append(value)
            else:
                raise KeyError(f"Metric '{key}' not found in logger. Please check the metric names.")


    def reset(self) -> None:
        """
        Reset the logger.
        """
        for phase in self.metrics:
            for key in self.metrics[phase]:
                self.metrics[phase][key] = []
    
    def save(self, filepath: str) -> None:
        """
        Save the metrics as a h5 file, nested in training and validations stats.

        Args:
            filepath (str): The path to save the metrics to.

        """
        with h5py.File(filepath, 'w') as f:
            for split, metric_dict in self.metrics.items():  # Iterate over "Train" and "Val"
                group = f.create_group(split)  # Create a group for each split
                for key, value in metric_dict.items():  # Iterate over metrics in each split
                    group.create_dataset(key, data=value)

    def load(self, filepath: str) -> None:
        """Load dataset from a h5 file"""
        metrics = {}
        with h5py.File(filepath, 'r') as f:
            for split in f.keys():
                metrics[split] = {} 
                for key in f[split].keys():  # Iterate over stored metric keys
                    metrics[split][key] = f[split][key][()]  # Load dataset values


    def print_last_metrics(self) -> None:
        """
        Prints the last metrics that were logged (last epoch)
        """
        epoch = len(self.metrics['Train']['loss'])
        print(
            f"Epoch {epoch} | " \
            f"train loss {self.metrics['Train']['loss'][-1]:.3f} | " \
            f"val loss {self.metrics['Val']['loss'][-1]:.3f} | " \
            f"val F1 {self.metrics['Val']['f1'][-1].mean().item():.3f}"
        )
        if self.class_names:
            for i, name in enumerate(self.class_names):
                print(
                    f"{name} Train F1: {self.metrics['Train']['f1'][-1][i].item():.3f} | " \
                    f"Val F1: {self.metrics['Val']['f1'][-1][i].item():.3f}"
                )


    def get_last_F1(self, phase: Literal['Train', 'Val']) -> float:
        """Returns the last training or validation F1 score"""
        val_f1_macro = [t.mean().item() for t in self.metrics[phase]['f1']]
        return val_f1_macro[-1]


    def get_metrics(self, phase: Literal['Train', 'Val'] = None) -> dict:
        """Returns all training or validation metrics"""
        if phase: 
            return self.metrics[phase]
        else:
            return self.metrics
    
    def plot_metrics(self, model_name: str = None) -> None:
        """
        Creates plots of the training statistics and saves them as pdf. 
        Includes (as functions of epoch):
            Train and Val F1 (macro)
            Train and Val Precision (macro)
            Train and Val Recall (macro)
            Train and Val IoU (macro)
            Train and Val loss
            Train and Val metrics as per-class plot (in case of multiclass training)
        """

        plt.style.use('ggplot')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 11.5,
            'ytick.labelsize': 11.5,
            'lines.linewidth': 2,
            'grid.alpha': 0.5,
            'grid.color': '#666666',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor' : 'black',
            'axes.prop_cycle': mpl.cycler(color=[
                '#1f77b4',
                '#ff7f0e', 
                '#9467bd',
                '#2ca02c',
                '#7f7f7f'])
        })
        current_dir = os.path.dirname(__file__)
        path = os.path.abspath(os.path.join(current_dir, '..', 'figures', model_name))
        print('SAVING IMAGES TO:', path)
        os.makedirs(path, exist_ok=True)
        epochs = np.arange(1, len(self.metrics['Train']['f1']) + 1)

        # F1 score (macro)
        train_f1_macro = [t.mean().item() for t in self.metrics['Train']['f1']]
        val_f1_macro = [t.mean().item() for t in self.metrics['Val']['f1']]
        _, ax = plt.subplots()
        ax.plot(epochs, train_f1_macro, label='Train')
        ax.plot(epochs, val_f1_macro, label='Validation')
        ax.set(xlabel='Epoch', ylabel='F1 Score')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path+'/F1_score.pdf', format='pdf', dpi=300, bbox_inches='tight')

        # Macro Precision
        train_p_macro = [t.mean().item() for t in self.metrics['Train']['precision']]
        val_p_macro = [t.mean().item() for t in self.metrics['Val']['precision']]
        _, ax = plt.subplots()
        ax.plot(epochs, train_p_macro, label='Train')
        ax.plot(epochs, val_p_macro, label='Validation')
        ax.set(xlabel='Epoch', ylabel='Precision')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path+'/Precision.pdf', format='pdf', dpi=300, bbox_inches='tight')

        # Macro Recall
        train_r_macro = [t.mean().item() for t in self.metrics['Train']['recall']]
        val_r_macro = [t.mean().item() for t in self.metrics['Val']['recall']]
        _, ax = plt.subplots()
        ax.plot(epochs, train_r_macro, label='Train')
        ax.plot(epochs, val_r_macro, label='Validation')
        ax.set(xlabel='Epoch', ylabel='Recall')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path+'/Recall.pdf', format='pdf', dpi=300, bbox_inches='tight')

        # Macro IoU
        train_iou_macro = [t.mean().item() for t in self.metrics['Train']['iou']]
        val_iou_macro = [t.mean().item() for t in self.metrics['Val']['iou']]
        _, ax = plt.subplots()
        ax.plot(epochs, train_iou_macro, label='Train')
        ax.plot(epochs, val_iou_macro, label='Validation')
        ax.set(xlabel='Epoch', ylabel='IoU')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path+'/IoU.pdf', format='pdf', dpi=300, bbox_inches='tight')

        # Loss
        _, ax = plt.subplots()
        ax.plot(epochs, self.metrics['Train']['loss'], label='Train')
        ax.plot(epochs, self.metrics['Val']['loss'], label='Validation')
        ax.set(xlabel='Epoch', ylabel='Loss')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path+'/Loss.pdf', format='pdf', dpi=300, bbox_inches='tight')

        #per_class_plots
        if self.class_names:
            _, axs = plt.subplots(4, len(self.class_names), figsize=(25, 20), sharex=True, sharey=True)
            for j, classname in enumerate(self.class_names):
                for i, metric in enumerate(['f1', 'precision', 'recall', 'iou']):
                    class_metric_train = [t[j].item() for t in self.metrics['Train'][metric]]
                    class_metric_val = [t[j].item() for t in self.metrics['Val'][metric]]
                    axs[i, j].plot(epochs, class_metric_train, label='Train')
                    axs[i, j].plot(epochs, class_metric_val, label='Val')
                    axs[i, j].set_title(f"{metric} for {classname}")
                    axs[i, j].legend()
            plt.tight_layout()
            plt.savefig(path+'/class_stats.pdf', format='pdf', dpi=300, bbox_inches='tight')