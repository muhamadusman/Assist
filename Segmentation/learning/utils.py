import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from data.dataset import MrSliceDataset
from data.utils import plot_img_and_target
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset


def tensor_center_crop(tensor, target_size):
    if tensor.ndim == 4:
        _, _, tensor_height, tensor_width = tensor.size()
    elif tensor.ndim == 3:
        _, tensor_height, tensor_width = tensor.size()
    diff_y = (tensor_height - target_size[0]) // 2
    diff_x = (tensor_width - target_size[1]) // 2
    return tensor[..., diff_y:diff_y + target_size[0], diff_x:diff_x + target_size[1]]


def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)


def get_checkpoint(experiment_path: Path, model_key: str) -> Tuple[Optional[Path], Optional[float], Optional[int]]:
    """Helper function to fetch checkpoint

    Args:
        experiment_path: path to trained model directory
        model_key: key of which model to fetch

    Returns:
        Optional[Path]: If found, returns model checkpoint path, else None
        Optional[float]: If found, returns value of metric for best checkpoint, else None
        Optional[int]: If found, returns epoch for best checkpoint, else None
    """
    assert model_key in {'last', 'val_mean_dice', 'val_macro_dice_mean'}

    if model_key == 'last':
        return get_last_checkpoint(experiment_path=experiment_path), None, None
    else:
        return get_best_checkpoint(experiment_path=experiment_path, metric_name=model_key, mode='max')


def get_last_checkpoint(experiment_path: Path) -> Optional[Path]:
    checkpoints_root_dir = experiment_path / 'lightning_logs'
    if not checkpoints_root_dir.exists():
        return None

    last_checkpoint = None
    latest_datetime = datetime(1, 1, 1, 0, 0)

    for checkpoint_folder in sorted(checkpoints_root_dir.iterdir()):
        checkpoint = checkpoint_folder / 'checkpoints' / 'last.ckpt'

        if checkpoint.exists():
            checkpoint_change_timestamp = datetime.fromtimestamp(checkpoint.stat().st_mtime)
            if checkpoint_change_timestamp > latest_datetime:
                latest_datetime = checkpoint_change_timestamp
                last_checkpoint = checkpoint
    return last_checkpoint


def get_best_checkpoint(experiment_path: Path,
                        metric_name: str = 'val_mean_dice',
                        mode: Optional[str] = None) -> Tuple[Optional[Path], Optional[float], Optional[int]]:
    """Locates checkpoint with best metric in experiment path

    Args:
        experiment_path: path to trained model folder
        metric_name: name of metric used for checkpointing during training
        mode: if metrich shoule be minimized or maximized. If None, this is decided automatically

    Returns:
        Tuple[Optional[Path], Optional[float], Optional[int]]: _description_
    """
    if mode is None:
        mode = 'min' if 'loss' in metric_name else 'max'

    assert mode in {'min', 'max'}
    current_best = 1e10
    sign = 1 if mode == 'min' else -1

    pattern = re.compile(fr'epoch=(\d+)-{metric_name}=([\d\.]+).ckpt')

    checkpoints_root_dir = experiment_path / 'lightning_logs'
    if not checkpoints_root_dir.exists():
        return None, None, None

    best_checkpoint = None
    best_epoch = None
    for path_object in checkpoints_root_dir.glob('**/*'):
        if path_object.is_file():
            match = re.match(pattern, path_object.name)
            if match is not None:
                this_score = sign*float(match.groups()[1])
                if this_score < current_best:
                    current_best = this_score
                    best_epoch = int(match.groups()[0])
                    best_checkpoint = path_object

    return best_checkpoint, sign*current_best, best_epoch


def write_results_to_xlsx(results, save_path, labels, include_foreground_metrics: bool):
    if include_foreground_metrics:
        labels[0] = 'foreground'
        labels[-1] = 'mean'

    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        for number, name in labels.items():
            lines = list()
            for subject_name, data in results.items():
                line = pd.DataFrame(index=[subject_name])

                for metric_name, metrics in data.items():
                    if number in metrics:
                        line[metric_name] = metrics[number]
                    else:
                        line[metric_name] = 'NaN'

                lines.append(line)

            pd_data = pd.concat(lines).sort_index()
            pd_data.to_excel(writer, sheet_name=name[:31])


def plot_compare_image(fig_handle, image, true_classes, predicted_classes):
    gs = fig_handle.add_gridspec(1, 3)
    ax = fig_handle.add_subplot(gs[0, 0])
    plot_img_and_target(image, predicted_classes)
    plt.axis('off')
    ax.set_title('prediction')
    ax = fig_handle.add_subplot(gs[0, 1], sharex=ax, sharey=ax)
    plot_img_and_target(image, true_classes)
    plt.axis('off')
    ax.set_title('true')

    compare = np.zeros_like(true_classes)
    compare[np.logical_and(true_classes != 0, predicted_classes != 0)] = 1  # tp
    compare[np.logical_and(true_classes == 0, predicted_classes != 0)] = 2  # fp
    compare[np.logical_and(true_classes != 0, predicted_classes == 0)] = 3  # fn

    ax = fig_handle.add_subplot(gs[0, 2], sharex=ax, sharey=ax)
    plot_img_and_target(image, compare)
    plt.axis('off')
    ax.set_title('compare')


def preload_if_possible(dataset: Union[ConcatDataset, MrSliceDataset], relative_margin: float = 0.9):
    """Checks available RAM and preloads all images if possible total size of dataset is less than 
       relative_margin*available ram.
    """
    available_ram = psutil.virtual_memory().available / 1024 ** 3

    if isinstance(dataset, ConcatDataset):
        estimated_size = 0
        for dset in dataset.datasets:
            estimated_size += dset.estimated_size
    else:
        estimated_size = dataset.estimated_size

    if estimated_size < relative_margin * available_ram:
        print('Loading dataset into memory')
        if isinstance(dataset, ConcatDataset):
            for dset in dataset.datasets:
                dset.preload()
        else:
            dataset.preload()
