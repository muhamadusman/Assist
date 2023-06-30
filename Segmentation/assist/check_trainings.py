import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import config


def get_last_checkpoint(experiment_path: Path) -> Optional[Path]:
    checkpoints_root_dir = experiment_path / 'lightning_logs'
    if not checkpoints_root_dir.exists():
        return None
    last_checkpoint = None
    for checkpoint_folder in sorted(checkpoints_root_dir.iterdir()):
        checkpoint = checkpoint_folder / 'checkpoints' / 'last.ckpt'
        if checkpoint.exists():
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


strings_to_print = []
for model_path in config.models.iterdir():
    last_path = get_last_checkpoint(model_path)
    dice, _, _ = get_best_checkpoint(model_path, 'val_mean_dice')
    macro, _, _ = get_best_checkpoint(model_path, 'val_macro_dice_mean')

    if last_path is None:
        timestamp_str = 'no last.cpkt found                               '
    else:
        timestamp_str = datetime.fromtimestamp(last_path.stat().st_mtime).strftime('%Y%m%d-%H:%M')
        elapsed_time = (last_path.stat().st_mtime - os.stat(model_path).st_ctime) / 3600
        timestamp_str = f'updated {timestamp_str}, trained for {elapsed_time:.1f} hours   '

    extra_str = ''
    if dice is not None:
        extra_str += 'dice '
    if macro is not None:
        extra_str += 'macro'
    strings_to_print.append(f'{model_path.name:100} {timestamp_str:25} {extra_str}')

for string_to_print in sorted(strings_to_print):
    print(string_to_print)
