import argparse
from math import ceil
from typing import Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset
from data.dataset import GanBrats, GanBratsFromFolder, MrSliceDataset
from data.transforms import get_transforms

from learning.lightning_module import BratsSegmentationModule, PolyLrCallback
from learning.models import UNet
from learning.utils import get_last_checkpoint, preload_if_possible, worker_init_fn
import config

torch.backends.cudnn.benchmark = False


def get_experiment_name(datasets: Sequence[MrSliceDataset], it):
    dataset_str = '-'.join(dataset.name for dataset in datasets)
    return f'SameExp-{dataset_str}-{it}'


def train_experiment(datasets: Sequence[MrSliceDataset],
                     learning_rate: float,
                     resume: bool = True,
                     padding_mode: str = 'valid',
                     output_size: int = 256,
                     coloraug: bool = False,
                     geoaug: bool = False,
                     maybe_load_to_ram: bool = True,
                     total_number_of_samples: Optional[float] = None,
                     split_using_instances: bool = True,
                     sample_real_more: bool = True,
                     it=0):

    pl.utilities.seed.seed_everything(0)

    num_workers = 16

    batch_size = config.base_batch_size
    if padding_mode == 'same':
        batch_size = round(2.5 * batch_size)
        if output_size == 128:
            batch_size = round(2*batch_size)
    elif padding_mode == 'valid':
        if output_size == 128:
            batch_size = round(1.2*batch_size)
    else:
        raise NotImplementedError(f'padding mode {padding_mode} not implemented')

    experiment_name = get_experiment_name(datasets, it=it)

    experiment_path = config.models / experiment_name

    model = UNet(output_size=(output_size, output_size), in_channels=4, n_classes=4, padding=padding_mode)

    train_transforms, val_transforms, _ = get_transforms(
        model.output_size, model.input_size, coloraug=coloraug, geoaug=geoaug)

    # split all datasets in train and val
    train_sets = []
    val_sets = []
    add_val_macro_dice_mean_checkpointer = False
    for dataset in datasets:
        if isinstance(dataset, GanBrats) or isinstance(dataset, GanBratsFromFolder):
            instance_level = False
            realdata = False
        else:
            instance_level = split_using_instances
            add_val_macro_dice_mean_checkpointer = split_using_instances
            realdata = True

        train, val = dataset.split(training_fraction=0.8, split_on_subject_level=instance_level)
        val.return_name = True

        if sample_real_more and realdata:
            train.repeat(n_repeats=ceil(80000 / len(train)))
            val.repeat(n_repeats=ceil(20000 / len(val)))

        train.set_transforms(*train_transforms)
        val.set_transforms(*val_transforms)
        train_sets.append(train)
        val_sets.append(val)

    train = ConcatDataset(train_sets)
    val = ConcatDataset(val_sets)

    if total_number_of_samples is None:
        n_epochs = 1000
    else:
        n_epochs = round(total_number_of_samples / len(train))

    if maybe_load_to_ram:
        preload_if_possible(train)
        preload_if_possible(val)

    module = BratsSegmentationModule(model=model, initial_learning_rate=learning_rate)

    train_loader = DataLoader(train,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(),
                              worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(),
                            worker_init_fn=worker_init_fn)

    if not resume and experiment_path.exists():
        print(f'{experiment_name} already exists and resume is False, skipping')
        return
    experiment_path.mkdir(exist_ok=True)

    previous_checkpoint = get_last_checkpoint(experiment_path) if resume else None

    checkpoints = [ModelCheckpoint(monitor='val_mean_dice', filename='{epoch}-{val_mean_dice:.10f}',
                                   save_last=True, save_top_k=1, mode='max'),
                   LearningRateMonitor(logging_interval='epoch'),
                   PolyLrCallback()]
    if add_val_macro_dice_mean_checkpointer:
        checkpoints.append(ModelCheckpoint(monitor='val_macro_dice_mean', filename='{epoch}-{val_macro_dice_mean:.10f}',
                                           save_last=False, save_top_k=1, mode='max'))

    trainer = pl.Trainer(gpus=1,
                         max_epochs=n_epochs,
                         default_root_dir=experiment_path,
                         callbacks=checkpoints,
                         resume_from_checkpoint=previous_checkpoint,
                         precision=16,
                         amp_backend="native",
                         gradient_clip_val=12,
                         gradient_clip_algorithm="norm")

    do_training = True
    if previous_checkpoint is not None:
        checkpoint = torch.load(previous_checkpoint)
        if checkpoint['epoch'] == n_epochs:
            print('Model already finished training')
            do_training = False

    print(f'Starting training for {experiment_name}')
    if do_training:
        trainer.fit(module, train_loader, val_loader)


if __name__ == '__main__':
    split_using_instances = True

    parser = argparse.ArgumentParser(description='Train BRATS 2D segmentations model.')
    parser.add_argument('it', type=int, nargs=1)

    args = parser.parse_args()

    datasets = []
    datasets.append(GanBratsFromFolder('synthetic_1GAN',
                                       pair_transforms=None,
                                       image_transforms=None,
                                       target_transforms=None))

    train_experiment(datasets,
                     learning_rate=1e-2,
                     resume=False,
                     padding_mode='valid',
                     coloraug=True,
                     geoaug=True,
                     total_number_of_samples=3.2e7,
                     split_using_instances=split_using_instances,
                     maybe_load_to_ram=False,
                     it=args.it[0])
