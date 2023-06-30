import argparse
from math import ceil
from typing import Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader

import config

from data.dataset import Brats184, Brats313, Brats2021Train, GanBrats, GanBratsFromFolder, MrSliceDataset
from data.transforms import get_transforms
from learning.lightning_module import BratsSegmentationModule, PolyLrCallback
from learning.models import UNet
from learning.utils import get_last_checkpoint, preload_if_possible


def get_experiment_name(datasets: Sequence[MrSliceDataset],
                        learning_rate: float,
                        padding_mode: str = 'valid',
                        output_size: int = 256,
                        coloraug: bool = False,
                        geoaug: bool = False,
                        split_using_instances: bool = True,
                        seed: int = 0):
    augstr = '-geoaug' if geoaug else ''
    if coloraug:
        augstr += '-imaug'
    if not split_using_instances:
        augstr += '-fullrandomsplit'
    dataset_str = '-'.join(dataset.name for dataset in datasets)
    return f'UNet-202209-{dataset_str}-{output_size}{augstr}-{padding_mode}-lr{learning_rate:.1e}-{seed}'


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
                     seed: int = 0):

    pl.utilities.seed.seed_everything(seed, workers=True)

    num_workers = 16

    batch_size = config.base_batch_size
    if padding_mode == 'same':
        batch_size = round(2.5 * batch_size)
        if output_size == 128:
            batch_size = round(2 * batch_size)
    elif padding_mode == 'valid':
        if output_size == 128:
            batch_size = round(1.2 * batch_size)
    else:
        raise NotImplementedError(f'padding mode {padding_mode} not implemented')

    experiment_name = get_experiment_name(datasets,
                                          learning_rate,
                                          padding_mode=padding_mode,
                                          output_size=output_size,
                                          coloraug=coloraug,
                                          geoaug=geoaug,
                                          split_using_instances=split_using_instances,
                                          seed=seed)

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
                              worker_init_fn=pl.utilities.seed.pl_worker_init_function)

    val_loader = DataLoader(val,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(),
                            worker_init_fn=pl.utilities.seed.pl_worker_init_function)

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
    parser.add_argument('seed', type=int, nargs=1, help='random seed')
    parser.add_argument('base_dataset', type=str, nargs=1, choices=(
        'Brats184', 'Brats313', 'Brats2021'), help='Brats184, Brats313, Brats2021')
    parser.add_argument('padding_mode', type=str, nargs=1, choices=(
        'valid', 'same'), help='valid or same, padding mode used i model')
    parser.add_argument('synthetic_folder_name', type=str, nargs=1)
    parser.add_argument('--include-original', dest='include_original', action='store_const', const=True, default=False,
                        help='If original dataset is to be included in training')
    parser.add_argument('--geoaug', dest='geoaug', action='store_const', const=True, default=False,
                        help='If geometric augmentation should be applied')
    parser.add_argument('--coloraug', dest='coloraug', action='store_const', const=True, default=False,
                        help='If color augmentation should be applied')

    args = parser.parse_args()

    datasets = []
    if args.include_original:
        print(args.base_dataset[0])
        if args.base_dataset[0] == 'Brats184':
            datasets.append(Brats184(None, None, None))
        elif args.base_dataset[0] == 'Brats313':
            datasets.append(Brats313(None, None, None))
        elif args.base_dataset[0] == 'Brats2021':
            datasets.append(Brats2021Train(None, None, None))

    if args.synthetic_folder_name[0] != 'None':
        datasets.append(GanBratsFromFolder(args.synthetic_folder_name[0],
                                           pair_transforms=None,
                                           image_transforms=None,
                                           target_transforms=None))

    train_experiment(datasets,
                     learning_rate=2e-2,
                     resume=False,
                     padding_mode=args.padding_mode[0],
                     coloraug=args.coloraug,
                     geoaug=args.geoaug,
                     total_number_of_samples=3.2e7,
                     split_using_instances=split_using_instances,
                     maybe_load_to_ram=False,
                     seed=args.seed[0])
