import argparse

from data.dataset import GanBrats, get_brats_diff
from learning.train import train_experiment


if __name__ == '__main__':
    split_using_instances = False

    parser = argparse.ArgumentParser(description='Train BRATS 2D segmentations model.')
    parser.add_argument('fraction', type=float, nargs=1, help='fraction of Bratsdiff to include, in (0, 1]')
    parser.add_argument('--include-synthetic', dest='include_synthetic', action='store_const', const=True,
                        default=False, help='If synthetic dataset is to be included in training')

    args = parser.parse_args()

    datasets = [get_brats_diff(args.fraction[0])]

    if args.include_synthetic:
        datasets.append(GanBrats(num_images=184,
                                 single_channel=True,
                                 augmentation='no',
                                 full_annotations=False,
                                 no_pixel_norm=False,
                                 stylegan=False,
                                 pair_transforms=None,
                                 image_transforms=None,
                                 target_transforms=None))

    train_experiment(datasets,
                     learning_rate=2e-2,
                     resume=False,
                     padding_mode='valid',
                     coloraug=True,
                     geoaug=True,
                     total_number_of_samples=3.2e7,
                     split_using_instances=split_using_instances,
                     maybe_load_to_ram=False)
