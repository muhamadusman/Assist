from data.dataset import Brats184, Brats2021Train, Brats313, GanBratsFromFolder
from learning.train_from_folder_name import train_experiment


if __name__ == '__main__':
    split_using_instances = True

    base_dataset = 'Brats2021'
    include_original = True
    include_synthetic = False
    single_channel = True
    augmentation = 'no'
    full_annotations = False
    stylegan = True
    padding_mode = 'valid'
    nopixelnorm = False
    coloraug = True
    geoaug = True

    datasets = []
    if include_original:
        if base_dataset == 'Brats184':
            datasets.append(Brats184(None, None, None))
        if base_dataset == 'Brats313':
            datasets.append(Brats313(None, None, None))
        if base_dataset == 'Brats2021':
            datasets.append(Brats2021Train(None, None, None))

    if include_synthetic:
        datasets.append(GanBratsFromFolder('Sample_StyleGAN1',
                                           pair_transforms=None,
                                           image_transforms=None,
                                           target_transforms=None))

    train_experiment(datasets,
                     learning_rate=2e-2,
                     resume=True,
                     padding_mode=padding_mode,
                     coloraug=coloraug,
                     geoaug=geoaug,
                     maybe_load_to_ram=False,
                     total_number_of_samples=3.2e5,
                     split_using_instances=split_using_instances,
                     seed=0)
