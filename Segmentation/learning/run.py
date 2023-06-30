from typing import Optional

import config
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from data.dataset import Brats2021Test, MrSliceDataset, label_mapping
from data.transforms import get_transforms
from data.utils import plot_img_and_target
from learning.metrics import pixelwise_evaluation_metrics
from learning.models import UNet
from learning.utils import get_checkpoint, write_results_to_xlsx


def run(experiment_name: str,
        dataset: MrSliceDataset,
        dataset_prefix: str,
        model_choice: str,
        chunk_size: int = 10,
        device: Optional[str] = None,
        overwrite: bool = False,
        skip_image_creation: bool = False):

    experiment_path = config.models / experiment_name
    save_name = experiment_name + f'-{model_choice}'
    save_folder = config.results / 'model_evaluation' / save_name
    image_save_folder = save_folder / f'{dataset_prefix}_images'
    prediction_save_folder = save_folder / f'{dataset_prefix}_predictions'
    result_save_path = save_folder / f'{dataset_prefix}_result.xlsx'

    if result_save_path.exists() and prediction_save_folder.exists() and not overwrite:
        print(f'results already exists for {experiment_name}, skipping')
        return

    model_checkpoint, val_metric, epoch = get_checkpoint(experiment_path=experiment_path, model_key=model_choice)
    if model_checkpoint is None:
        print(f'Unable to find model for {experiment_name} and model choice {model_choice}')
        return
    if val_metric is None:
        print(f'Loading model from {experiment_name}')
    else:
        print(f'Loading model from {experiment_name} with {val_metric} {model_choice} at epoch {epoch}')

    save_folder.mkdir(exist_ok=True)
    image_save_folder.mkdir(exist_ok=True)
    prediction_save_folder.mkdir(exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    padding_mode = experiment_path.stem.split('-')[-2]
    if padding_mode not in {'valid', 'same'}:
        padding_mode = 'valid'
    model = UNet(output_size=(256, 256), in_channels=4, n_classes=4, output_all_levels=False, padding=padding_mode)
    model.load_from_lightning(checkpoint=torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # augmentation settings does not affect validation transforms
    _, val_transforms, _ = get_transforms(model.output_size, model.input_size)
    dataset.set_transforms(*val_transforms)
    dataset.return_name = True

    results = dict()
    fig_handle = plt.figure(figsize=(19.20, 10.80))
    for subject in tqdm(dataset.get_available_subjects()):
        prediction_volume, target_volume = None, None
        subject_indices = dataset.get_indices_for_subject(subject)
        names, ims, targets = [], [], []
        for dataset_index in subject_indices:
            name, im, target = dataset[dataset_index]
            names.append(name)
            ims.append(im)
            targets.append(target)

        input_stack = torch.stack(ims)  # slices are stacked in batch dimension
        target_volume = torch.stack(targets)
        prediction_volume = torch.zeros_like(target_volume)

        for i in range(0, input_stack.shape[0], chunk_size):
            with torch.no_grad():
                output = model(input_stack[i:i + chunk_size].to(device))
            prediction_volume[i:i + chunk_size] = torch.argmax(output, dim=1)

        prediction_volume = prediction_volume.cpu().numpy()

        if not skip_image_creation:
            for name, im, predicted_classes, true_classes in zip(names, ims, prediction_volume, targets):
                image = dataset.by_name(name, apply_transform=False)
                fig_handle.clear()
                # plot_compare_image(fig_handle, image[..., 0], true_classes, predicted_classes)
                plot_img_and_target(image[..., 0], predicted_classes)
                plt.axis('off')
                fig_handle.savefig(image_save_folder / f'{name}.png', bbox_inches='tight')

        np.save(prediction_save_folder / f'{subject}.npy', prediction_volume)
        results[subject] = pixelwise_evaluation_metrics(
            target_volume.cpu().numpy(), prediction_volume, classes=(1, 2, 3), include_foreground_metrics=True)
    plt.close()
    write_results_to_xlsx(results=results, save_path=result_save_path,
                          labels=label_mapping, include_foreground_metrics=True)


if __name__ == '__main__':
    from assist.experiment_names_lookup import get_all_trained_models
    from data.dataset import BratsTest

    overwrite = False
    chunk_size = 10

    expnames = get_all_trained_models()
    expnames = [en for en in expnames if '184' not in en]

    bratstest = None
    bratstest2021 = None
    for experiment_name in expnames:
        if 'fullrandomsplit' in experiment_name or \
                ('Brats184' not in experiment_name and
                 'Brats313' not in experiment_name and
                 'Brats2021Train' not in experiment_name):
            model_choice = 'val_mean_dice'
        else:
            model_choice = 'val_macro_dice_mean'

        if 'BRATS21' in experiment_name or 'Brats2021Train' in experiment_name:
            if bratstest2021 is None:
                bratstest2021 = Brats2021Test(None, None, None)
            dataset = bratstest2021
        else:
            if bratstest is None:
                bratstest = BratsTest(None, None, None)
            dataset = bratstest
        try:
            run(experiment_name,
                dataset,
                dataset_prefix='test',
                model_choice=model_choice,
                chunk_size=chunk_size,
                overwrite=overwrite,
                skip_image_creation=True)
        except Exception as e:
            print(e)
            continue
