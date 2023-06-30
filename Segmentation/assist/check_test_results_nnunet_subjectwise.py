from pathlib import Path

import nibabel as nib
import numpy as np

from data.dataset import BratsTest, label_mapping
from learning.metrics import pixelwise_evaluation_metrics
from learning.utils import plot_compare_image, write_results_to_xlsx
from matplotlib import pyplot as plt
import config

nnunet_model_path = Path('/home/mans/Eigenvision/Projects/ASSIST/nn-unet-models')
src_path = nnunet_model_path / 'nnUNet/2d/Task501_Brats184/nnUNetTrainerV2__nnUNetPlansv2.1/test_results_raw'
experiment_name = 'nnUNet184'
dataset_prefix = 'test'
dataset = BratsTest(None, None, None)

save_folder = config.results / 'model_evaluation' / experiment_name
save_folder.mkdir(exist_ok=True)
image_save_folder = save_folder / f'{dataset_prefix}_images'
image_save_folder.mkdir(exist_ok=True)
prediction_save_folder = save_folder / f'{dataset_prefix}_predictions'
prediction_save_folder.mkdir(exist_ok=True)
result_save_path = save_folder / f'{dataset_prefix}_result.xlsx'


res_files = list(src_path.glob('*.nii.gz'))
results = dict()
fig_handle = plt.figure(figsize=(19.20, 10.80))

dataset.return_name = True
for subject in dataset.get_available_subjects():
    prediction_volume, target_volume = None, None
    subject_indices = dataset.get_indices_for_subject(subject)
    for count, dataset_index in enumerate(subject_indices):

        name, image, target = dataset[dataset_index]
        for file_path in res_files:
            if name[:-3] in str(file_path):
                res_file = file_path
                break
        else:
            raise RuntimeError(f'Could not find {name[:-3]}')

        name = res_file.stem[:-4]
        img = nib.load(res_file)
        predicted_classes = np.transpose(img.get_fdata()[..., 0].astype(np.uint8))

        if prediction_volume is None:
            prediction_volume = np.zeros((len(subject_indices),) + predicted_classes.shape, dtype=np.uint8)
            target_volume = np.zeros((len(subject_indices),) + target.shape, dtype=np.uint8)
        prediction_volume[count] = predicted_classes
        target_volume[count] = target

        fig_handle.clear()
        plot_compare_image(fig_handle, image[..., 0], target, predicted_classes)
        fig_handle.savefig(image_save_folder / f'{name}.png', bbox_inches='tight')

    np.save(prediction_save_folder / f'{subject}.npy', prediction_volume)
    results[subject] = pixelwise_evaluation_metrics(
        target_volume, prediction_volume, classes=(1, 2, 3), include_foreground_metrics=True)

write_results_to_xlsx(results=results, save_path=result_save_path,
                      labels=label_mapping, include_foreground_metrics=True)
