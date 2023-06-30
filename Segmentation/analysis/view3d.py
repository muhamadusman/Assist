import numpy as np
from bone_segmentation.DataStructures import ImageToPatientTransformation
from bone_segmentation.Viewer import ImageViewer

import config
from data.dataset import BratsTest

dataset_prefix = 'test'
dataset = BratsTest(None, None, None)
experiment_name = 'UNet-20220221-Brats184-256-geoaug-imaug-valid-lr1.0e-02'
# experiment_name = 'nnUNet184'

prediction_folder = config.results / 'model_evaluation' / experiment_name / f'{dataset_prefix}_predictions'

subjects_to_view = {'Subject_285'}

for subject in dataset.get_available_subjects():
    if subject not in subjects_to_view:
        continue
    input_volume, target_volume = None, None
    subject_indices = dataset.get_indices_for_subject(subject)
    for count, dataset_index in enumerate(subject_indices):
        input, target = dataset[dataset_index]

        if input_volume is None:
            target_volume = np.zeros((len(subject_indices),) + target.shape, dtype=np.uint8)
            input_volume = np.zeros((len(subject_indices),) + input.shape, dtype=np.uint8)
        target_volume[count] = target
        input_volume[count] = input

    prediction_volume = np.load(prediction_folder / f'{subject}.npy')

    image_ranges = {mod: (0, 256) for mod in ('T1', 'T1CE', 'T2', 'FLAIR')}

    compare = np.zeros_like(target_volume)
    compare[np.logical_and(target_volume != 0, prediction_volume != 0)] = 1  # tp
    compare[np.logical_and(target_volume == 0, prediction_volume != 0)] = 2  # fp
    compare[np.logical_and(target_volume != 0, prediction_volume == 0)] = 3  # fn

    v = ImageViewer({'T1': input_volume[..., 0],
                     'T1CE': input_volume[..., 1],
                     'T2': input_volume[..., 2],
                     'FLAIR': input_volume[..., 3]},
                    image_ranges=image_ranges,
                    masks={'pred': prediction_volume, 'target': target_volume, 'compare': compare},
                    transformation=ImageToPatientTransformation.identity(),
                    title=subject)
    v.show()
