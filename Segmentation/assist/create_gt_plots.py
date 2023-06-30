from tqdm import tqdm
from data.dataset import BratsTest

from matplotlib import pyplot as plt

from data.utils import plot_img_and_target
dataset = BratsTest(None, None, None)
dataset.return_name = True

image_save_folder = dataset.root.parent / 'plots'
image_save_folder.mkdir()

fig_handle = plt.figure(figsize=(19.20, 10.80))
for subject in tqdm(dataset.get_available_subjects()):
    prediction_volume, target_volume = None, None
    subject_indices = dataset.get_indices_for_subject(subject)
    names, ims, targets = [], [], []
    for dataset_index in subject_indices:
        name, im, target = dataset[dataset_index]

        fig_handle.clear()
        plot_img_and_target(im[..., 0], target)
        plt.axis('off')
        fig_handle.savefig(image_save_folder / f'{name}.png', bbox_inches='tight')
plt.close()
