from copy import deepcopy

import config
import numpy as np
from data.dataset import Brats184, Brats313, GanBrats, GanBratsFromFolder, label_mapping
from matplotlib import pyplot as plt

modalities = ('T1', 'T1CE', 'T2', 'FLAIR')
label_colors = {'background': 'k', 'necrotic_non_enhancing_tumor_core': 'b',
                'peritumoral_edema': 'r', 'gd_enhancing_tumor': 'g'}


def statistics_from_dataset(dataset):
    label_volumes = {label_name: [] for label_name in label_mapping.values()}

    label_mapping_with_bg = deepcopy(label_mapping)
    label_mapping_with_bg[0] = 'background'

    label_intensities = dict()
    for modality in modalities:
        label_intensities[modality] = {label_name: np.zeros(
            (256,), dtype=np.int64) for label_name in label_mapping_with_bg.values()}

    for image, pred in dataset:
        for label_index, label_name in label_mapping_with_bg.items():
            label_mask = pred == label_index

            pixel_volume = label_mask.sum()
            if label_name != 'background':
                label_volumes[label_name].append(pixel_volume)

            if pixel_volume == 0:
                continue

            for i, modality in enumerate(modalities):
                if label_name == 'background':
                    this_label_mask = np.logical_and(label_mask, image[..., i] > 0)
                else:
                    this_label_mask = label_mask

                intensities = image[..., i][this_label_mask]
                hist, _ = np.histogram(intensities, bins=np.arange(257))
                label_intensities[modality][label_name] += hist

    return label_volumes, label_intensities


def compare_statistics(num_images: int, synthetic_dataset, save_figs: bool = False, show_figs: bool = True):
    res_dir = config.results / 'dataset_evaluation' / synthetic_dataset.name
    if save_figs and res_dir.exists():
        print(f'result folder for {synthetic_dataset.name} exists, skipping')

    if save_figs:
        res_dir.mkdir(exist_ok=True)
    assert num_images in {184, 313}

    if num_images == 184:
        base_dataset = Brats184(None, None, None)
    else:
        base_dataset = Brats313(None, None, None)

    label_volumes, label_intensities = statistics_from_dataset(base_dataset)
    synthetic_label_volumes, synthetic_label_intensities = statistics_from_dataset(synthetic_dataset)

    for label_name, volumes in label_volumes.items():
        plt.figure()
        counts, bin_edges = np.histogram(volumes, bins=100)
        plotx = bin_edges[:-1] + 0.5*np.diff(bin_edges)

        plotx = plotx[1:]
        counts = counts[1:]/sum(counts)
        plt.plot(plotx, counts, 'b')

        counts, bin_edges = np.histogram(synthetic_label_volumes[label_name], bins=100)
        synth_plotx = bin_edges[:-1] + 0.5*np.diff(bin_edges)

        synth_plotx = synth_plotx[1:]
        counts = counts[1:]/sum(counts)
        plt.plot(synth_plotx, counts, 'b--')

        plt.xlabel('label volume [pixels]')
        plt.title(label_name)

        if save_figs:
            plt.tight_layout()
            plt.savefig(res_dir / f'volumes_{label_name}.png')
        if not show_figs:
            plt.close()

    for modality, intensities in label_intensities.items():
        plt.figure()

        synthetic_intensities = synthetic_label_intensities[modality]
        for label_name, counts in intensities.items():
            plt.plot(counts/counts.sum(), label_colors[label_name], label=label_name)
            plt.plot(synthetic_intensities[label_name] /
                     synthetic_intensities[label_name].sum(), '--' + label_colors[label_name])

        plt.legend()
        plt.xlabel('intensity')
        plt.title(modality)

        y_max = 0
        for label in label_mapping.values():
            norm_true = intensities[label] / intensities[label].sum()
            norm_fake = synthetic_intensities[label] / synthetic_intensities[label].sum()
            this_max = max(norm_true.max(), norm_fake.max())
            if this_max > y_max:
                y_max = this_max
        plt.ylim(-0.001, 1.2*y_max)

        if save_figs:
            plt.tight_layout()
            plt.savefig(res_dir / f'intensities_{modality}.png')
        if not show_figs:
            plt.close()

    if show_figs:
        plt.show()


if __name__ == '__main__':
    dataset = GanBratsFromFolder('styleGAN2_fullAnonation_Gamma2_012600kimg', pair_transforms=None,
                                 image_transforms=None, target_transforms=None)
    compare_statistics(313, dataset, save_figs=True, show_figs=False)

    num_images = 313
    for single_channel in (True, False):
        for augmentation in ('no', 'mirror', 'rotation'):
            for full_annotations in (True, False):
                for nopixelnorm in (True, False):
                    for stylegan in (True, False):
                        try:
                            synthetic_dataset = GanBrats(num_images,
                                                         single_channel=single_channel,
                                                         augmentation=augmentation,
                                                         full_annotations=full_annotations,
                                                         no_pixel_norm=nopixelnorm,
                                                         stylegan=stylegan,
                                                         pair_transforms=None,
                                                         image_transforms=None,
                                                         target_transforms=None,
                                                         must_exist=True)
                            compare_statistics(num_images, synthetic_dataset, save_figs=True, show_figs=False)
                        except RuntimeError as e:
                            print(e)
                            continue
