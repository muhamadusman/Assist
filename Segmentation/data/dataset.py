import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import config
import cv2
import numpy as np
from torch.utils.data import Dataset

from data.segmentation_transforms import SegmentationTransform
from data.utils import ImgAndTargetPlotter, copy_from_storage_if_needed

# from file naming Seg1, Seg2, Seg3
label_mapping = {1: 'necrotic_non_enhancing_tumor_core', 2: 'peritumoral_edema', 3: 'gd_enhancing_tumor'}


class MrSliceDataset(Dataset):
    def __init__(self,
                 input_paths: List[Tuple[Path, Path, Path, Path]],
                 target_paths: List[Union[Tuple[Path, Path, Path], Path]],
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any],
                 full_annotations: bool = False):

        self.name = type(self).__name__
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.pair_transforms = pair_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.full_annotations = full_annotations
        self.return_name = False

        self.preloaded_inputs = dict()
        self.preloaded_targets = dict()

    def load_input(self, index):
        if index in self.preloaded_inputs:
            return self.preloaded_inputs[index]
        n_channels = len(self.input_paths[index])
        base_img = cv2.imread(str(self.input_paths[index][0]), cv2.IMREAD_GRAYSCALE)
        im = np.zeros(base_img.shape + (n_channels,), dtype=base_img.dtype)
        im[..., 0] = base_img

        for c in range(1, n_channels):
            im[..., c] = cv2.imread(str(self.input_paths[index][c]), cv2.IMREAD_GRAYSCALE)
        return im

    def load_target(self, index):
        if index in self.preloaded_targets:
            return self.preloaded_targets[index]
        if isinstance(self.target_paths[index], tuple):
            target = cv2.imread(str(self.target_paths[index][1]), cv2.IMREAD_GRAYSCALE) > 128
            target = 2*target.astype(np.uint8)
            for c in (2, 0):
                channel_mask = cv2.imread(str(self.target_paths[index][c]), cv2.IMREAD_GRAYSCALE) > 128
                if np.any(np.logical_and(channel_mask > 0, target > 0)):
                    channel_mask = np.logical_and(channel_mask, np.logical_not(target > 0))
                target += (c+1) * channel_mask.astype(np.uint8)
        else:
            generated_target = cv2.imread(str(self.target_paths[index]), cv2.IMREAD_GRAYSCALE)
            target = np.zeros_like(generated_target, dtype=np.uint8)
            if self.full_annotations:
                # labels are coded as values
                # 0: background, mapped to 0
                # 51: CSF, mapped to 0
                # 102: gray matter, mapped to 0
                # 153: white matter, mapped to 0
                # 180: necrotic_non_enhancing_tumor_core, mapped to 1
                # 205: peritumoral_edema, mapped to 2
                # 235: gd_enhancing_tumor, mapped to 3
                target[np.logical_and(target == 0, generated_target > (235 + 205)/2)] = 3
                target[np.logical_and(target == 0, generated_target > (205 + 180)/2)] = 2
                target[np.logical_and(target == 0, generated_target > (180 + 153)/2)] = 1
            else:
                rescaled_target = 5/255*generated_target.astype(np.float32)
                # no guarantee that GAN generated labels are only 0, 1, 2, 4. Assign to closest label
                # GD-enhancing tumor (BRATS value 4, our 3)
                target[np.logical_and(target == 0, rescaled_target > 3.0)] = 3
                # peritumoral edema (BRATS value 2, our 2)
                target[np.logical_and(target == 0, rescaled_target > 1.5)] = 2
                # necrotic and non-enhancing tumor core (BRATS value 1, our 1)
                target[np.logical_and(target == 0, rescaled_target > 0.5)] = 1
        return target

    def __getitem__(self, index):
        print_name = self.input_paths[index][0].stem

        im = self.load_input(index)
        target = self.load_target(index)

        if self.pair_transforms is not None:
            (im, target) = self.pair_transforms(im, target)

        if self.image_transforms is not None:
            im = self.image_transforms(im)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        if self.return_name:
            return print_name, im, target
        return im, target

    def by_name(self, name, apply_transform=True):
        matching_indices = [i for i, paths in enumerate(self.input_paths) if name in name in paths[0].stem]
        assert len(matching_indices) == 1
        if apply_transform:
            return self[matching_indices[0]]
        else:
            return self.load_input(matching_indices[0])

    def set_transforms(self,
                       pair_transforms: Optional[SegmentationTransform],
                       image_transforms: Optional[Any],
                       target_transforms: Optional[Any]):
        self.pair_transforms = pair_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def get_available_subjects(self) -> List[str]:
        subjects = set()
        for input_path in self.input_paths:
            res = re.search(r'Subject_\d+', str(input_path[0]))
            subjects.add(res[0])
        return sorted(list(subjects))

    def get_indices_for_subject(self, subject_name) -> List[int]:
        pattern = re.compile(subject_name + r'_slice_(\d+)')
        slice_index_to_dataset_index = dict()
        for dataset_index, input_path in enumerate(self.input_paths):
            res = pattern.search(str(input_path[0]))
            if res is not None:
                slice_index = int(res[1])
                slice_index_to_dataset_index[slice_index] = dataset_index
        indices = []
        for slice_index in sorted(slice_index_to_dataset_index.keys()):
            indices.append(slice_index_to_dataset_index[slice_index])

        return indices

    def split(self, training_fraction: float, split_on_subject_level: bool = True, seed: int = 0):
        np.random.seed(seed)
        if split_on_subject_level:
            subjects = self.get_available_subjects()
            training_subjects = sorted(list(np.random.choice(subjects, int(
                training_fraction * len(subjects)+0.5), replace=False)))
            validation_subjects = sorted([subject for subject in subjects if subject not in training_subjects])

            assert len(set(training_subjects).intersection(set(validation_subjects))) == 0

            return self.get_subset_by_subject_names(training_subjects), \
                self.get_subset_by_subject_names(validation_subjects)
        else:
            training_indices = np.random.choice(len(self), int(training_fraction * len(self)+0.5), replace=False)
            validation_indices = np.array([i for i in range(len(self)) if i not in training_indices])

            assert len(set(training_indices).intersection(set(validation_indices))) == 0

            return self.get_subset(training_indices), \
                self.get_subset(validation_indices)

    def get_subset(self, subset_indices, seed: int = 0):
        subset_inputs = [self.input_paths[i] for i in subset_indices]
        subset_targets = [self.target_paths[i] for i in subset_indices]

        subset = MrSliceDataset(subset_inputs,
                                subset_targets,
                                pair_transforms=self.pair_transforms,
                                image_transforms=self.image_transforms,
                                target_transforms=self.target_transforms)

        subset.name = self.name
        return subset

    def get_subset_by_subject_names(self, subject_names):
        subset_indices = []
        for subject in subject_names:
            subset_indices.extend(self.get_indices_for_subject(subject))

        return self.get_subset(subset_indices=subset_indices)

    def __len__(self):
        return len(self.input_paths)

    def repeat(self, n_repeats: int):
        """Extends dataset by repeating n_repeats times,
           useful when we want to sample real data a specific amount of times
        """
        self.input_paths *= n_repeats
        self.target_paths *= n_repeats

    @property
    def estimated_size(self):
        im = self.load_input(0)
        target = self.load_target(0)
        return len(self) * (im.nbytes + target.nbytes) / 1024 ** 3

    def preload(self):
        for i in range(len(self)):
            self.preloaded_inputs[i] = self.load_input(i)
            self.preloaded_targets[i] = self.load_target(i)


class BratsBase(MrSliceDataset):
    def __init__(self,
                 root: Path,
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any],
                 separate_folders: bool = True,
                 single_segmentation_channel: bool = False,
                 full_annotations: bool = False):

        copy_from_storage_if_needed(root.parent)
        self.root = root

        # images
        all_images = dict()
        for image_folder in ('T1', 'T1CE', 'T2', 'FLAIR'):
            if separate_folders:
                modality_folder = self.root / image_folder
                if not modality_folder.exists() and image_folder == 'T1CE':
                    modality_folder = self.root / 'T1ce'
                if not modality_folder.exists() and image_folder == 'FLAIR':
                    modality_folder = self.root / 'flair'
                all_images[image_folder] = sorted(modality_folder.glob('*.png'))
            else:
                all_images[image_folder] = sorted(self.root.glob(f'*{image_folder}.png'))

        input_paths = list()
        for i in range(len(all_images['T1'])):
            input_paths.append((all_images['T1'][i], all_images['T1CE'][i],
                               all_images['T2'][i], all_images['FLAIR'][i]))

        # targets
        all_targets = dict()
        if single_segmentation_channel:
            if separate_folders:
                for potential_seg_folder in ('SEG', 'Seg', 'seg'):
                    seg_folder = self.root / potential_seg_folder
                    target_paths = sorted(seg_folder.glob('*.png'))
                    if len(target_paths) != 0:
                        break

            else:
                target_paths = sorted(self.root.glob('*Seg.png'))
        else:
            for target_folder in ('Seg1', 'Seg2', 'Seg3'):
                if separate_folders:
                    modality_folder = self.root / target_folder
                    all_targets[target_folder] = sorted(modality_folder.glob('*.png'))
                else:
                    all_targets[target_folder] = sorted(self.root.glob(f'*{target_folder}.png'))

            target_paths = list()
            for i in range(len(all_targets['Seg1'])):
                target_paths.append((all_targets['Seg1'][i], all_targets['Seg2'][i], all_targets['Seg3'][i]))

        super().__init__(input_paths,
                         target_paths,
                         pair_transforms,
                         image_transforms,
                         target_transforms,
                         full_annotations)


class Brats184(BratsBase):
    def __init__(self,
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any]):

        root = config.datasets / 'BRATS_184subjects' / \
            'brats2020_184subjects_noaugmentation_multiplesegmentationchannels_min15percentcoverage'

        super().__init__(root, pair_transforms, image_transforms, target_transforms)


class Brats313(BratsBase):
    def __init__(self,
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any]):

        root = config.datasets / 'BRATS_313subjects' / \
            'brats2020_313subjects_noaugmentation_multiplesegmentationchannels_min15percentcoverage'

        super().__init__(root, pair_transforms, image_transforms, target_transforms)


class Brats2021Train(BratsBase):
    def __init__(self,
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any]):

        root = config.datasets / 'BRATS2021_1195subjects' / \
            'brats2021_1195subjects_noaugmentation_singlesegmentationchannel_min15percentcoverage'

        super().__init__(root, pair_transforms, image_transforms, target_transforms, single_segmentation_channel=True)


class Brats2021Test(BratsBase):
    def __init__(self,
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any]):

        root = config.datasets / 'BRATS2021_56subjects' / \
            'brats2021_56subjects_noaugmentation_singlesegmentationchannel_min15percentcoverage_test'

        super().__init__(root, pair_transforms, image_transforms, target_transforms, single_segmentation_channel=True)


def get_brats_diff(fraction: float = 1.0, seed: int = 0):
    """Get dataset that is Brats313 with the subjects of Brats184 removed

    Args:
        fraction: fraction of images to include in dataset, in (0, 1.0]
        seed: random seed

    Returns:
        MrSliceDataset
    """
    np.random.seed(seed)
    assert fraction <= 1.0 and fraction > 0

    brats184 = Brats184(None, None, None)
    brats313 = Brats313(None, None, None)

    subjects184 = brats184.get_available_subjects()
    subjects313 = brats313.get_available_subjects()

    subjects_diff = list(set(subjects313) - set(subjects184))

    assert len(subjects313) - len(subjects184) == len(subjects_diff)

    if fraction < 1.0:
        subjects_diff = sorted(subjects_diff)
        np.random.shuffle(subjects_diff)
        num_to_include = round(fraction * len(subjects_diff))
        subjects_diff = subjects_diff[:num_to_include]

    diff_dataset = brats313.get_subset_by_subject_names(subjects_diff)
    diff_dataset.name = f'BratsDiff_{len(subjects_diff)}'

    return diff_dataset


class BratsTest(BratsBase):
    def __init__(self,
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any]):

        root = config.datasets / 'BRATS_testset_56subjects' / \
            'brats2020_56subjects_noaugmentation_multiplesegmentationchannels_min15percentcoverage_test'

        super().__init__(root, pair_transforms, image_transforms, target_transforms)


class GanBratsFromFolder(BratsBase):
    def __init__(self,
                 folder_name: str,
                 pair_transforms: Optional[SegmentationTransform],
                 image_transforms: Optional[Any],
                 target_transforms: Optional[Any],
                 must_exist: bool = False):

        root = config.datasets / folder_name

        if not root.exists() and must_exist:
            raise RuntimeError(f'Cannot find {folder_name}')
        copy_from_storage_if_needed(root)

        separate_folders = 'stylegan' in folder_name.lower() or 'diffusion' in folder_name.lower() or \
            'styelagan' in folder_name.lower()
        single_segmentation_channel = 'multiplesegmentationchannels' not in folder_name.lower()
        full_annotations = 'fullannotations' in folder_name.lower() or 'fullanonation' in folder_name.lower()

        inner_folder = os.listdir(root)

        if len(inner_folder) == 1:
            root = root / inner_folder[0]
        elif len(inner_folder) != 5:
            raise RuntimeError(f'Unknown number of inner folders: {len(inner_folder)}.')

        super().__init__(root,
                         pair_transforms=pair_transforms,
                         image_transforms=image_transforms,
                         target_transforms=target_transforms,
                         separate_folders=separate_folders,
                         single_segmentation_channel=single_segmentation_channel,
                         full_annotations=full_annotations)

        self.name = folder_name


class GanBrats(BratsBase):
    def __init__(self,
                 num_images: int,
                 single_channel: bool,
                 augmentation: str,
                 full_annotations: bool,
                 no_pixel_norm: bool = False,
                 no_leaky_relu: bool = False,
                 no_smoothing: bool = False,
                 no_repeat: bool = False,
                 stylegan: bool = False,
                 pair_transforms: Optional[SegmentationTransform] = None,
                 image_transforms: Optional[Any] = None,
                 target_transforms: Optional[Any] = None,
                 must_exist: bool = False):

        assert num_images in {184, 313}
        assert augmentation in {'no', 'mirror', 'rotation'}

        ch_str, ending = ('single', '') if single_channel else ('multiple', 's')
        full_annotation_str = '_fullannotations' if full_annotations else ''

        if no_pixel_norm:
            full_annotation_str += '_nopixelnorm'
        if no_leaky_relu:
            full_annotation_str += '_noleakyrelu'
        if no_smoothing:
            full_annotation_str += '_nosmoothing'
        if no_repeat:
            full_annotation_str += '_norepeat'

        gan_str = 'stylegan' if stylegan else 'synthetic'
        folder_name = f'BRATS_{gan_str}_{num_images}subjects_{augmentation}augmentation_{ch_str}' + \
            f'segmentationchannel{ending}{full_annotation_str}'
        root = config.datasets / folder_name

        if not root.exists() and must_exist:
            raise RuntimeError(f'Cannot find {folder_name}')

        copy_from_storage_if_needed(root)

        inner_folder = os.listdir(root)
        if len(inner_folder) == 1:
            root = root / inner_folder[0]
        elif len(inner_folder) != 5:
            raise RuntimeError(f'Unknown number of inner folders: {len(inner_folder)}.')

        super().__init__(root,
                         pair_transforms,
                         image_transforms,
                         target_transforms,
                         separate_folders=stylegan,
                         single_segmentation_channel=single_channel,
                         full_annotations=full_annotations)

        self.name = f'Gan{num_images}{augmentation}aug{ch_str}{full_annotation_str}'
        if stylegan:
            self.name += '_stylegan'

    def get_available_subjects(self):
        raise NotImplementedError('get_available_subjects not implemented for synthetic datasets')


if __name__ == '__main__':
    dataset = GanBratsFromFolder('184_styleGAN2_Gamma2_013000kimg', pair_transforms=None,
                                 image_transforms=None, target_transforms=None)

    dataset.return_name = True

    for name, img, target in dataset:
        if np.any(np.logical_and(target > 1, target < 255)):
            plotter = ImgAndTargetPlotter(img, {'target': target}, titles=['T1', 'T1CE', 'T2', 'FLAIR'])
            print(np.unique(target))
            plotter.show()
