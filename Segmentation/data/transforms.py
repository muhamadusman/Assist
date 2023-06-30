
import numpy as np
import torch
from torchvision import transforms

import data.segmentation_transforms as segmentation_transforms
from data.image_transforms import GammaAugmentor, GaussianBlurAugmentation, GaussianNoiseAugmentation, \
    ResolutionAugmentation, ZscoringNormalize
from data.utils import ImgAndTargetPlotter, symmetric_pad_to_shape


class ToFloat:
    def __init__(self, scaling: float):
        self.scaling = scaling

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.scaling * img.astype(np.float32)


class AnnoToTensor:
    def __call__(self, mask):
        arr = mask.astype(np.int64)
        return torch.tensor(arr)


def get_transforms(output_size,
                   input_size,
                   coloraug: bool = False,
                   geoaug: bool = False,
                   individual_norm: bool = True):

    if individual_norm:
        normalization = ZscoringNormalize()
    else:
        normalization = transforms.Normalize(mean=4*[0.5], std=4*[1.0])

    if geoaug:
        train_segmentation_transforms = segmentation_transforms.Compose([
            segmentation_transforms.RandomRotate(30, reshape=False, apply_probability=0.75),
            segmentation_transforms.RandomScaleAndCrop(
                input_size, low=0.9, high=1.1, keep_aspect=False, apply_probability=0.75)
        ])
    else:
        train_segmentation_transforms = segmentation_transforms.Compose([
            segmentation_transforms.RandomCrop(input_size)
        ])

    if coloraug:
        train_image_transforms = transforms.Compose([
            ToFloat(scaling=1/256),
            GaussianNoiseAugmentation(noise_variance_range=(
                0, 0.05), apply_probability=0.5, channel_apply_probability=0.5),
            GaussianBlurAugmentation(sigma_range=(0.5, 1.0), apply_probability=0.2, channel_apply_probability=0.5),
            ResolutionAugmentation(zoom_range=(0.75, 1.0), apply_probability=0.25, channel_apply_probability=0.5),
            GammaAugmentor(gamma_range=(0.85, 1.2), apply_probability=0.1,  invert_image=True),
            GammaAugmentor(gamma_range=(0.85, 1.2), apply_probability=0.2,  invert_image=False),
            transforms.ToTensor(),
            normalization
        ])
    else:
        train_image_transforms = transforms.Compose([
            ToFloat(scaling=1/256),
            transforms.ToTensor(),
            normalization
        ])
    trainval_target_transforms = transforms.Compose([
        AnnoToTensor(),
        transforms.CenterCrop(output_size)
    ])

    val_test_image_transforms = transforms.Compose([
        ToFloat(scaling=1/256),
        transforms.ToTensor(),
        transforms.CenterCrop(input_size),
        normalization
    ])

    train_transforms = train_segmentation_transforms, train_image_transforms, trainval_target_transforms
    val_transforms = None, val_test_image_transforms, trainval_target_transforms
    test_transforms = None, val_test_image_transforms, None

    return train_transforms, val_transforms, test_transforms


if __name__ == '__main__':
    from learning.models import UNet
    from matplotlib import pyplot as plt

    from data.dataset import BratsTest
    model = UNet(output_size=(256, 256), in_channels=4, n_classes=4, padding='same')

    train, val, test = get_transforms(output_size=model.output_size,
                                      input_size=model.input_size,
                                      coloraug=True,
                                      geoaug=True)
    train_dataset = BratsTest(*train)
    val_dataset = BratsTest(*val)

    for train_sample, val_sample in zip(train_dataset, val_dataset):
        train_im, train_target = train_sample
        val_im, val_target = val_sample

        train_im = np.moveaxis(np.array(train_im), (0, 1, 2), (2, 0, 1))
        train_target = np.squeeze(np.array(train_target))
        train_target = symmetric_pad_to_shape(train_target, train_im.shape[:2], pad_val=255)

        val_im = np.moveaxis(np.array(val_im), (0, 1, 2), (2, 0, 1))
        val_target = np.squeeze(np.array(val_target))
        val_target = symmetric_pad_to_shape(val_target, val_im.shape[:2], pad_val=255)

        ImgAndTargetPlotter(train_im, {'target': train_target})
        ImgAndTargetPlotter(val_im, {'target': val_target})
        if np.any(np.logical_and(train_target > 0, train_target < 255)):
            plt.show()
        else:
            plt.close('all')
