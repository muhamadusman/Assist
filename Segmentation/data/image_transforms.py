from typing import Tuple

import numpy as np
import torch

from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import zoom


class GaussianNoiseAugmentation:
    def __init__(self,
                 noise_variance_range: Tuple[float, float],
                 channel_apply_probability: float,
                 apply_probability: float,
                 only_on_foreground: bool = True):
        self.noise_variance_range = noise_variance_range
        self.channel_apply_probability = channel_apply_probability
        self.apply_probability = apply_probability
        self.only_on_foreground = only_on_foreground

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.apply_probability:
            for c in range(img.shape[2]):
                if np.random.uniform() <= self.channel_apply_probability:
                    if self.only_on_foreground:
                        fg_mask = img[..., c] != img[..., c].min()
                        channel_data = img[..., c]
                        channel_data = channel_data[fg_mask]
                    else:
                        channel_data = img[..., c]

                    noise_variance = np.random.uniform(*self.noise_variance_range)
                    channel_data = channel_data + np.random.normal(0.0, noise_variance, size=channel_data.shape)

                    if self.only_on_foreground:
                        augmented_data = channel_data
                        channel_data = img[..., c]
                        channel_data[fg_mask] = augmented_data
                    img[..., c] = channel_data
        return img


class GaussianBlurAugmentation:
    def __init__(self, sigma_range: Tuple[float, float], channel_apply_probability: float, apply_probability: float):
        self.sigma_range = sigma_range
        self.apply_probability = apply_probability
        self.channel_apply_probability = channel_apply_probability

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.apply_probability:
            for c in range(img.shape[2]):
                if np.random.uniform() <= self.channel_apply_probability:
                    sigma = np.random.uniform(*self.sigma_range)
                    img[..., c] = gaussian_filter(img[..., c], sigma, order=0)
        return img


class ResolutionAugmentation:
    def __init__(self, zoom_range: Tuple[float, float], channel_apply_probability: float, apply_probability: float):
        self.zoom_range = zoom_range
        self.channel_apply_probability = channel_apply_probability
        self.apply_probability = apply_probability

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.apply_probability:
            original_shape = img.shape[:2]
            for c in range(img.shape[2]):

                if np.random.uniform() < self.channel_apply_probability:
                    zoom_factor = np.random.uniform(*self.zoom_range)

                    downsampled = zoom(img[..., c], zoom_factor)
                    upsampled = zoom(downsampled, 1/zoom_factor, order=0)
                    shape_for_assign = tuple(min(us, os) for us, os in zip(upsampled.shape, original_shape))
                    img[:shape_for_assign[0], :shape_for_assign[1],
                        c] = upsampled[:shape_for_assign[0], :shape_for_assign[1]]
        return img


class GammaAugmentor:
    def __init__(self,
                 gamma_range: Tuple[float, float],
                 apply_probability: float,
                 keep_mean_and_std: bool = True,
                 only_on_foreground: bool = True,
                 invert_image: bool = False):
        """Performs gamma augmentation. Note that half of the time, gamma is randomized from range < 1 and half of the
        time, gamma is randomized from range > 1 (following nnunet).

        Args:
            gamma_range: range of gamma augmentation factor
            apply_probability: probability of which to apply augmentation
            keep_mean_and_std: if true, keeps same mean and std
            only_on_foreground: if true, only applies augmentation on part of channel that is not background
            invert_image: if gamma augmentation should be applied to inverted image
        """
        self.apply_probability = apply_probability
        self.gamma_range = gamma_range
        self.keep_mean_and_std = keep_mean_and_std
        self.only_on_foreground = only_on_foreground
        self.invert_image = invert_image
        self.epsilon = 1e-7

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.invert_image:
            img *= -1.0

        if np.random.random() < self.apply_probability:
            for c in range(img.shape[2]):
                channel_data = img[..., c]
                if self.only_on_foreground:
                    fg_mask = channel_data > channel_data.min()
                    channel_data = channel_data[fg_mask]

                if self.keep_mean_and_std:
                    channel_mean = channel_data.mean()
                    channel_std = channel_data.std()

                if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                    gamma = np.random.uniform(self.gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])

                img_min = channel_data.min()
                img_range = channel_data.max() - img_min
                channel_data = np.power(((channel_data - img_min) / float(img_range + self.epsilon)),
                                        gamma) * float(img_range + self.epsilon) + img_min

                if self.keep_mean_and_std:
                    channel_data = channel_data - channel_data.mean()
                    channel_data = channel_data / (channel_data.std() + 1e-8) * channel_std
                    channel_data = channel_data + channel_mean

                if self.only_on_foreground:
                    augmented_data = channel_data
                    channel_data = img[..., c]
                    channel_data[fg_mask] = augmented_data
                img[..., c] = channel_data

        if self.invert_image:
            img *= -1.0

        return img


class ScaleAugmentation:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, img: np.ndarray) -> np.ndarray:
        n_channels = img.shape[2]
        rands = (self.high-self.low) * np.random.rand(n_channels) + self.low
        img = img * rands.reshape((1, 1, n_channels))
        return img.astype(np.uint8)


class OffsetAugmentation:
    def __init__(self, low, high, keep_zeros=True):
        self.low = low
        self.high = high
        self.keep_zeros = keep_zeros

    def __call__(self, img: np.ndarray) -> np.ndarray:
        zero_map = None
        if self.keep_zeros:
            zero_map = img == 0

        n_channels = img.shape[2]
        rands = (self.high-self.low) * np.random.rand(n_channels) + self.low
        img = img + rands.reshape((1, 1, n_channels))

        if zero_map is not None:
            img[zero_map] = 0
            img[img < 0] = 0
        return img.astype(np.uint8)


class ZscoringNormalize:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        normalized_channels = []
        for channel_im in torch.unbind(img, dim=0):
            foreground_mask = channel_im != channel_im.min()
            channel_mean = channel_im[foreground_mask].mean()
            channel_std = channel_im[foreground_mask].std()
            channel_im = (channel_im - channel_mean)/channel_std
            normalized_channels.append(channel_im)
        return torch.stack(normalized_channels, dim=0)
