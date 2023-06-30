import abc
from typing import Tuple, Union

import numpy as np
from scipy.ndimage.interpolation import rotate, zoom

import config
from data.utils import symmetric_pad_to_shape


class SegmentationTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class Compose(SegmentationTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert img.shape[:2] == mask.shape
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(SegmentationTransform):
    def __init__(self,
                 size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert img.shape[:2] == mask.shape
        h, w = img.shape[:2]
        th, tw = self.size

        img = symmetric_pad_to_shape(img, self.size, pad_val=0)
        mask = symmetric_pad_to_shape(mask, self.size, pad_val=config.ignore_value)

        h, w = img.shape[:2]

        if w == tw and h == th:
            return img, mask

        x1 = np.random.randint(0, w - tw) if w > tw else 0
        y1 = np.random.randint(0, h - th) if h > th else 0

        return img[y1:y1+th, x1:x1+tw, :], mask[y1:y1+th, x1:x1+tw]


class CenterCrop(SegmentationTransform):
    def __init__(self, size:  Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert img.shape[:2] == mask.shape
        w, h = img.shape[:2]
        th, tw = self.size

        if w == tw and h == th:
            return img, mask

        img = symmetric_pad_to_shape(img, self.size, pad_val=0)
        mask = symmetric_pad_to_shape(mask, self.size, pad_val=config.ignore_value)

        h, w = img.shape[:2]

        x1 = int(round((w - tw) / 2))
        y1 = int(round((h - th) / 2))
        return img[y1:y1+th, x1:x1+tw, :], mask[y1:y1+th, x1:x1+tw]


class RandomRotate(SegmentationTransform):
    def __init__(self, degree, reshape: bool = True, apply_probability: float = 1.0):
        self.degree = degree
        self.reshape = reshape
        self.apply_probability = apply_probability

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.apply_probability:
            angle = np.random.random() * 2 * self.degree - self.degree

            img = rotate(img, angle, reshape=self.reshape)
            mask = rotate(mask, angle, reshape=self.reshape, order=0, cval=config.ignore_value)

        return img, mask


class RandomScaleAndCrop(SegmentationTransform):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 low: float,
                 high: float,
                 keep_aspect: bool = True,
                 apply_probability: float = 1.0):
        self.low = low
        self.high = high
        self.keep_aspect = keep_aspect
        self.crop = RandomCrop(size)
        self.apply_probability = apply_probability

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert img.shape[:2] == mask.shape

        if np.random.random() < self.apply_probability:
            zoom_w = np.random.uniform(self.low, self.high)
            zoom_h = zoom_w if self.keep_aspect else np.random.uniform(self.low, self.high)

            img = zoom(img, (zoom_h, zoom_w, 1))
            mask = zoom(mask, (zoom_h, zoom_w), order=0, cval=config.ignore_value)

        return self.crop(img, mask)
