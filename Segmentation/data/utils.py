import os
import shutil
import tarfile
from typing import Optional, Sequence, Union, Dict

import config
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision.transforms.transforms import Compose, Normalize


class ImgAndTargetPlotter:
    def __init__(self,
                 img: np.ndarray,
                 segs: Optional[Dict[str, np.ndarray]],
                 seg_colormap: ListedColormap = ListedColormap(['#FFFFFF00', 'g', 'b', 'r', 'c', 'm']),
                 seg_alpha: float = 0.25,
                 titles: Optional[Sequence[str]] = None):

        self.img = img
        self.seg_dict = segs
        self.show_seg = True
        self.seg_colormap = seg_colormap
        self.contour_colormap = ListedColormap(seg_colormap.colors[1:])
        self.seg_alpha = seg_alpha

        self.seg_index = 0
        self.segs = []
        self.seg_names = []
        if self.seg_dict is not None:
            for name, seg in self.seg_dict.items():
                self.segs.append(seg)
                self.seg_names.append(name)
        self.show_as_countours = False
        self.titles = titles

        if img.ndim == 2:
            self.fig, self.axes = plt.subplots(1, 1)
            self.img = np.expand_dims(self.img, axis=-1)
        else:
            assert self.img.shape[2] <= 4, 'Viewer does not support images with more than 4 channels'
            self.fig, self.axes = plt.subplots(2, 2, sharex=True, sharey=True)

        if titles is not None:
            assert len(self.titles) == self.img.shape[2]
        self._draw()
        self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)

    def _draw(self):
        for channel in range(self.img.shape[2]):
            ax_index = np.unravel_index(channel, self.axes.shape)
            ax = self.axes[ax_index]
            ax.clear()
            ax.imshow(self.img[:, :, channel], cmap='gray')
            if self.show_seg and self.seg_dict is not None:
                if self.show_as_countours:
                    ax.contour(self.segs[self.seg_index],
                               levels=np.arange(0.5, 5.5, 1),
                               cmap=self.contour_colormap,
                               linewidths=0.5,
                               interpolation='nearest')
                else:
                    ax.imshow(self.segs[self.seg_index],
                              vmin=0,
                              vmax=5,
                              cmap=self.seg_colormap,
                              alpha=self.seg_alpha,
                              interpolation='nearest')
            if self.titles is not None:
                ax.set_title(self.titles[channel])
            ax.axis('off')

        if self.show_seg and self.seg_dict is not None:
            self.fig.suptitle(self.seg_names[self.seg_index])
        else:
            self.fig.suptitle('')

        self.fig.canvas.draw()

    def show(self):
        plt.show()

    def _on_keypress(self, event):
        if event.key == 'h':
            self.show_seg = not self.show_seg
        if event.key == 'm':
            self.seg_index = (self.seg_index + 1) % len(self.segs)
        if event.key == 'c':
            self.show_as_countours = not self.show_as_countours
        self._draw()


def plot_img_and_target(img, target):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
        if target is not None:
            plt.imshow(target, vmin=0, vmax=5, cmap=ListedColormap(
                ['#FFFFFF00', 'g', 'b', 'r', 'c', 'm']), alpha=0.25, interpolation='nearest')
    else:
        assert img.shape[2] <= 4
        for i in range(img.shape[2]):
            plt.subplot(2, 2, i + 1)
            plot_img_and_target(img[:, :, i], target)


def revert_normalization_for_plot(im: torch.Tensor, transform: Union[Compose, Normalize]):
    normalize = None
    if isinstance(transform, Compose):
        for tf in transform.transforms:
            if isinstance(tf, Normalize):
                normalize = tf
                break
    else:
        normalize = transform

    if normalize is None:
        mins = torch.stack(tuple(imdim.min() for imdim in torch.unbind(im, dim=0)), dim=0)
        maxs = torch.stack(tuple(imdim.max() for imdim in torch.unbind(im, dim=0)), dim=0)
        std = 1/(maxs-mins)
        mean = im.shape[0]*[0.5]
    else:
        mean = normalize.mean
        std = normalize.std
    return np.round(255*(np.moveaxis(np.array(im), (0, 1, 2), (2, 0, 1))*np.array(std)+np.array(mean))).astype(np.uint8)


def symmetric_pad_to_shape(img: np.ndarray, shape, pad_val) -> np.ndarray:
    h, w = img.shape[:2]
    th, tw = shape

    if w < tw:
        x1 = int(round((w - tw) / 2))
        if img.ndim == 3:
            img = np.pad(img, pad_width=((0, 0), (-x1, x1 + tw - w), (0, 0)), constant_values=pad_val)
        else:
            img = np.pad(img, pad_width=((0, 0), (-x1, x1 + tw - w)), constant_values=pad_val)
    if h < th:
        y1 = int(round((h - th) / 2))
        if img.ndim == 3:
            img = np.pad(img, pad_width=((-y1, y1 + th - h), (0, 0), (0, 0)), constant_values=pad_val)
        else:
            img = np.pad(img, pad_width=((-y1, y1 + th - h), (0, 0)), constant_values=pad_val)
    return img


def copy_from_storage_if_needed(source_folder):
    """Copies dataset stored in config.storage as a .tar file to dataset folder and extracts"""
    if not source_folder.exists():
        storage_file = f'{str(source_folder).replace(str(config.datasets), str(config.storage))}.tar'
        tar_filename = storage_file.split('/')[-1]
        print(f'copying from {storage_file} to {source_folder}')

        if not config.datasets.exists():
            config.datasets.mkdir()

        shutil.copyfile(storage_file, config.datasets / tar_filename)

        tar = tarfile.open(config.datasets / tar_filename)
        tar.extractall(path=config.datasets)
        tar.close()

        os.remove(config.datasets / tar_filename)
