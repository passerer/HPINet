import os
import random
from functools import partial

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from utils import modcrop


def load_image(image_file):
    return cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]


def load_npy(npy_file):
    return np.load(npy_file)


def is_image_file(image_file, extension):
    return any(image_file.endswith(ext) for ext in extension)


class FolderDataset(Dataset):
    """Build dataset from folder.
    Args:
        hr_dir (str): High-resolution image directory.
        lr_dir (str): low-resolution image directory.
        upscale (int): Upscale factor.
    """

    def __init__(self, hr_dir, lr_dir, upscale):
        super(FolderDataset, self).__init__()
        self.hr_files = self.get_image_file(hr_dir)
        self.lr_files = self.get_image_file(lr_dir)
        assert len(self.hr_files) == len(self.lr_files), \
            'Inconsistent number of image files: hr {}/lr {}'.format(len(self.hr_files),
                                                                     len(self.lr_files))
        self.upscale = upscale

    @staticmethod
    def get_image_file(image_dir):
        return sorted(filter(partial(is_image_file, extension=['.bmp', '.png', '.jpg']),
                             [os.path.join(image_dir, x) for x in os.listdir(image_dir)]))

    def __getitem__(self, index):
        lr = load_image(self.lr_files[index])
        hr = load_image(self.hr_files[index])
        lr = to_tensor(lr.copy())
        hr = to_tensor(modcrop(hr.copy(), self.upscale))
        return lr, hr

    def __len__(self):
        return len(self.lr_files)


class DIV2KTrainDataset(Dataset):
    """Build DIV2K dataset for training.
    Args:
        upscale (int): Upscale factor.
        patch_size (int): Size of patch. To reduce memory during training,
            usually use cropped patch instead of entire image.
        root (str): Root path of DIV2K.
        repeat (int): Repeat times of dataset.
        n_train (int): original numbers of images for training. DIV2K contains 900 images,
            800 of which are for training and the rest are for testing/validating.
        ext (str): image suffix. '.png' is more common,
            but you are suggested to convert images to '.npy' to speed up IO
    """

    def __init__(self, upscale, patch_size, root, repeat=1, n_train=800, ext='.png'):
        super(DIV2KTrainDataset, self).__init__()
        self.upscale = upscale
        self.patch_size = patch_size
        self.root = root
        self.repeat = repeat
        self.n_train = n_train
        if ext not in ('.png', '.npy'):
            raise KeyError('Unwanted image extension: {}'.format(ext))
        self.ext = ext
        self._set_filesystem(self.root)
        self.hr_files = self.get_image_file(self.hr_dir)[:n_train]
        self.lr_files = self.get_image_file(self.lr_dir)[:n_train]
        assert len(self.hr_files) == len(self.lr_files), \
            'Inconsistent number of image files: hr {}/lr {}'.format(len(self.hr_files),
                                                                     len(self.lr_files))
        self.load_file = load_image if ext == '.png' else load_npy

    def get_image_file(self, image_dir):
        return sorted(filter(partial(is_image_file, extension=[self.ext, ]),
                             [os.path.join(image_dir, x) for x in os.listdir(image_dir)]))

    def _set_filesystem(self, dir_data):
        self.hr_dir = os.path.join(self.root, 'DIV2K_train_HR')
        self.lr_dir = os.path.join(self.root, 'DIV2K_train_LR_bicubic/X{}'.format(self.upscale))

    def __getitem__(self, index):
        index = self._get_index(index)
        lr = self.load_file(self.lr_files[index])
        hr = self.load_file(self.hr_files[index])
        patch_size = self.patch_size
        if hr.shape[0] < patch_size or hr.shape[1] < patch_size:
            next_index = random.randrange(0, len(self))
            return self.__getitem__(next_index)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = self.augment(lr, hr)
        lr, hr = to_tensor(lr.copy()), to_tensor(hr.copy())
        return lr, hr

    def __len__(self):
        return self.n_train * self.repeat

    def _get_index(self, idx):
        return idx % self.n_train

    @staticmethod
    def augment(lr, hr, hflip=True, vflip=True, rgb_permute=False):
        """Augment function.
        Args:
            lr (ndarray): input image.
            hr (Tensor): high resolution counterpart of input.
            hflip (bool): Whether to flip horizontally.
            vflip (bool): Whether to flip vertically.
            rgb_permute (bool): Whether to permute rgb channels.
        """
        hflip = hflip and random.random() < 0.5
        vflip = vflip and random.random() < 0.5
        rot90 = vflip and random.random() < 0.5
        rgb_permute = rgb_permute and random.random() < 0.3
        if rgb_permute:
            permutation = np.random.permutation(3)

        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            if rgb_permute: img = img[:, :, permutation]

            return img

        return _augment(lr), _augment(hr)

    def _get_patch(self, lr, hr):
        lr_h, lr_w = lr.shape[:2]
        hr_p = self.patch_size
        lr_p = hr_p // self.upscale
        lr_x = random.randrange(0, lr_w - lr_p + 1)
        lr_y = random.randrange(0, lr_h - lr_p + 1)
        hr_x, hr_y = self.upscale * lr_x, self.upscale * lr_y
        lr = lr[lr_y:lr_y + lr_p, lr_x:lr_x + lr_p]
        hr = hr[hr_y:hr_y + hr_p, hr_x:hr_x + hr_p]
        return lr, hr
