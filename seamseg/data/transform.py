from argparse import ArgumentError
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tfn

from seamseg.utils.bbx import extract_boxes

from functools import partial


class ISSTransform:
    """Transformer function for instance segmentation

    Parameters
    ----------
    shortest_size : int
        Outputs size of the shortest image dimension. If scale augmentation is enabled this is multiplied by a random
        value sampled from the range defined by `random_scale`
    longest_max_size : int
        Maximum size of the input image after the initial scaling: images larger than this are rescaled so that their
        longest dimension becomes `longest_max_size` before applying any other transformation
    rgb_mean : tuple of float or None
        Per-channel mean values to use when normalizing the images, or None to disable mean normalization
    rgb_std : tuple of float or None
        Per-channel std values to use when normalizing the images, or None to disable std normalization
    random_flip : bool
        If true enable random flipping
    random_scale : tuple of float or None
        If not None, apply random scaling. If `random_scale` has two elements, they are interpreted as lower and upper
        limits for continuous sampling. If it has more, they are interpreted as discrete scale values, one of each will
        be chosen at random each time `ISSTransform` is applied to an image.
    """

    def __init__(
        self,
        shortest_size,
        longest_max_size,
        rgb_mean=None,
        rgb_std=None,
        random_flip=False,
        random_scale=None,
    ):
        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.random_flip = random_flip
        self.random_scale = random_scale

    def _adjusted_scale(self, in_width, in_height, target_size):
        min_size = min(in_width, in_height)
        max_size = max(in_width, in_height)
        scale = target_size / min_size

        if int(max_size * scale) > self.longest_max_size:
            scale = self.longest_max_size / max_size

        return scale

    @staticmethod
    def _random_flip(img, msk):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in msk]
            return img, msk
        else:
            return img, msk

    def _random_target_size(self):
        if len(self.random_scale) == 2:
            target_size = random.uniform(
                self.shortest_size * self.random_scale[0], self.shortest_size * self.random_scale[1]
            )
        else:
            target_sizes = [self.shortest_size * scale for scale in self.random_scale]
            target_size = random.choice(target_sizes)
        return int(target_size)

    def _normalize_image(self, img):
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    @staticmethod
    def _compact_labels(msk, cat, iscrowd):
        ids = np.unique(msk)
        if 0 not in ids:
            ids = np.concatenate((np.array([0], dtype=np.int32), ids), axis=0)

        ids_to_compact = np.zeros((ids.max() + 1,), dtype=np.int32)
        ids_to_compact[ids] = np.arange(0, ids.size, dtype=np.int32)

        msk = ids_to_compact[msk]
        cat = cat[ids]
        iscrowd = iscrowd[ids]

        return msk, cat, iscrowd

    def __call__(self, img, msk, cat, iscrowd):
        # Random flip
        if self.random_flip:
            img, msk = self._random_flip(img, msk)

        # Adjust scale, possibly at random
        if self.random_scale is not None:
            target_size = self._random_target_size()
        else:
            target_size = self.shortest_size
        scale = self._adjusted_scale(img.size[0], img.size[1], target_size)

        out_size = tuple(int(dim * scale) for dim in img.size)
        img = img.resize(out_size, resample=Image.BILINEAR)
        msk = [m.resize(out_size, resample=Image.NEAREST) for m in msk]

        # Wrap in np.array
        cat = np.array(cat, dtype=np.int32)
        iscrowd = np.array(iscrowd, dtype=np.uint8)

        # Image transformations
        img = tfn.to_tensor(img)
        img = self._normalize_image(img)

        # Label transformations
        msk = np.stack([np.array(m, dtype=np.int32, copy=False) for m in msk], axis=0)
        msk, cat, iscrowd = self._compact_labels(msk, cat, iscrowd)

        # Convert labels to torch and extract bounding boxes
        msk = torch.from_numpy(msk.astype(np.long))
        cat = torch.from_numpy(cat.astype(np.long))
        iscrowd = torch.from_numpy(iscrowd)
        bbx = extract_boxes(msk, cat.numel())

        return dict(img=img, msk=msk, cat=cat, iscrowd=iscrowd, bbx=bbx)


class ISSTestTransform:
    """Transformer function for instance segmentation, test time

    Parameters
    ----------
    shortest_size : int
        Outputs size of the shortest image dimension.
    rgb_mean : tuple of float or None
        Per-channel mean values to use when normalizing the images, or None to disable mean normalization
    rgb_std : tuple of float or None
        Per-channel std values to use when normalizing the images, or None to disable std normalization
    """

    def __init__(self, shortest_size, rgb_mean=None, rgb_std=None):
        self.shortest_size = shortest_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def _adjusted_scale(self, in_width, in_height):
        min_size = min(in_width, in_height)
        scale = self.shortest_size / min_size
        return scale

    def _normalize_image(self, img):
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    def __call__(self, img):
        # Adjust scale
        scale = self._adjusted_scale(img.size[0], img.size[1])

        out_size = tuple(int(dim * scale) for dim in img.size)
        img = img.resize(out_size, resample=Image.BILINEAR)

        # Image transformations
        img = tfn.to_tensor(img)
        img = self._normalize_image(img)

        return img


class Mapilary_output_to_city_output(object):
    def __init__(self) -> None:
        super().__init__()
        # convert the relevant classes. Rest is converted to nan
        lookup_dict = {
            10: 0,  # road
            12: 1,  # sidewalk
            31: 11,  # person
            58: 13,  # car
        }
        self.void_value = 255
        values = []
        for i in range(65):
            if i in lookup_dict:
                values.append(lookup_dict[i])
            else:
                values.append(self.void_value)
        self.lookup_dict = dict(zip(range(65), values))

        self.vistas_stuff = 28
        self.vistas_things = 37

        self.city_stuff = 11
        self.city_things = 8

    def __call__(self, inp):

        # convert to cpu
        inp = dict(
            [(key, val.cpu()) if torch.is_tensor(val) else (key, val) for (key, val) in inp.items()]
        )
        inp["sem_pred"] = inp["sem_pred"].apply_(lambda x: self.lookup_dict[x])
        inp["cls_pred"] = inp["cls_pred"].apply_(lambda x: self.lookup_dict[x + self.vistas_stuff])
        inp["cls_pred"] = inp["cls_pred"].apply_(
            lambda x: x - self.city_stuff if x != self.void_value else x
        )

        # remove void predictions from things stuff
        void_msk = inp["cls_pred"] == self.void_value
        # first void prediction is necessary

        for key in ["bbx_pred", "cls_pred", "obj_pred", "msk_pred"]:
            inp[key] = inp[key][~void_msk]

        return inp


class TrainedToTarget:
    """Transforms data which was trained on Mapillary Vistas to Cityscapes GT"""

    def __init__(self, mode, void_value=255) -> None:
        self.void_value = void_value
        if mode == "M2C":  # Mode to convert from Mapillary to Cityscapes
            self.lookup_dict = {
                10: 0,  # Road
                12: 1,  # Sidewalk
                14: 2,  # Building
                4: 3,  # Wall
                1: 4,  # Fence
                48: 5,  # Pole
                50: 5,  # Pole
                51: 6,  # Traffic light
                49: 7,  # Traffic sign
                52: 7,  # Traffic sign
                53: 7,  # Traffic sign
                22: 8,  # Vegetation
                21: 9,  # Terain
                19: 10,  # Sky
                31: 11,  # Person
                32: 12,  # Rider
                33: 12,  # Rider
                34: 12,  # Rider
                58: 13,  # Car
                63: 14,  # Truck
                57: 15,  # Bus
                60: 17,  # motorcycle
                55: 18,  # bike
            }
            self.vistas_stuff = 28
            self.vistas_things = 37

        elif mode is None:
            raise ArgumentError(
                "Need to provid a lookup dictionary to map values between Mapilary output of Net to target"
            )

    def _func_mapper(self, val: int):
        if val in self.lookup_dict:
            return self.lookup_dict[val]
        else:
            return self.void_value

    def __call__(self, panoptic_input):
        msk_pred, cat_pred, obj_pred, iscrowd_pred = panoptic_input
        cat_pred = cat_pred.apply_(self._func_mapper)

        void_msk = cat_pred == self.void_value
        # First void value is tolerated
        void_msk[0] = 0

        offset = 0
        for i in range(len(void_msk)):
            if void_msk[i]:
                msk_pred[msk_pred == i] = 0
                offset += 1
            else:
                msk_pred[msk_pred == i] = i - offset

        cat_pred = cat_pred[~void_msk]
        obj_pred = obj_pred[~void_msk]
        iscrowd_pred = iscrowd_pred[~void_msk]

        return [msk_pred, cat_pred, obj_pred, iscrowd_pred]
