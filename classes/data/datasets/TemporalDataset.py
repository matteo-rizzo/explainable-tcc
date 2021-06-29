from typing import Tuple

import numpy as np
import torch
import torch.utils.data as data

from auxiliary.settings import PATH_TO_DATASET
from auxiliary.utils import hwc_chw, gamma_correct, brg_to_rgb
from classes.data.DataAugmenter import DataAugmenter


class TemporalDataset(data.Dataset):

    def __init__(self, mode, input_size):
        self.__input_size = input_size
        self.__da = DataAugmenter(input_size)
        self._mode = mode
        self._path_to_dataset = PATH_TO_DATASET
        self._data_dir, self._label_dir = "ndata_seq", "nlabel"
        self._paths_to_items = []

    def __getitem__(self, index: int) -> Tuple:
        path_to_sequence = self._paths_to_items[index]
        label_path = path_to_sequence.replace(self._data_dir, self._label_dir)

        x = np.array(np.load(path_to_sequence), dtype='float32')
        illuminant = np.array(np.load(label_path), dtype='float32')
        m = torch.from_numpy(self.__da.augment_mimic(x).transpose((0, 3, 1, 2)).copy())

        if self._mode == "train":
            x, color_bias = self.__da.augment_sequence(x, illuminant)
            color_bias = np.array([[[color_bias[0][0], color_bias[1][1], color_bias[2][2]]]], dtype=np.float32)
            m = torch.mul(m, torch.from_numpy(color_bias).view(1, 3, 1, 1))
        else:
            x = self.__da.resize_sequence(x)

        x = np.clip(x, 0.0, 255.0) * (1.0 / 255)
        x = hwc_chw(gamma_correct(brg_to_rgb(x)))

        x = torch.from_numpy(x.copy())
        illuminant = torch.from_numpy(illuminant.copy())

        return x, m, illuminant, path_to_sequence

    def __len__(self) -> int:
        return len(self._paths_to_items)
