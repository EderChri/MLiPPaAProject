"""
Dataset module.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd

from constants import *
from utils import cart2cyl, sort_by_angle, load_variable_len_data


class TrajectoryDataset(Dataset):

    def __init__(self, root, labels, normalize=False, shuffle=False, to_tensor=True):
        self.root = root
        self.labels = load_variable_len_data(labels)
        #pd.read_csv(labels, header=None)
        self.data = load_variable_len_data(self.root)
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data = self.data.fillna(value=PAD_TOKEN)
        self.labels = self.labels.fillna(value=PAD_TOKEN)
        self.total_events = len(self.data)
        self.normalize = normalize
        self.to_tensor = to_tensor

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        # load event
        data = self.data.iloc[[idx]].values.tolist()[0]
        event_id = int(data[0])
        event_labels = self.labels.iloc[[event_id]].values.tolist()[0]

        labels = event_labels[2::2]
        x = data[1::DIMENSION + 1]
        y = data[2::DIMENSION + 1]
        z = None
        x = [float(value) for value in x]
        y = [float(value) for value in y]
        labels = [float(value) for value in labels]
        if DIMENSION == 3:
            z = data[3::DIMENSION + 1]
            z = [float(value) for value in z]
        if DIMENSION == 2:
            z = [PAD_TOKEN] * len(x)

        # normalise
        if self.normalize:
            raise NotImplementedError()

        convert_list = []
        for i in range(len(x)):
            if DIMENSION == 3:
                convert_list.append((x[i], y[i], z[i]))
            if DIMENSION == 2:
                convert_list.append((x[i], y[i]))

        sorted_coords = sort_by_angle(convert_list)
        labels = np.sort(labels)

        if DIMENSION == 2:
            x, y = zip(*sorted_coords)
        else:
            x, y, z = zip(*sorted_coords)

        # convert to tensors
        if self.to_tensor:
            pad_x = PADDING_LEN - len(x)
            pad_y = PADDING_LEN - len(y)
            pad_z = PADDING_LEN - len(z)
            pad_labels = PADDING_LEN - len(labels)

            x = F.pad(torch.tensor(x).float(), pad=(0, pad_x), value=PAD_TOKEN)
            y = F.pad(torch.tensor(y).float(), pad=(0, pad_y), value=PAD_TOKEN)
            z = F.pad(torch.tensor(z).float(), pad=(0, pad_z), value=PAD_TOKEN)
            labels = F.pad(torch.tensor(labels).float(), pad=(0, pad_labels), value=PAD_TOKEN)

        del data
        return event_id, x, y, z, labels
