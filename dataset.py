"""
Dataset module.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import *
from utils import sort_by_angle, load_variable_len_data


class TrajectoryDataset(Dataset):

    def __init__(self, root, labels, normalize=False, shuffle=False, to_tensor=True, test=False):
        self.root = root
        self.test = test
        if not self.test:
            self.labels = load_variable_len_data(labels)
        else:
            self.labels = None
        self.data = load_variable_len_data(self.root)

        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.total_events = len(self.data)
        self.normalize = normalize
        self.to_tensor = to_tensor

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        # load event
        labels = None
        data = self.data.iloc[[idx]].values.tolist()[0]
        event_id = int(data[0])
        if not self.test:
            event_labels = self.labels.iloc[[event_id]].values.tolist()[0]
            labels = event_labels[2::2]
            if DIMENSION == 2:
                labels = [float(value) for value in labels if value is not None]
            if DIMENSION == 3:
                tmp_lbls = []
                for angle_list in labels:
                    if angle_list is None:
                        continue
                    angle_list = angle_list.split(';')
                    tmp_lbls.append((float(angle_list[0]), float(angle_list[1])))
                labels = tmp_lbls

            labels = np.sort(labels)

        x = data[1::DIMENSION + 1]
        y = data[2::DIMENSION + 1]
        z = None
        x = [float(value) for value in x if value is not None]
        y = [float(value) for value in y if value is not None]
        if DIMENSION == 3:
            z = data[3::DIMENSION + 1]
            z = [float(value) for value in z if value is not None]
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

        if DIMENSION == 2:
            x, y = zip(*sorted_coords)
        else:
            x, y, z = zip(*sorted_coords)

        # convert to tensors
        if self.to_tensor:
            x = torch.tensor(x).float()
            y = torch.tensor(y).float()
            z = torch.tensor(z).float()
            if not self.test:
                labels = torch.tensor(labels).float()

        del data
        return event_id, x, y, z, labels
