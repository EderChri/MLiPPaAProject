"""
Dataset module.
"""
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from glob import glob
import pandas as pd

from constants import DIMENSION, NR_DETECTORS


class Point:

    def __int__(self, coordinates, track):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = None
        if len(coordinates) == 3:
            self.z = coordinates[2]
        self.track = track


class TrajectoryDataset(Dataset):

    def __init__(self, root, labels, normalize=False, shuffle=False, to_tensor=True):
        self.root = root
        self.labels = pd.read_csv(labels, header=None)
        self.data = pd.read_csv(self.root, header=None)
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
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

        labels = event_labels[1::2]
        x = data[1::DIMENSION+1]
        y = data[2::DIMENSION+1]
        z = None
        if DIMENSION == 3:
            z = data[3::DIMENSION+1]
        if DIMENSION == 2:
            z = [0] * len(x)
        tracks = [track for track in range(int(data[-1])+1)]

        # normalise
        if self.normalize:
            raise NotImplementedError()

        # convert to tensors
        if self.to_tensor:
            x = torch.tensor(x).float()
            y = torch.tensor(y).float()
            z = torch.tensor(z).float()
            tracks = torch.tensor(tracks).int()
            labels = torch.tensor(labels).float()

        del data
        return event_id, x, y, z, tracks, labels
