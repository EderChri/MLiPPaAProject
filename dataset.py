"""
Dataset module.
"""
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import *
from utils import sort_by_angle, load_variable_len_data


class TrajectoryDataset(Dataset):
    """
    This is the dataset module that provides loading and preprocessing of the trajectory data set
    """

    def __init__(self, root, labels, shuffle=False, to_tensor=True, test=False):
        """
        Constructor that initialises the data set either as training/validation data set or as test data set
        :param root: Path to the main data set, expects csv
        :param labels: Path to the label data set, expects csv
        :param shuffle: Boolean that decides whether data set should be shuffeled or not
        :param to_tensor: Boolean used to convert data to tensors, currently only True is supported
        :param test: Flag indicating whether the data set is intended as test data set or not
        """
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
        self.to_tensor = to_tensor

    def __len__(self):
        """
        Method returning the total number of events
        :return: total number of events, equal to the length of the data set
        """
        return self.total_events

    def __getitem__(self, idx):
        """
        Method used by data loader to load each item
        :param idx: Index of the item to be loaded
        :return: preprocessed tupel of event_id, x, y, z, labels
        """
        # load event
        labels = None
        # Load variable length data into data frame
        data = self.data.iloc[[idx]].values.tolist()[0]
        event_id = int(data[0])
        if not self.test:
            # Load variable length labels into data frame
            event_labels = self.labels.iloc[[event_id]].values.tolist()[0]
            # Ignore the event_id in each line and only take the track parameter not the track_id
            labels = event_labels[2::2]
            if DIMENSION == 2:
                labels = [float(value) for value in labels if not math.isnan(value)]
            # In case of 3D data convert "track_param1;track_param2" in list with tuples (track_param1, track_param2)
            if DIMENSION == 3:
                tmp_lbls = []
                for angle_list in labels:
                    if not isinstance(angle_list, str):
                        continue
                    angle_list = angle_list.split(';')
                    tmp_lbls.append((float(angle_list[0]), float(angle_list[1])))
                labels = tmp_lbls

            # Sort the labels from -pi to pi as the points will be sorted later on as well
            labels = np.sort(labels)

        # Ignore the even_id and only take the x values in the data frame
        x = data[1::DIMENSION + 1]
        # Ignore the even_id and first x value and only take the x values in the data frame
        y = data[2::DIMENSION + 1]
        z = None
        x = [float(value) for value in x if not math.isnan(value)]
        y = [float(value) for value in y if not math.isnan(value)]
        if DIMENSION == 3:
            # Ignore the even_id and first x and y values and only take the x values in the data frame
            z = data[3::DIMENSION + 1]
            z = [float(value) for value in z if not math.isnan(value)]
        if DIMENSION == 2:
            # In case of 2D data just pad the third dimension
            z = [PAD_TOKEN] * len(x)

        convert_list = []
        for i in range(len(x)):
            if DIMENSION == 3:
                convert_list.append((x[i], y[i], z[i]))
            if DIMENSION == 2:
                convert_list.append((x[i], y[i]))

        # Sort the data from -pi to pi to avoid the model learning the coordinate position for predicting the track
        # parameter
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
