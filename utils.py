import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from constants import PAD_TOKEN, PADDING_LEN_INPUT, PADDING_LEN_LBL, DIMENSION


def cart2cyl(x, y, z=None):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi, z) if z is not None else (rho, phi)


def sort_by_angle(cartesian_coords):
    dist_coords = np.array(cartesian_coords)
    distances = np.round(np.linalg.norm(dist_coords, axis=1))
    # Sort first by rho, round the rho, then sort by phi (sorting by the angle on decoder)
    cylindrical_coords = [cart2cyl(*coord) for coord in cartesian_coords]
    sorted_indices = np.lexsort((list(zip(*cylindrical_coords))[1], distances))
    sorted_cartesian_coords = [cartesian_coords[i] for i in sorted_indices]
    return sorted_cartesian_coords


def earth_mover_distance(y_true, y_pred):
    distance = torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1))
    return torch.mean(torch.mean(distance, dim=tuple(range(1, distance.ndim))))


def custom_collate(batch):
    event_ids = []
    xs, ys, zs = [], [], []
    labels = []
    labels_pad, lbl_lens = None, None

    for b in batch:
        event_ids.append(b[0])
        xs.append(b[1])
        ys.append(b[2])
        zs.append(b[3])
        if b[4] is not None:
            labels.append(b[4])

    x_lens = [len(val) for val in xs]
    # In case this is not a test data set
    if len(labels) > 0:
        lbl_lens = [len(lbl) for lbl in labels]
        if DIMENSION == 2:
            labels[0] = nn.ConstantPad1d((0, PADDING_LEN_LBL - labels[0].shape[0]), PAD_TOKEN)(labels[0])
        if DIMENSION == 3:
            labels[0] = nn.ConstantPad2d((0, 0, 0, PADDING_LEN_LBL - labels[0].shape[0]), PAD_TOKEN)(labels[0])
        labels_pad = pad_sequence(labels, batch_first=False, padding_value=PAD_TOKEN)
    # Convert the lists to tensors, except for the event_id since it might not be a tensor
    xs[0] = nn.ConstantPad1d((0, PADDING_LEN_INPUT - xs[0].shape[0]), PAD_TOKEN)(xs[0])
    ys[0] = nn.ConstantPad1d((0, PADDING_LEN_INPUT - ys[0].shape[0]), PAD_TOKEN)(ys[0])
    zs[0] = nn.ConstantPad1d((0, PADDING_LEN_INPUT - zs[0].shape[0]), PAD_TOKEN)(zs[0])

    xs_pad = pad_sequence(xs, batch_first=False, padding_value=PAD_TOKEN)
    ys_pad = pad_sequence(ys, batch_first=False, padding_value=PAD_TOKEN)
    zs_pad = pad_sequence(zs, batch_first=False, padding_value=PAD_TOKEN)
    x = torch.stack((xs_pad, ys_pad, zs_pad), dim=1)

    # Return the final processed batch
    return event_ids, x.transpose(1, 2), x_lens, labels_pad, lbl_lens


def load_variable_len_data(path):

    # adapted from
    # https://stackoverflow.com/questions/27020216/import-csv-with-different-number-of-columns-per-row-using-pandas
    with open(path, 'r') as f:
        col_count = [len(l.split(",")) for l in f.readlines()]

    column_names = [i for i in range(0, max(col_count))]

    data = pd.read_csv(path, header=None, delimiter=",", names=column_names)
    return data

def create_mask_src(src, device):
    src_seq_len = src.shape[0]
    padding_vector = torch.full((src_seq_len,), PAD_TOKEN)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src.transpose(0, 2) == padding_vector).all(dim=0)

    return src_mask, src_padding_mask


def create_output_pred_mask(tensor, indices):
    indices_arr = np.array(indices)
    row_indices = np.arange(tensor.shape[1])[:, np.newaxis]
    col_indices = np.arange(tensor.shape[0])
    mask = col_indices < indices_arr[row_indices]
    return mask.T
