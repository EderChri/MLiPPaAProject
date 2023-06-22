import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from constants import PAD_TOKEN


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
    xs = []
    ys = []
    zs = []
    labels = []

    for b in batch:
        # Assuming z (b[3]) is the variable that can be None
        if b[3] is not None:
            event_ids.append(b[0])
            xs.append(b[1])
            ys.append(b[2])
            zs.append(b[3])
            labels.append(b[4])

    # Convert the lists to tensors, except for the event_id since it might not be a tensor
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    zs = torch.stack(zs)
    labels = torch.stack(labels)
    x = torch.stack((xs,ys, zs),dim=1)

    x_lens = [len(val) for val in x]
    y_lens = [len(y) for y in labels]

    x_pad = pad_sequence(x, batch_first=False, padding_value=PAD_TOKEN)
    labels_pad = pad_sequence(labels, batch_first=False, padding_value=PAD_TOKEN)
    # Return the final processed batch
    return event_ids, x_pad, labels_pad, x_lens, y_lens


def load_variable_len_data(path):
    data = pd.read_fwf(path, header=None)
    return data[0].str.split(',', expand=True)


def create_mask_src(src, device):
    src_seq_len = src.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src[:, :, 0] == PAD_TOKEN).transpose(0, 1)

    return src_mask, src_padding_mask
