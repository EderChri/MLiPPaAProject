import numpy as np
import pandas as pd
import torch

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
    tracks = []
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

    # Return the final processed batch
    return event_ids, xs, ys, zs, labels


def load_variable_len_data(path):
    data = pd.read_fwf(path, header=None)
    return data[0].str.split(',', expand=True)


def create_mask_src(src):
    masks = []
    for sample in src:
        mask = [0 if token == PAD_TOKEN else 1 for token in sample]
        masks.append(mask)
    return masks
