import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from constants import PAD_TOKEN, PADDING_LEN_INPUT, PADDING_LEN_LBL, DIMENSION


def cart2cyl(x, y, z=None):
    """
    Function to convert cartesian coordinates to cylindrical
    :param x: x coordinate of point
    :param y: y coordinate of point
    :param z: z coordinate of point
    :return: Tuple containing the cylindrical coordinates
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi, z) if z is not None else (rho, phi)


def sort_by_angle(cartesian_coords):
    """
    Sorts cartesian coordinates by angle, converting them to cylindrical and back
    :param cartesian_coords: numpy array of cartesian coordinates
    :return: Returns the coordinates sorted from -pi to pi
    """
    dist_coords = np.array(cartesian_coords)
    distances = np.round(np.linalg.norm(dist_coords, axis=1))
    # Sort first by rho, round the rho, then sort by phi (sorting by the angle on decoder)
    cylindrical_coords = [cart2cyl(*coord) for coord in cartesian_coords]
    sorted_indices = np.lexsort((list(zip(*cylindrical_coords))[1], distances))
    sorted_cartesian_coords = [cartesian_coords[i] for i in sorted_indices]
    return sorted_cartesian_coords


def custom_collate(batch):
    """
    Custom collate function that adds padding when needed
    :param batch: batch of data
    :return: Tuple of tensors of padded event_ids, stacked points, length of the points, labels, and length of labels
    """
    event_ids = []
    xs, ys, zs = [], [], []
    labels = []
    labels_pad, lbl_lens = None, None

    # get coordinates and labels out of points and labels batch
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
    """
    Function to load variable length data into a pandas data frame
    adapted from
    https://stackoverflow.com/questions/27020216/import-csv-with-different-number-of-columns-per-row-using-pandas
    :param path: location of the csv file
    :return: dataframe
    """
    #
    with open(path, 'r') as f:
        col_count = [len(l.split(",")) for l in f.readlines()]

    column_names = [i for i in range(0, max(col_count))]

    data = pd.read_csv(path, header=None, delimiter=",", names=column_names)
    return data


def create_mask_src(src, device):
    """
    Creates src and src padding mask for transformer
    :param src: the whole patch of data ready for prediction
    :param device: the device the data is on
    :return: Tuple of src_mask and src_padding_mask
    """
    src_seq_len = src.shape[0]
    padding_vector = torch.full((src_seq_len,), PAD_TOKEN)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src.transpose(0, 2) == padding_vector).all(dim=0)

    return src_mask, src_padding_mask


def create_output_pred_mask(tensor, indices):
    """
    Creates mask for the predictions
    :param tensor: predictions as received from the transformer model
    :param indices: the indices where padding had happened in the input
    :return: Mask for the output
    """
    indices_arr = np.array(indices)
    row_indices = np.arange(tensor.shape[1])[:, np.newaxis]
    col_indices = np.arange(tensor.shape[0])
    mask = col_indices < indices_arr[row_indices]
    return mask.T


def check_size_compatibility(check_point_dict, model_dict):
    """
    Function to check whether the current models model dictionary is compatible with the one in the checkpoint
    :param check_point_dict: model dictionary from a saved model
    :param model_dict: current model's dictionary
    :return: Boolean, flag indicating if both models are compatible or not
    """
    compatible = True
    for key in check_point_dict.keys():
        if key in model_dict:
            # Check the size of each parameter
            if model_dict[key].shape != check_point_dict[key].shape:
                print(f"Mismatch found for {key}")
                print(f"Model size: {model_dict[key].shape}, Checkpoint size: {check_point_dict[key].shape}")
                compatible = False
        else:
            print(f"{key} not found in the current model.")
    if not compatible:
        print("=" * 50)
        print("Model hyperparameter sizes mismatch with saved models sizes")
        print("Please fix incompatibilities by changing parameters in constants.py")
    return compatible
