import os
from collections import namedtuple
import numpy as np
import torch
import math
from torch.utils.data import random_split, DataLoader
import tqdm
import constants
from constants import *
from dataset import TrajectoryDataset
from transformer import FittingTransformer
from utils import create_mask_src, create_output_pred_mask, custom_collate

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_pred(data_path, label_path, cfg, model_path, output_path):
    """
    Function to set up the prediction run, generate dataloader and run the predictions
    :param data_path: path to the data
    :param label_path: path to the labels
    :param cfg: config for the transformer model to load
    :param model_path: path to the transformer model
    :param output_path: name of the output file
    :return:
    """
    dataset = TrajectoryDataset(data_path, label_path)
    torch.manual_seed(7)  # for reproducibility
    # split dataset into training and validation sets
    full_len = len(dataset)
    train_full_len = int(full_len * 0.8)
    train_len = int(train_full_len * .9)
    test_len = train_full_len - train_len
    val_len = full_len - train_full_len
    train_set_full, val_set, = random_split(dataset, [train_full_len, val_len],
                                            generator=torch.Generator().manual_seed(7))
    train_set, test_set = random_split(train_set_full, [train_len, test_len],
                                       generator=torch.Generator().manual_seed(7))
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE,
                             num_workers=1, shuffle=False, collate_fn=custom_collate)
    # Transformer model
    transformer = FittingTransformer(num_encoder_layers=cfg.num_encoder_layers,
                                     d_model=cfg.d_model,
                                     n_head=cfg.head,
                                     input_size=3,
                                     output_size=20,
                                     dim_feedforward=cfg.dim_feedforward,
                                     dropout=cfg.dropout)
    checkpoint = torch.load(f"models/{model_path}")
    transformer.load_state_dict(checkpoint['model_state_dict'])
    print_data(predict(transformer, test_loader), output=output_path)
    print("Successfully predicted")


def predict(model, loader, disable_tqdm=False):
    """
    Function to predict using a trained model
    :param model: trained model
    :param loader: data loader for predictions
    :param disable_tqdm: Flag to indicate whether tqdm output should be suppressed or not
    :return: Dictionary of predictions, keys are events, values are tensors of predictions
    """
    torch.set_grad_enabled(False)
    model.eval()
    n_batches = int(math.ceil(len(loader.dataset) / TEST_BATCH_SIZE))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=disable_tqdm)
    predictions = {}

    for i, data in t:
        event_id, x, src_len, _, _ = data
        x = x.to(DEVICE)

        src_mask, src_padding_mask = create_mask_src(x, DEVICE)
        # run model
        pred = model(x, src_mask, src_padding_mask)
        padding_len = np.round(np.divide(src_len, 5))
        if DIMENSION == 2:
            pred = pred.transpose(0, 1)
            pred_mask = create_output_pred_mask(pred, padding_len)
            pred = pred * torch.tensor(pred_mask).float()
        if DIMENSION == 3:
            pred = pred[0].transpose(0, 1), pred[1].transpose(0, 1)
            pred = torch.stack([pred[0], pred[1]])
            for slice_ind in range(pred.shape[0]):
                slice_mask = create_output_pred_mask(pred[slice_ind, :, :], padding_len)
                pred[slice_ind, :, :] = pred[slice_ind, :, :] * torch.tensor(slice_mask).float()
            pred = pred.transpose(0, 2)
            pred = pred.transpose(1, 0)

        # Append predictions to the list
        for i, e_id in enumerate(event_id):
            predictions[e_id] = pred[:, i]

    return predictions


def print_data(data, output=f"prediction_{constants.DIMENSION}d.txt"):
    """
    Helper function to print the predictions
    :param data: data to print
    :param output: file to print predictions to
    :return:
    """
    with open(output, "w") as file:
        for event_id, predictions in data.items():
            # Remove all padding tokens predicted
            pred_lists = predictions[predictions.nonzero()].tolist()
            flat_pred_list = [pred for sublist in pred_lists for pred in sublist]
            value = ','.join(map(str, flat_pred_list))
            file.write(f"{event_id},{value}\n")


if __name__ == '__main__':

    if DIMENSION == 2:
        data_path, label_path = "output_2d_baseline.txt", "parameter_2d_baseline.txt"
        model_path = "transformer_encoder_best_2d_baseline"
        Config = namedtuple("Config",
                            ["batch_size", "num_encoder_layers", "d_model", "head", "dim_feedforward", "lr", "dropout"])
        cfg = Config(batch_size=BATCH_SIZE, num_encoder_layers=ENCODER_LAYERS,
                     d_model=D_MODEL, head=HEAD, dim_feedforward=DIM_FEEDFORWARD, lr=0.001, dropout=DROPOUT)
        run_pred(data_path, label_path, cfg, model_path, "predictions/prediction_2d_baseline.txt")

        data_path, label_path = "output_2d.txt", "parameter_2d.txt"
        model_path = "transformer_encoder_best_2d_kind-sweep-5"
        Config = namedtuple("Config",
                            ["batch_size", "num_encoder_layers", "d_model", "head", "dim_feedforward", "lr", "dropout"])
        cfg = Config(batch_size=128, num_encoder_layers=4,
                     d_model=32, head=HEAD, dim_feedforward=1, lr=0.0003761, dropout=0.1)
        run_pred(data_path, label_path, cfg, model_path, "predictions/prediction_2d_best.txt")
    if DIMENSION == 3:
        data_path, label_path = "output_3d_baseline.txt", "parameter_3d_baseline.txt"
        model_path = "transformer_encoder_best_3d_baseline"
        Config = namedtuple("Config",
                            ["batch_size", "num_encoder_layers", "d_model", "head", "dim_feedforward", "lr", "dropout"])
        cfg = Config(batch_size=BATCH_SIZE, num_encoder_layers=ENCODER_LAYERS,
                     d_model=D_MODEL, head=HEAD, dim_feedforward=DIM_FEEDFORWARD, lr=0.001, dropout=DROPOUT)
        run_pred(data_path, label_path, cfg, model_path, "predictions/prediction_3d_baseline.txt")

        data_path, label_path = "output_3d.txt", "parameter_3d.txt"
        model_path = "transformer_encoder_best_3d_revived-sweep-7"
        Config = namedtuple("Config",
                            ["batch_size", "num_encoder_layers", "d_model", "head", "dim_feedforward", "lr", "dropout"])
        cfg = Config(batch_size=256, num_encoder_layers=8,
                     d_model=32, head=2, dim_feedforward=2, lr=0.0001181, dropout=0.1)
        run_pred(data_path, label_path, cfg, model_path, "predictions/prediction_3d_best.txt")
