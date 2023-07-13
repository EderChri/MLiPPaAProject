from collections import namedtuple

import tqdm
import torch
import os
import math
import numpy as np
from torch.utils.data import DataLoader, random_split
from timeit import default_timer as timer
from dataset import TrajectoryDataset
from constants import *
from transformer import FittingTransformer
from utils import custom_collate, create_mask_src, create_output_pred_mask, check_size_compatibility
import wandb

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# training dataset
dataset = TrajectoryDataset(DATA_PATH, LABEL_PATH)
torch.manual_seed(7)  # for reproducibility
# split dataset indices into training and validation sets
full_len = len(dataset)
train_full_len = int(full_len * 0.8)
train_len = int(train_full_len * .9)
test_len = train_full_len - train_len
val_len = full_len - train_full_len


def train_eval_inner(model, t, train=True, optim=None):
    """
    Function used for both training and evaluation, used to avoid duplicate code
    :param model: the model to train
    :param t: instance of tqdm
    :param train: Flag indicating whether to perform training or evaluation
    :param optim: optimizer
    :return: loss
    """
    losses = 0.
    for i, data in t:
        # Get data
        event_id, x, src_len, labels, lbl_len = data
        x = x.to(DEVICE)
        if labels is not None:
            labels = labels.to(DEVICE)

        # Create masks for transformer
        src_mask, src_padding_mask = create_mask_src(x, DEVICE)
        # run model and predict
        pred = model(x, src_mask, src_padding_mask)
        if train:
            optim.zero_grad()
        # Create binary label mask
        mask = (labels != PAD_TOKEN).float()
        # Calculate the length of padding for the predictions, so no information is taken from the labels
        padding_len = np.round(np.divide(src_len, NR_DETECTORS))
        labels = labels * mask

        if DIMENSION == 2:
            pred = pred.transpose(0, 1)
            pred_mask = create_output_pred_mask(pred, padding_len)
            pred = pred * torch.tensor(pred_mask).float()
            # loss calculation
            loss = LOSS_FN(pred, labels)

        if DIMENSION == 3:
            pred = pred[0].transpose(0, 1), pred[1].transpose(0, 1)
            pred = torch.stack([pred[0], pred[1]])
            # For 3D data slice through the tensor array and mask each separately
            for slice_ind in range(pred.shape[0]):
                slice_mask = create_output_pred_mask(pred[slice_ind, :, :], padding_len)
                pred[slice_ind, :, :] = pred[slice_ind, :, :] * torch.tensor(slice_mask).float()
            pred = pred.transpose(0, 2)
            pred = pred.transpose(1, 0)
            # loss calculation
            loss = LOSS_FN(pred, labels)

        if train:
            loss.backward()  # compute gradients
            optim.step()  # backprop
        t.set_description("loss = %.8f" % loss.item())
        losses += loss.item()
    return losses


# training function (to be called per epoch)
def train_epoch(model, optim, disable_tqdm, batch_size, loader):
    """
    Train one epoch of the model
    :param model: model to train
    :param optim: optimizer
    :param disable_tqdm: Flag to indicate whether to deactivate tqdm output or not
    :param batch_size: batch_size used for training
    :param loader: data loader used
    :return: average loss
    """
    torch.set_grad_enabled(True)
    model.train()
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=disable_tqdm)
    losses = train_eval_inner(model, t, train=True, optim=optim)
    return losses / len(loader)


# test function
def evaluate(model, disable_tqdm, batch_size, loader):
    """
    Function to evaluate a model
    :param model: model to evaluate
    :param disable_tqdm: Flag to indicate whether to deactivate tqdm output or not
    :param batch_size: batch_size used for training
    :param loader: data loader used
    :return: average loss
    """
    model.eval()
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=disable_tqdm)

    with torch.no_grad():
        losses = train_eval_inner(model, t, train=False)

    return losses / len(loader)


def train(t_loader, v_loader, transformer, optimizer, batch_size, run_name="custom"):
    """
    Function to train a model or evaluate an already existing model
    :param t_loader: training data loader
    :param v_loader: validation data loader
    :param transformer: model to train/evaluate
    :param optimizer: optimizer
    :param batch_size: batch_size used
    :param run_name: name which the model should be saved as or loaded from
    :return:
    """
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    disable = False
    epoch, count = 0, 0

    print("Starting training...")

    for epoch in range(epoch, EPOCHS):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, disable, batch_size, t_loader)
        end_time = timer()
        val_loss = evaluate(transformer, disable, batch_size, v_loader)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, "
               f"Val loss: {val_loss:.8f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if RANDOM_SEARCH:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "dim": DIMENSION})

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("Saving best model with val_loss: {}".format(val_loss))
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'count': count,
            }, f"models/transformer_encoder_best_{DIMENSION}d_{run_name}")
            count = 0
        else:
            print("Saving last model with val_loss: {}".format(val_loss))
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'count': count,
            }, f"models/transformer_encoder_last_{DIMENSION}d_{run_name}")
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break


def fine_tune(cfg=None):
    """
    Wrapper function to allow running sweeps to train models or load and evaluate existing models
    :param cfg: namedTuple, only used for loading and evaluating an existing model
    :return:
    """
    if RANDOM_SEARCH:
        # only initialise when performing sweeping
        wandb.init()
        cfg = wandb.config
    # Split the data set and generate according data loaders
    train_set_full, val_set, = random_split(dataset, [train_full_len, val_len],
                                            generator=torch.Generator().manual_seed(7))
    train_set, test_set = random_split(train_set_full, [train_len, test_len],
                                       generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                              num_workers=1, shuffle=True, collate_fn=custom_collate)
    valid_loader = DataLoader(val_set, batch_size=cfg.batch_size,
                              num_workers=1, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE,
                             num_workers=1, shuffle=False, collate_fn=custom_collate)
    # Transformer model
    transformer = FittingTransformer(num_encoder_layers=cfg.num_encoder_layers,
                                     d_model=cfg.d_model,
                                     n_head=cfg.head,
                                     input_size=3,
                                     output_size=MAX_NR_TRACKS,
                                     dim_feedforward=cfg.dim_feedforward,
                                     dropout=cfg.dropout)
    transformer = transformer.to(DEVICE)
    print(transformer)

    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    # loss and optimiser
    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg.lr)

    if TRAIN:
        if RANDOM_SEARCH:
            # Save the model with the name according to the wandb run
            train(train_loader, valid_loader, transformer, optimizer, cfg.batch_size, wandb.run.name)
        else:
            train(train_loader, valid_loader, transformer, optimizer, cfg.batch_size)
    else:
        # Evaluate an existing model
        disable = False
        print("Loading saved model...")
        if MODEL_NAME.split('_')[-2] != f"{DIMENSION}d":
            print(
                "Model not trained for correct dimension of data! Try another model or change DIMENSION in constants.py"
            )
            return
        if not os.path.exists(f"models/{MODEL_NAME}"):
            print("Model does not exist!")
            return
        checkpoint = torch.load(f"models/{MODEL_NAME}")
        # Check if saved model is compatible with defined hyperparameters set in constants.py
        if not check_size_compatibility(checkpoint['model_state_dict'], transformer.state_dict()):
            return
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        print(f"Number of epochs trained: {epoch}")
        print(f"Test set MSE: {evaluate(transformer, disable, cfg.batch_size, test_loader)}")


if __name__ == '__main__':
    Config = namedtuple("Config",
                        ["batch_size", "num_encoder_layers", "d_model", "head", "dim_feedforward", "lr", "dropout"])
    cfg = Config(batch_size=BATCH_SIZE, num_encoder_layers=ENCODER_LAYERS,
                 d_model=D_MODEL, head=HEAD, dim_feedforward=DIM_FEEDFORWARD, lr=0.001, dropout=DROPOUT)
    if RANDOM_SEARCH:
        if not TRAIN:
            print("If RANDOM_SEARCH is True, TRAIN must be True")
        else:
            wandb.login()
            sweep_id = wandb.sweep(sweep=SWEEP_CONFIGURATION, project="ml-in-particle-physics-and-astronomy")
            wandb.agent(sweep_id, count=10, function=fine_tune)
    else:
        fine_tune(cfg)
