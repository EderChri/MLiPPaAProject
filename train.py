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
from utils import custom_collate, earth_mover_distance, create_mask_src, create_output_pred_mask
import wandb

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# training dataset
dataset = TrajectoryDataset(DATA_PATH, LABEL_PATH)

def train_eval_inner(model, t, train=True, optim=None):
    losses = 0.
    for i, data in t:
        event_id, x, src_len, labels, lbl_len = data
        x = x.to(DEVICE)
        if labels is not None:
            labels = labels.to(DEVICE)

        src_mask, src_padding_mask = create_mask_src(x, DEVICE)
        # run model
        pred = model(x, src_mask, src_padding_mask)
        if train:
            optim.zero_grad()
        mask = (labels != PAD_TOKEN).float()
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
    torch.set_grad_enabled(True)
    model.train()
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=disable_tqdm)
    losses = train_eval_inner(model, t, train=True, optim=optim)
    return losses / len(loader)


# test function
def evaluate(model, disable_tqdm, batch_size, loader):
    model.eval()
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=disable_tqdm)

    with torch.no_grad():
        losses = train_eval_inner(model, t, train=False)

    return losses / len(loader)


def predict(model, loader, disable_tqdm=False):
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


def train(t_loader, v_loader, transformer, optimizer, batch_size):
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    disable, load = False, False
    epoch, count = 0, 0

    if load:
        print("Loading saved model...")
        checkpoint = torch.load(f"models/transformer_encoder_last_{DIMENSION}d")
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        min_val_loss = min(val_losses)
        count = checkpoint['count']
        print(epoch, val_losses)
    else:
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
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

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
            }, f"models/transformer_encoder_best_{DIMENSION}d")
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
            }, f"models/transformer_encoder_last_{DIMENSION}d")
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break


def fine_tune():
    wandb.init()
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
    train_loader = DataLoader(train_set, batch_size=wandb.config.batch_size,
                              num_workers=1, shuffle=True, collate_fn=custom_collate)
    valid_loader = DataLoader(val_set, batch_size=wandb.config.batch_size,
                              num_workers=1, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE,
                             num_workers=1, shuffle=False, collate_fn=custom_collate)
    # Transformer model
    transformer = FittingTransformer(num_encoder_layers=wandb.config.num_encoder_layers,
                                     d_model=wandb.config.d_model,
                                     n_head=wandb.config.head,
                                     input_size=3,
                                     output_size=20,
                                     dim_feedforward=wandb.config.dim_feedforward,
                                     dropout=wandb.config.dropout)
    transformer = transformer.to(DEVICE)
    print(transformer)

    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    # loss and optimiser
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    if TRAIN:
        train(train_loader, valid_loader, transformer, optimizer, wandb.config.batch_size)
    else:
        disable, load = False, False
        epoch, count = 0, 0
        print("Loading saved model...")
        checkpoint = torch.load(f"models/transformer_encoder_last_{DIMENSION}d")
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        min_val_loss = min(val_losses)
        count = checkpoint['count']
        print(f"Number of epochs trained: {epoch}")
        print(f"MSE: {evaluate(transformer, disable, wandb.config.batch_size, test_loader)}")


if __name__ == '__main__':
    wandb.login()
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIGURATION, project="ml-in-particle-physics-and-astronomy")
    wandb.agent(sweep_id, count=35, function=fine_tune)
