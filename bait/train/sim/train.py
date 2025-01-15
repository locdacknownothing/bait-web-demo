import argparse
import glob
import os
import pickle
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import AugmentedData, TripletData
from loss import TripletLoss
from model.backbones import backbones
from model.net import EmbeddingNet, EmbeddingNetL2, TripletNet
from model.utils import count_trainable_params

import config as cfg


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    total_losses = []
    epoch_time = time.time()
    losses = []
    log_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)

        target = target if len(target) > 0 else None
        if target is not None:
            target = target.to(device)

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        if loss > 0:
            losses.append(loss.item())
            total_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == log_interval - 1:
            message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.4f}".format(
                batch_idx * len(data[0]),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                np.mean(losses),
                time.time() - log_time,
            )
            for metric in metrics:
                message += "\t{}: {}".format(metric.name(), metric.value())

            print(message)
            losses = []
            log_time = time.time()

    epoch_time = time.time() - epoch_time
    return np.mean(total_losses), epoch_time, metrics


def test_epoch(val_loader, model, loss_fn, device, log_interval, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        model.eval()
        val_losses = []
        epoch_time = time.time()
        losses = []
        log_time = time.time()

        for batch_idx, (data, target) in enumerate(val_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            data = tuple(d.to(device) for d in data)

            target = target if len(target) > 0 else None
            if target is not None:
                target = target.to(device)

            outputs = model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = (
                loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            )
            if loss > 0:
                val_losses.append(loss.item())
                losses.append(loss.item())

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == log_interval - 1:
                message = "Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.4f}".format(
                    batch_idx * len(data[0]),
                    len(val_loader.dataset),
                    100.0 * batch_idx / len(val_loader),
                    np.mean(losses),
                    time.time() - log_time,
                )
                for metric in metrics:
                    message += "\t{}: {}".format(metric.name(), metric.value())

                print(message)
                losses = []
                log_time = time.time()

    epoch_time = time.time() - epoch_time
    return np.mean(val_losses), epoch_time, metrics


def fit(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    device,
    log_interval,
    history,
    metrics=[],
    start_epoch=0,
    weights_folder=None,
    history_path=None,
):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        optimizer.step()
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        train_loss, train_epoch_time, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, device, log_interval, metrics
        )
        scheduler.step()

        message = "Epoch: {}/{}. Train set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, train_loss
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        val_loss, val_epoch_time, metrics = test_epoch(
            val_loader, model, loss_fn, device, log_interval, metrics
        )

        message += "\nEpoch: {}/{}. Validation set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, val_loss
        )

        message += "\nEpoch time: {:.4f} s".format(train_epoch_time + val_epoch_time)

        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        print(message)

        # NOTE: save weight each 5-epoch interval
        if weights_folder and (epoch + 1) % 5 == 0:
            save_path = os.path.join(weights_folder, f"{epoch}.pth")
            torch.save(model, save_path)

        if history_path:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["epochs_run"] += 1
            
            with open(history_path, "wb") as f:
                pickle.dump(history, f)


def get_latest_weights(weights_folder):
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder, exist_ok=True)
        return None

    weights = glob.glob(os.path.join(weights_folder, "*.pth"))
    if not weights:
        return None

    return max(weights, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))


def main():
    target_shape = (224, 224)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(target_shape)]
    )
    # transform = backbone["weights"].transforms()

    train = datasets.ImageFolder(
        cfg.TRAIN_DATASET_DIR,
        transform=transform,
    )
    val = datasets.ImageFolder(
        cfg.VAL_DATASET_DIR,
        transform=transform,
    )

    # augmented_train = AugmentedData(train)
    # augmented_val = AugmentedData(val)

    triplet_train_dataset = TripletData(train)
    triplet_val_dataset = TripletData(val)

    print(
        f"Lengths of triplet train, val datasets: {len(triplet_train_dataset)}, {len(triplet_val_dataset)}"
    )

    triplet_train_loader = DataLoader(
        triplet_train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )
    triplet_val_loader = DataLoader(
        triplet_val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )

    device = f"cuda:{cfg.DEVICE_ID}" if torch.cuda.is_available() else "cpu"

    backbone = backbones[cfg.BACKBONE_NAME]
    weights_folder = os.path.join(f"./weights/{cfg.BACKBONE_NAME}", cfg.MODEL_SAVE_NAME)
    save_path = get_latest_weights(weights_folder)

    if cfg.PRETRAINED:
        model = torch.load(cfg.PRETRAINED, map_location=torch.device(device))
    elif save_path:
        model = torch.load(save_path, map_location=torch.device(device))
    else:
        embedding_net = EmbeddingNetL2(backbone["model"])
        model = TripletNet(embedding_net)
        model = model.to(device)

    print(f"Model parameters: {count_trainable_params(model)}")

    loss_fn = TripletLoss(cfg.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 100

    history_path = f"./history/{cfg.BACKBONE_NAME}/"
    try:
        history_file = os.path.join(history_path, cfg.HISTORY_SAVE_NAME)
        with open(history_file, "rb") as f:
            history = pickle.load(f)

        start_epoch = history["epochs_run"]
    except:
        os.makedirs(history_path, exist_ok=True)
        start_epoch = 0
        history = {"train_loss": [], "val_loss": [], "epochs_run": start_epoch}

    print(f"Train ...")
    print(f"Start training at epoch {start_epoch}")
    fit(
        triplet_train_loader,
        triplet_val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        cfg.NUM_EPOCHS,
        device,
        log_interval,
        history,
        start_epoch=start_epoch,
        weights_folder=weights_folder,
        history_path=history_file,
    )


def test_only():
    target_shape = (224, 224)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(target_shape)]
    )
    val = datasets.ImageFolder(
        cfg.VAL_DATASET_DIR,
        transform=transform,
    )
    triplet_val_dataset = TripletData(val)
    triplet_val_loader = DataLoader(
        triplet_val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )

    print(f"Lengths of triplet val dataset: {len(triplet_val_dataset)}")

    device = f"cuda:{cfg.DEVICE_ID}" if torch.cuda.is_available() else "cpu"
    loss_fn = TripletLoss(cfg.MARGIN)
    log_interval = 100
    model = torch.load(cfg.PRETRAINED, map_location=torch.device(device))

    print("Test ...")
    loss, epoch_time, _ = test_epoch(triplet_val_loader, model, loss_fn, device, log_interval, [])
    print(f"Loss: {loss:.4f}\tTime: {epoch_time:.4f}s")


# def parse_args():
#     parser = argparse.ArgumentParser()

#     # Add arguments
#     parser.add_argument(
#         "--backbone_name",
#         type=str,
#         default="mobilenet_v3_large",
#         help="Name of model's backbone",
#     )
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
#     parser.add_argument(
#         "--num_epochs", type=int, default=30, help="Number of training epochs"
#     )
#     parser.add_argument(
#         "--learning_rate", type=float, default=0.001, help="Learning rate"
#     )
#     parser.add_argument(
#         "--margin", type=float, default=1.0, help="Margin of triplet loss"
#     )

#     # Parse the arguments
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":
    if not cfg.TEST_ONLY:
        main()
    else:
        test_only()
