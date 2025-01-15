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
from dataset import TripletData
from loss import TripletLoss
from model.backbones import backbones
from model.net import EmbeddingNet, TripletNet
from model.utils import count_trainable_params
from eval import Metric

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


def eval_epoch(val_loader, model, device):
    model.eval()
    eval_time = time.time()
    embeddings = []
    labels = []

    for data, target in val_loader:
        data = data.to(device)
        with torch.no_grad():
            if type(model) is TripletNet:
                embedding = model.embedding_net(data)
            else:
                embedding = model(data)
            
        embeddings.extend(embedding.cpu())
        labels.extend(target)
    
    metric = Metric(np.array(embeddings), np.array(labels))
    score = metric.silhouette()

    eval_time = time.time() - eval_time
    return score, eval_time


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
    eval_interval=5,
    metrics=[],
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
    best_score = -float('inf')

    if weights_folder:
        best_path = os.path.join(weights_folder, "best.pth")
        if os.path.exists(best_path):
            best_model = model.to(device)
            best_model.load_state_dict(torch.load(best_path, map_location=torch.device(device)))
            best_score, _ = eval_epoch(val_loader, best_model, device)

    for epoch in range(n_epochs):
        train_loss, train_epoch_time, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, device, log_interval, metrics
        )
        scheduler.step()

        message = "Epoch: {}/{}. Train set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, train_loss
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        message += "\nEpoch time: {:.4f} s".format(train_epoch_time)

        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        print(message)

        if (epoch + 1) % eval_interval == 0:
            score, eval_time = eval_epoch(val_loader, model, device)
            message = "Epoch: {}/{}. Validation set: Evaluation score: {:.4f}".format(
                epoch + 1, n_epochs, score
            )

            message += "\nEpoch time: {:.4f} s".format(eval_time)


            if weights_folder and score > best_score:
                save_path = os.path.join(weights_folder, "best.pth")
                torch.save(model.state_dict(), save_path)
                message += f"\nSave best model to {save_path}"

            print(message)

        if weights_folder:
            last_path = os.path.join(weights_folder, "last.pth")
            torch.save(model.state_dict(), last_path)

        if history_path:
            history["train_loss"].append(train_loss)
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

    train = datasets.ImageFolder(
        cfg.TRAIN_DATASET_DIR,
        transform=transform,
    )
    val = datasets.ImageFolder(
        cfg.VAL_DATASET_DIR,
        transform=transform,
    )

    triplet_train_dataset = TripletData(train)
    val_dataset = val  # just eval, do not need to build triplet

    print(
        f"Lengths of triplet train, val datasets: {len(triplet_train_dataset)}, {len(val_dataset)}"
    )

    triplet_train_loader = DataLoader(
        triplet_train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )

    device = f"cuda:{cfg.DEVICE_ID}" if torch.cuda.is_available() else "cpu"

    backbone = backbones[cfg.BACKBONE_NAME]
    weights_folder = os.path.join(f"./weights/{cfg.BACKBONE_NAME}", cfg.MODEL_SAVE_NAME)
    os.makedirs(weights_folder, exist_ok=True)
    save_path = os.path.join(weights_folder, "last.pth")

    embedding_net = EmbeddingNet(backbone["model"])
    model = TripletNet(embedding_net)
    model = model.to(device)

    if cfg.PRETRAINED:
        model.load_state_dict(torch.load(cfg.PRETRAINED, map_location=torch.device(device)))
    elif os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=torch.device(device)))

    # print(f"Model parameters: {count_trainable_params(model)}")

    loss_fn = TripletLoss(cfg.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
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
        history = {"train_loss": [], "epochs_run": start_epoch}

    print(f"Train ...")
    fit(
        triplet_train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        cfg.NUM_EPOCHS,
        device,
        log_interval,
        history,
        weights_folder=weights_folder,
        history_path=history_file,
    )


if __name__ == "__main__":
    if not cfg.TEST_ONLY:
        main()
