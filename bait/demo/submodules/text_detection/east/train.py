import argparse
import glob
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import lr_scheduler

from dataset import custom_dataset
from loss import Loss
from model import EAST
from utils_td.file import get_weights

# import warnings
# warnings.filterwarnings('ignore')


def get_datasets(
    train_img_path, train_gt_path, val_img_path, val_gt_path, batch_size, num_workers
):
    train_set = custom_dataset(train_img_path, train_gt_path)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_set = custom_dataset(val_img_path, val_gt_path)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train(
    train_loader,
    criterion,
    model,
    device,
    optimizer,
    epoch,
    epoch_iter,
    mini_batch_size,
    log_interval,
):
    model.train()
    epoch_loss = []
    epoch_time = time.time()

    for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
        start_time = time.time()
        img, gt_score, gt_geo, ignored_map = (
            img.to(device),
            gt_score.to(device),
            gt_geo.to(device),
            ignored_map.to(device),
        )

        pred_score, pred_geo = model(img)
        loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
        loss_item = loss.item()
        epoch_loss.append(loss_item)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % log_interval == 0:
            print(
                "Train: Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}".format(
                    epoch + 1,
                    epoch_iter,
                    i + 1,
                    mini_batch_size,
                    time.time() - start_time,
                    loss_item,
                )
            )
    epoch_loss = np.array(epoch_loss)
    epoch_loss = np.mean(epoch_loss[epoch_loss > 0])
    epoch_time = time.time() - epoch_time

    return epoch_loss, epoch_time


def test(
    val_loader,
    criterion,
    model,
    device,
    epoch,
    epoch_iter,
    mini_batch_size,
    log_interval,
):
    with torch.no_grad():
        model.eval()
        epoch_loss = []
        epoch_time = time.time()

        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(val_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = (
                img.to(device),
                gt_score.to(device),
                gt_geo.to(device),
                ignored_map.to(device),
            )

            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            loss_item = loss.item()
            epoch_loss.append(loss_item)

            if (i + 1) % log_interval == 0:
                print(
                    "Test: Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}".format(
                        epoch + 1,
                        epoch_iter,
                        i + 1,
                        mini_batch_size,
                        time.time() - start_time,
                        loss_item,
                    )
                )

    epoch_loss = np.array(epoch_loss)
    epoch_loss = np.mean(epoch_loss[epoch_loss > 0])
    epoch_time = time.time() - epoch_time

    return epoch_loss, epoch_time


def get_checkpoint(checkpoint_file, model, optimizer, device, pretrained=False):
    start_epoch = 0
    train_hist = []
    val_hist = []

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=device)

        start_epoch = checkpoint["epochs_run"]
        train_hist = list(checkpoint["train_loss_history"])
        val_hist = list(checkpoint["val_loss_history"])
        model.load_state_dict(checkpoint["model_state_dict"])
        state_dict = checkpoint["optimizer_state_dict"]
        optimizer.load_state_dict(state_dict)
    else:
        if pretrained:
            print("Using pretrained east-vgg16.path")
            pretrained_weight_path = get_weights(
                "/mnt/data/text_detection/pretrained/east-vgg16.pth"
            )
            model.load_state_dict(
                torch.load(pretrained_weight_path, map_location=device)
            )

    return start_epoch, model, optimizer, train_hist, val_hist


def fit(
    train_img_path,
    train_gt_path,
    val_img_path,
    val_gt_path,
    pths_path,
    batch_size,
    num_workers,
    epoch_iter,
    log_interval,
    pretrained,
):
    torch.backends.cudnn.benchmark = True

    train_file_num = len(os.listdir(train_img_path))
    val_file_num = len(os.listdir(val_img_path))

    train_loader, val_loader = get_datasets(
        train_img_path,
        train_gt_path,
        val_img_path,
        val_gt_path,
        batch_size,
        num_workers,
    )

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[epoch_iter // 2], gamma=0.1
    )

    checkpoint_file = get_latest_weights(pths_path)
    # os.makedirs(pths_path, exist_ok=True)
    # checkpoint_file = "./checkpoint/Data_Cook/200.pth"  # continue from best

    start_epoch, model, optimizer, train_hist, val_hist = get_checkpoint(
        checkpoint_file, model, optimizer, device, pretrained
    )

    data_parallel = False
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     data_parallel = True

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, epoch_iter):
        scheduler.step()
        train_loss, train_time = train(
            train_loader,
            criterion,
            model,
            device,
            optimizer,
            epoch,
            epoch_iter,
            int(train_file_num / batch_size),
            log_interval,
        )

        val_loss, val_time = test(
            val_loader,
            criterion,
            model,
            device,
            epoch,
            epoch_iter,
            int(val_file_num / batch_size),
            log_interval,
        )

        print(
            "train_loss is {:.8f}, train_time is {:.8f}".format(train_loss, train_time)
        )

        print("val_loss is {:.8f}, val_time is {:.8f}".format(val_loss, val_time))

        print(time.asctime(time.localtime(time.time())))
        print("=" * 50)

        model_state_dict = (
            model.module.state_dict() if data_parallel else model.state_dict()
        )
        optimizer_state_dict = optimizer.state_dict()

        if train_hist is None or val_hist is None:
            train_hist = []
            val_hist = []

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        torch.save(
            {
                "epochs_run": epoch + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "train_loss_history": train_hist,
                "val_loss_history": val_hist,
            },
            os.path.join(pths_path, "{}.pth".format(epoch + 1)),
        )


def get_latest_weights(weights_folder: str):
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder, exist_ok=True)
        return None

    weights = glob.glob(os.path.join(weights_folder, "*.pth"))
    if not weights:
        return None

    return max(weights, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default="./data_cook")
    parser.add_argument("--ckpt-path", type=str, default="./checkpoint/data_cook")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epoch-iter", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Using pretrained EAST model",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    fit(
        os.path.join(args.data_path, "train_img"),
        os.path.join(args.data_path, "train_gt"),
        os.path.join(args.data_path, "test_img"),
        os.path.join(args.data_path, "test_gt"),
        args.ckpt_path,
        args.batch_size,
        args.num_workers,
        args.epoch_iter,
        args.log_interval,
        args.pretrained,
    )
