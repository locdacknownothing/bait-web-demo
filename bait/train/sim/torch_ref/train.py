import os
import time

import torch
import torchvision.transforms as transforms
from loss import TripletMarginLoss
from model import EmbeddingNet
from sampler import PKSampler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from custom_dataset import AugmentedData
import config as cfg


def train_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq):
    start_time = time.time()
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        samples, targets = data[0].to(device), data[1].to(device)

        embeddings = model(samples)

        loss, frac_pos_triplets = criterion(embeddings, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print(
                f"[{epoch:d}, {i:d}] | loss: {avg_loss:.4f} | % avg hard triplets: {avg_trip:.2f}%"
            )
            running_loss = 0
            running_frac_pos_triplets = 0

    print("train time: {:.4f} s".format(time.time() - start_time))


def find_best_threshold(dists, targets, device):
    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh.to(device)
        correct = torch.sum(predictions == targets.to(device)).item()
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = 100.0 * best_correct / dists.size(0)

    return best_thresh, accuracy


@torch.inference_mode()
def evaluate(model, loader, device):
    start_time = time.time()
    model.eval()
    embeds, labels = [], []
    dists, targets = None, None

    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1]
    targets = targets[mask == 1]

    threshold, accuracy = find_best_threshold(dists, targets, device)

    print(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")
    print("eval time: {:.4f} s".format(time.time() - start_time))


def save(model, epoch, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = "epoch_" + str(epoch) + "__" + file_name
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)


def get_transform():
    # rand_aug = transforms.Compose(
    #     [
    #         transforms.ColorJitter(brightness=0.5, contrast=0.5),
    #         transforms.RandomRotation(30),
    #         transforms.GaussianBlur(kernel_size=3),
    #     ]
    # )

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    return transform


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    p = args.labels_per_batch
    k = args.samples_per_label
    batch_size = p * k

    model = EmbeddingNet()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))

    model.to(device)

    criterion = TripletMarginLoss(margin=args.margin, mining="batch_hard")
    optimizer = Adam(model.parameters(), lr=args.lr)
    transform = get_transform()

    # Using FMNIST to demonstrate embedding learning using triplet loss. This dataset can
    # be replaced with any classification dataset.
    # train_dataset = FashionMNIST(args.dataset_dir, train=True, transform=transform, download=True)
    # test_dataset = FashionMNIST(args.dataset_dir, train=False, transform=transform, download=True)

    train_dataset = ImageFolder(args.train_dataset_dir, transform=transform)
    test_dataset = ImageFolder(args.test_dataset_dir, transform=transform)

    # train_dataset = AugmentedData(custom_train_dataset, n=args.samples_per_label)
    # test_dataset = AugmentedData(custom_test_dataset, n=args.samples_per_label)

    # targets is a list where the i_th element corresponds to the label of i_th dataset element.
    # This is required for PKSampler to randomly sample from exactly p classes. You will need to
    # construct targets while building your dataset. Some datasets (such as ImageFolder) have a
    # targets attribute with the same format.
    # targets = train_dataset.get_targets()
    targets = train_dataset.targets

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=PKSampler(targets, p, k),
        pin_memory=True,
        num_workers=args.workers, # comment to avoid out of limit files
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers, # comment to avoid out of limit files
    )

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("Evaluating...")
        evaluate(model, test_loader, device)
        return

    for epoch in range(1, args.epochs + 1):
        print("Training...")
        train_epoch(
            model, optimizer, criterion, train_loader, device, epoch, args.print_freq
        )

        print("Saving...")
        save(model, epoch, args.save_dir, "ckpt.pth")

        # print("Evaluating...")
        # evaluate(model, test_loader, device)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Embedding Learning")

    parser.add_argument(
        "--train-dataset-dir",
        default=cfg.train_dataset_dir,
        type=str,
        help="Train dataset directory path",
    )
    parser.add_argument(
        "--test-dataset-dir",
        default=cfg.test_dataset_dir,
        type=str,
        help="Test/val dataset directory path",
    )
    parser.add_argument(
        "-p",
        "--labels-per-batch",
        default=cfg.labels_per_batch,
        type=int,
        help="Number of unique labels/classes per batch",
    )
    parser.add_argument(
        "-k",
        "--samples-per-label",
        default=cfg.samples_per_label,
        type=int,
        help="Number of samples per label in a batch",
    )
    parser.add_argument(
        "--eval-batch-size",
        default=cfg.eval_batch_size,
        type=int,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--epochs",
        default=cfg.epochs,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=cfg.num_workers,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--lr", default=cfg.lr, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--margin", default=cfg.margin, type=float, help="Triplet loss margin"
    )
    parser.add_argument(
        "--print-freq", default=cfg.print_freq, type=int, help="print frequency"
    )
    parser.add_argument(
        "--save-dir", default=cfg.save_dir, type=str, help="Model save directory"
    )
    parser.add_argument(
        "--resume", default=cfg.resume, type=str, help="path of checkpoint"
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
