import __main__
import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from tqdm import tqdm

from model.net import EmbeddingNet, EmbeddingNetL2, TripletNet
import config as cfg

device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelLoader:
    def __init__(self, model_file: str):
        self.model_file = model_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # backbone = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        backbone = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.IMAGENET1K_V2")
        emb_net = EmbeddingNet(backbone)
        self.model = TripletNet(emb_net)

    def load(self):
        setattr(__main__, "TripletNet", TripletNet)
        setattr(__main__, "EmbeddingNet", EmbeddingNet)

        model = torch.load(self.model_file, map_location=torch.device(self.device))
        return model
    
    def load_state_dict(self):
        self.model.load_state_dict(torch.load(
            self.model_file, 
            weights_only=True, 
        ))
        self.model = self.model.to(self.device)
        return self.model

    def get_device(self):
        return self.device


def get_embeddings_and_labels(
    data: datasets.ImageFolder, model: nn.Module, device: str
) -> tuple:
    model.eval()
    embeddings = []
    labels = []

    for (img_path, label_), (img, label) in tqdm(zip(data.imgs, data), total=len(data)):
        assert label_ == label
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            if type(model) is TripletNet:
                embedding = model.embedding_net(img).squeeze(0)
            else:
                embedding = model(img).squeeze(0)

        embeddings.append(embedding.cpu())
        labels.append(label)

    return np.array(embeddings), np.array(labels)


def pairwise_embedding(
    embeddings_1: list | np.ndarray,
    embeddings_2: list | np.ndarray | None = None,
) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity

    print("Calculating embedding similarity matrix ...")
    sim_matrix = None
    np_embeddings_1 = np.array(embeddings_1)

    if not embeddings_2:
        sim_matrix = cosine_similarity(np_embeddings_1)
    else:
        np_embeddings_2 = np.array(embeddings_2)
        sim_matrix = cosine_similarity(np_embeddings_1, np_embeddings_2)

    return np.array(sim_matrix)


class Metric:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = labels

        self.unique_labels = np.unique(labels)
        self.centroids = np.array(
            [embeddings[labels == label].mean(axis=0) for label in self.unique_labels]
        )
        self.all_distances = cosine_distances(embeddings, self.centroids)

        self.intra_cluster_distances = None
        self.nearest_cluster_distances = None

    def intra_cluster_distance(self):
        intra_cluster_distances = np.zeros(self.embeddings.shape[0])
        for i, label in enumerate(self.labels):
            unique_label_index = list(self.unique_labels).index(label)

            # Sim to the centroid of the cluster it belongs to
            intra_cluster_distances[i] = self.all_distances[i, unique_label_index]

        self.intra_cluster_distances = intra_cluster_distances
        return np.mean(intra_cluster_distances)

    def nearest_cluster_distance(self):
        nearest_cluster_distances = np.zeros(self.embeddings.shape[0])
        for i, label in enumerate(self.labels):
            unique_label_index = list(self.unique_labels).index(label)

            # Sim to the centroids of all other clusters
            other_cluster_distances = np.delete(
                self.all_distances[i], unique_label_index
            )
            # Sim to the nearest cluster
            nearest_cluster_distances[i] = np.min(other_cluster_distances)

        self.nearest_cluster_distances = nearest_cluster_distances
        return np.mean(nearest_cluster_distances)

    def silhouette(self):
        if self.intra_cluster_distances is None:
            self.intra_cluster_distance()
        if self.nearest_cluster_distances is None:
            self.nearest_cluster_distance()

        silhouette_scores = np.zeros(self.embeddings.shape[0])
        for i, (a, b) in enumerate(
            zip(self.intra_cluster_distances, self.nearest_cluster_distances)
        ):
            silhouette_scores[i] = (b - a) / (max(a, b) + 1e-9)

        return np.mean(silhouette_scores)


def main(args):
    image_folder = args.data
    model_file = args.model
    target_shape = (224, 224)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(target_shape)]
    )

    data = datasets.ImageFolder(image_folder, transform)
    model = ModelLoader(model_file).load_state_dict()
    # model = ModelLoader(model_file).load()

    embeddings, labels = get_embeddings_and_labels(data, model, device)
    matrix = pairwise_embedding(embeddings)
    print(f"Average: {np.mean(matrix)}, maximum: {np.max(matrix[matrix < 0.99])}, minimum: {np.min(matrix)}")

    metric = Metric(embeddings, labels)
    print("Intra: {:.4f}".format(metric.intra_cluster_distance()))
    print("Nearest: {:.4f}".format(metric.nearest_cluster_distance()))
    print("Silhouette: {:.4f}".format(metric.silhouette()))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default=cfg.VAL_DATASET_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
