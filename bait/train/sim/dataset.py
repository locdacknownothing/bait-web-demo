import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AugmentedData(Dataset):
    def __init__(self, data, n=2):
        """
        Args:
            data (ImageFolder): the original datasets
            n (int): the number of augmented images
        """

        self.imgs = data.imgs * n
        self.n = n

        base_transform = data.transform
        augmenter = transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
        if base_transform:
            self.transform = transforms.Compose(
                [
                    augmenter,
                    base_transform,
                ]
            )
        else:
            self.transform = augmenter

    def __len__(self):
        return len(self.imgs)


class TripletData(Dataset):
    """
    Get positive and negative images of anchor randomly
    """

    def __init__(self, dataset: Dataset):
        self.data = dataset.imgs
        self.transform = dataset.transform
        self.labels = np.array(self.data)[:, -1]
        self.data_img = np.array(self.data)[:, 0]

        self.labels_set = set(self.labels)

        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }

        # self.items = [self.getitem(index) for index in range(len(self.data))]

    def __getitem__(self, index):
        # return self.items[index], tuple()
        return self.getitem(index), tuple()

    def getitem(self, index):
        img1, label1 = self.data_img[index], self.labels[index].item()

        positive_index = index
        while positive_index == index:
            if len(self.label_to_indices[label1]) == 1:
                raise ValueError("Class has only one image")

            positive_index = np.random.choice(self.label_to_indices[label1])

        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])

        img2 = self.data_img[positive_index]
        img3 = self.data_img[negative_index]

        img1 = Image.open(img1)
        img2 = Image.open(img2)
        img3 = Image.open(img3)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.data)
