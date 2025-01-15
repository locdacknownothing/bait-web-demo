import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AugmentedData(Dataset):
    def __init__(self, data, n=2):
        """
        Args:
            data (ImageFolder): the original dataset
            n (int): the number of augmented copies of the original dataset
        """
        super(AugmentedData, self).__init__()
        self.data = data
        self.n = n

        self.transform = data.transform
        # augmenter = transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
        # if self.base_transform:
        #     self.transform = transforms.Compose(
        #         [
        #             augmenter,
        #             self.base_transform,
        #         ]
        #     )
        # else:
        #     self.transform = augmenter

    def __len__(self):
        return len(self.data) * self.n

    def __getitem__(self, index):
        original_index = index % len(self.data)
        image_path, label = self.data.imgs[original_index]

        if self.transform:
            image = Image.open(image_path)
            image = self.transform(image)

        return image, label

    # custom method to get list of augmented labels
    def get_targets(self):
        return self.data.targets * self.n
