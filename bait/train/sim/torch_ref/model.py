import torch.nn as nn
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            backbone = models.mobilenet_v3_large(
                weights="MobileNet_V3_Large_Weights.IMAGENET1K_V2"
            )

            # backbone = models.efficientnet_v2_l(
            #     weights="EfficientNet_V2_L_Weights.IMAGENET1K_V1"
            # )

        backbone = list(backbone.children())[:-1]
        self.convnet = nn.Sequential(*backbone)

    def forward(self, x):
        x = self.convnet(x).flatten(start_dim=1)
        x = x.view(x.size()[0], -1)
        x = nn.functional.normalize(x, dim=1)
        return x
