from torchvision import models

backbones = {
    "efficientnet_b3": {
        "model": models.efficientnet_b3(
            weights="EfficientNet_B3_Weights.IMAGENET1K_V1"
        ),
        "weights": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "short_name": "eff_b3",
    },
    "mobilenet_v3_large": {
        "model": models.mobilenet_v3_large(
            weights="MobileNet_V3_Large_Weights.IMAGENET1K_V2"
        ),
        "weights": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "short_name": "mobile_v3_l",
    },
    "resnet18": {
        "model": models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1"),
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
        "short_name": "res18",
    },
    "efficientnet_v2_s": {
        "model": models.efficientnet_v2_s(
            weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1"
        ),
        "weights": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "short_name": "eff_v2_s",
    },
    "inception_v3": {
        "model": models.inception_v3(weights="Inception_V3_Weights.IMAGENET1K_V1"),
        "weights": models.Inception_V3_Weights.IMAGENET1K_V1,
        "short_name": "inc_v3",
    },
    "resnet50": {
        "model": models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1"),
        "weights": models.ResNet50_Weights.IMAGENET1K_V1,
        "short_name": "res50",
    },
}
