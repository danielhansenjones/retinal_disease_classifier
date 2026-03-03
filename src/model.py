import timm
import torch
import torch.nn as nn


class DualEyeModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, dropout: float):
        super().__init__()
        # Shared backbone - same weights process both eyes
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features * 2, num_classes),
        )

    def enable_grad_checkpointing(self):
        self.backbone.set_grad_checkpointing(enable=True)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # Process both eyes in a single backbone pass - avoids sequential GPU idle time
        both = torch.cat([left, right], dim=0)
        features = self.backbone(both)
        left_feat, right_feat = features.chunk(2, dim=0)
        return self.classifier(torch.cat([left_feat, right_feat], dim=1))


def build_model(backbone: str, num_classes: int, dropout: float) -> DualEyeModel:
    return DualEyeModel(backbone, num_classes, dropout)


def freeze_backbone(model: DualEyeModel):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: DualEyeModel):
    for param in model.parameters():
        param.requires_grad = True