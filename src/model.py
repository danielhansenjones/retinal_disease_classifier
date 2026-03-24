from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


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
        try:
            self.backbone.set_grad_checkpointing(enable=True)
        except (AssertionError, AttributeError):
            # timm doesn't support checkpointing for this backbone - apply it
            # manually to the repeat blocks which hold the bulk of activations
            _applied = []
            for name in ("repeat", "repeat_1", "repeat_2"):
                block = getattr(self.backbone, name, None)
                if isinstance(block, nn.Sequential) and len(block) > 1:
                    original = block.forward
                    segments = max(1, len(block) // 2)

                    def make_checkpointed(seq, n):
                        def forward(x):
                            return checkpoint_sequential(seq, n, x, use_reentrant=False)
                        return forward

                    block.forward = make_checkpointed(block, segments)
                    _applied.append(name)
            if _applied:
                print(f"Applied manual gradient checkpointing to: {', '.join(_applied)}")
            else:
                print("Warning: grad checkpointing not supported and no repeat blocks found - running without it")

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # Process both eyes in a single backbone pass - avoids sequential GPU idle time
        both = torch.cat([left, right], dim=0)
        features = self.backbone(both)
        left_feat, right_feat = features.chunk(2, dim=0)
        return self.classifier(torch.cat([left_feat, right_feat], dim=1))


def build_model(backbone: str, num_classes: int, dropout: float) -> DualEyeModel:
    return DualEyeModel(backbone, num_classes, dropout)


class EnsembleModel(nn.Module):
    """Averages sigmoid probabilities from multiple DualEyeModels.

    Each model may expect different input normalization. Pass un-normalized
    float tensors in [0, 1] (Resize + ToTensor only - no Normalize); this
    class applies per-backbone normalization internally via registered buffers.
    """

    def __init__(
        self,
        models: list[DualEyeModel],
        norm_means: list[list[float]],
        norm_stds: list[list[float]],
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        for i, (mean, std) in enumerate(zip(norm_means, norm_stds)):
            self.register_buffer(f"mean_{i}", torch.tensor(mean, dtype=torch.float32).view(3, 1, 1))
            self.register_buffer(f"std_{i}", torch.tensor(std, dtype=torch.float32).view(3, 1, 1))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        probs = []
        for i, model in enumerate(self.models):
            mean = getattr(self, f"mean_{i}")
            std = getattr(self, f"std_{i}")
            l_norm = (left - mean) / std
            r_norm = (right - mean) / std
            probs.append(torch.sigmoid(model(l_norm, r_norm)))
        return torch.stack(probs).mean(dim=0)


def load_ensemble(
    checkpoints: list[Path],
    backbone_names: list[str],
    norm_means: list[list[float]],
    norm_stds: list[list[float]],
    num_classes: int,
    dropout: float,
    device: torch.device,
) -> EnsembleModel:
    models = []
    for ckpt_path, backbone in zip(checkpoints, backbone_names):
        m = build_model(backbone, num_classes, dropout).to(device)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return EnsembleModel(models, norm_means, norm_stds).to(device)


def freeze_backbone(model: DualEyeModel):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: DualEyeModel):
    for param in model.parameters():
        param.requires_grad = True