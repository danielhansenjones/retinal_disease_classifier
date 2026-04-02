import torch
import torch.nn as nn

from src.model import EnsembleModel


class _StubBackbone(nn.Module):
    def __init__(self, num_classes: int, bias: float = 0.0):
        super().__init__()
        self.bias = bias
        self.linear = nn.Linear(3, num_classes)

    def forward(self, left, right):
        # Mimic DualEyeModel signature: returns logits of shape (B, num_classes)
        b = left.shape[0]
        x = left.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).expand(b, 3)
        return self.linear(x) + self.bias


def test_ensemble_output_shape_and_probability_range():
    num_classes = 8
    models = [_StubBackbone(num_classes, bias=0.5), _StubBackbone(num_classes, bias=-0.3)]
    means = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
    stds = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]
    ensemble = EnsembleModel(models, means, stds)

    left = torch.rand(4, 3, 32, 32)
    right = torch.rand(4, 3, 32, 32)
    out = ensemble(left, right)

    assert out.shape == (4, num_classes)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_ensemble_applies_per_backbone_normalization():
    models = [_StubBackbone(2), _StubBackbone(2)]
    means = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    stds = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    ensemble = EnsembleModel(models, means, stds)

    # Buffers registered, distinct, and on the same device as the module
    assert ensemble.mean_0.flatten().tolist() == [0.0, 0.0, 0.0]
    assert ensemble.mean_1.flatten().tolist() == [1.0, 1.0, 1.0]

    # Forward should not raise; verifies normalization broadcasting works
    left = torch.rand(2, 3, 16, 16)
    right = torch.rand(2, 3, 16, 16)
    out = ensemble(left, right)
    assert out.shape == (2, 2)