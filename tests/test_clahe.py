import cv2
import numpy as np
from PIL import Image

from src.dataset import apply_clahe


def _random_rgb(seed=0, size=64):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_clahe_preserves_dtype_and_shape():
    img = _random_rgb()
    out = apply_clahe(img)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == img.size
    assert np.array(out).dtype == np.uint8


def test_clahe_modifies_image():
    img = _random_rgb()
    out = apply_clahe(img)
    assert not np.array_equal(np.array(img), np.array(out))


def test_clahe_targets_luminance_more_than_chroma():
    # After LAB->RGB->LAB round-trip, A/B drift due to uint8 clamping. The right
    # invariant is that CLAHE perturbs L substantially more than A/B.
    img = _random_rgb(seed=42)
    out = apply_clahe(img)

    lab_in = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB).astype(int)
    lab_out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2LAB).astype(int)

    l_diff = np.abs(lab_in[:, :, 0] - lab_out[:, :, 0]).mean()
    a_diff = np.abs(lab_in[:, :, 1] - lab_out[:, :, 1]).mean()
    b_diff = np.abs(lab_in[:, :, 2] - lab_out[:, :, 2]).mean()

    assert l_diff > 0
    assert l_diff > 3 * a_diff
    assert l_diff > 3 * b_diff