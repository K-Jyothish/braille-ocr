import numpy as np
import pytest
import torch

from brailleocr.detection.classical import ClassicalParams, detect as classical_detect
from brailleocr.detection.data import CropDataset, collate_crops, load_index
from brailleocr.detection.inference import predict_prob
from brailleocr.detection.losses import focal_heatmap_loss, mask_bce_loss
from brailleocr.detection.models import PlainUNet, ResNetUNet
from brailleocr.detection.postprocess import heatmap_peaks, mask_centroids
from brailleocr.detection.targets import disk_mask, gaussian_heatmap
from brailleocr.evaluation import match_dots


def test_heatmap_peaks_recover_subpixel_centres():
    pts = np.array([[20.3, 15.7], [40.0, 15.0], [27.6, 40.2]])
    heat = gaussian_heatmap((64, 64), pts, 1.5)
    xy, conf = heatmap_peaks(heat, threshold=0.5)
    tp, fp, fn, _ = match_dots(xy, pts, 0.5)
    assert (tp, fp, fn) == (3, 0, 0)
    assert np.all(conf > 0.99)


def test_heatmap_peaks_separate_adjacent_dots():
    pts = np.array([[10.0, 10.0], [17.0, 10.0], [10.0, 17.0]])  # 7 px pitch, the smallest in the data
    xy, _ = heatmap_peaks(gaussian_heatmap((32, 32), pts, 1.2), threshold=0.5)
    assert len(xy) == 3


def test_mask_centroids():
    pts = np.array([[10.0, 10.0], [30.0, 22.0]])
    xy, conf = mask_centroids(disk_mask((40, 40), pts, 3.0), threshold=0.5)
    assert match_dots(xy, pts, 0.5)[:3] == (2, 0, 0)
    assert np.all(conf == 1.0)


def test_losses_prefer_correct_prediction():
    target = torch.from_numpy(gaussian_heatmap((32, 32), np.array([[16.0, 16.0]]), 1.5))[None, None]
    good = torch.logit(target.clamp(1e-4, 1 - 1e-4))
    bad = torch.zeros_like(good)
    assert focal_heatmap_loss(good, target) < focal_heatmap_loss(bad, target)
    mask = (target > 0.5).float()
    assert mask_bce_loss(good * 10, mask) < mask_bce_loss(bad, mask)


@pytest.mark.parametrize("model", [PlainUNet(), ResNetUNet("resnet18", pretrained=False)])
def test_models_output_full_resolution(model):
    x = torch.rand(2, 3, 64, 96)
    assert model(x).shape == (2, 1, 64, 96)


def test_predict_prob_tiling_matches_single_pass():
    torch.manual_seed(0)
    model = PlainUNet().eval()
    img = (np.random.default_rng(0).uniform(0, 255, (100, 150, 3))).astype(np.uint8)
    tiled = predict_prob(model, img, torch.device("cpu"), tile=64, overlap=16)
    assert tiled.shape == (100, 150)
    # Away from tile seams' context limits the tiled result equals a direct pass on the padded page.
    padded = np.pad(img, ((0, 4), (0, 2), (0, 0)), mode="reflect")
    with torch.no_grad():
        direct = torch.sigmoid(model(torch.from_numpy(padded).permute(2, 0, 1)[None].float() / 255))[0, 0].numpy()
    assert np.abs(tiled[20:-20, 20:-20] - direct[20:80, 20:130]).max() < 1e-4


def test_crop_dataset_targets_align(tmp_path):
    from brailleocr.synth.dataset import generate_split

    root = generate_split(
        ["The quick brown fox jumps over the lazy dog.", "She had never seen anything like it in all her life."],
        "train", tmp_path, n_pages=1, seed=0,
    )
    recs = load_index(root)
    for target in ("heatmap", "mask"):
        ds = CropDataset(recs, target=target, crop=128, crops_per_page=3, p_near_dot=1.0)
        x, y = collate_crops([ds[0]])
        assert x.shape == (3, 3, 128, 128) and y.shape == (3, 1, 128, 128)
        assert 0 <= x.min() and x.max() <= 1
        assert y.max() == 1.0  # crops near a dot contain at least one target


def test_classical_finds_dots_on_clean_page(tmp_path):
    import json

    from brailleocr.detection.data import read_rgb
    from brailleocr.synth.dataset import generate_split

    root = generate_split(["The quick brown fox jumps over the lazy dog."] * 3, "test", tmp_path, n_pages=1, seed=0, levels=(0.0,))
    rec = load_index(root)[0]
    label = json.loads(rec.label.read_text())
    xy, _ = classical_detect(read_rgb(rec.image), ClassicalParams())
    tp, fp, fn, _ = match_dots(xy, np.array([d[:2] for d in label["dots"]]), 3.0)
    assert tp / (tp + fn) > 0.8
