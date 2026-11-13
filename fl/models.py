"""Model backbones (ARCHITECTURE.md repo layout: "3 backbones, GroupNorm,
1-ch stem, from scratch" -- MVP scope cut to 2: ResNet-50, MobileNetV2;
EfficientNet-B0 dropped per the 3-day-deadline scope decision, see
ROADMAP.md §0.1). TORCH-DEPENDENT -- not run in this sandbox (no torch
installed, no network to install it). Validate in
notebooks/00_Setup_and_Crypto_Tests.ipynb on Colab before trusting a
forward pass actually runs; see STATUS.md for what "written but
untested" means here.

Two deliberate departures from torchvision's stock backbones, both
required by the non-IID FEMNIST setting (REVIEW.md flagged BatchNorm as
a known FL failure mode under non-IID data: BatchNorm's running
mean/var are themselves local-distribution statistics, so averaging
BatchNorm-containing models across clients with different label/writer
distributions averages away exactly the information the model needs):

1. GroupNorm instead of BatchNorm everywhere. GroupNorm normalizes
   within each sample using channel groups, not across the batch, so
   it carries no cross-batch statistic that could leak (or be
   corrupted by) another client's local distribution.
2. A 1-channel stem (MNIST/FEMNIST are grayscale 28x28), replacing the
   stock 3-channel ImageNet stem, with weights initialized from
   scratch -- no ImageNet pretraining, consistent with ARCHITECTURE.md
   explicitly ruling out pretrained-weight shortcuts (a pretrained
   stem would make the "learned entirely through this FL protocol"
   claim weaker, and complicates the privacy story since pretrained
   weights encode information not subject to this pipeline's DP
   accounting).
"""
from __future__ import annotations


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "fl/models.py needs torch + torchvision (not installed in this "
            "sandbox). Validate on Colab -- see "
            "notebooks/00_Setup_and_Crypto_Tests.ipynb."
        ) from e


def _replace_batchnorm_with_groupnorm(module, groups: int = 32):
    import torch.nn as nn

    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            num_channels = child.num_features
            # GroupNorm requires num_channels % groups == 0; fall back to
            # a smaller group count (down to 1 = LayerNorm-equivalent)
            # rather than crashing on odd channel counts from a 1-ch stem.
            g = groups
            while num_channels % g != 0 and g > 1:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, num_channels))
        else:
            _replace_batchnorm_with_groupnorm(child, groups=groups)
    return module


def build_resnet50(num_classes: int, in_channels: int = 1):
    """ResNet-50, from-scratch weights, 1-channel stem, GroupNorm
    throughout (replacing every BatchNorm2d). `num_classes`: 10 for
    MNIST (CASE1), 62 for FEMNIST (CASE2, digits+upper+lower case).
    """
    _require_torch()
    import torch.nn as nn
    import torchvision

    model = torchvision.models.resnet50(weights=None)  # from scratch, no ImageNet weights
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = _replace_batchnorm_with_groupnorm(model)
    return model


def build_resnet18(num_classes: int, in_channels: int = 1):
    """ResNet-18, from-scratch weights, 1-channel stem, GroupNorm
    throughout. Primary MVP backbone as of 2026-11-13 (see
    STATUS.md/FIX.md): FL-literature convention for this class of
    benchmark (FedAvg, FedProx, SCAFFOLD, MOON, the non-IID BatchNorm
    papers that motivated this file's GroupNorm swap all use
    ResNet-18 or smaller, never ResNet-50), and ~47% of ResNet-50's
    `model_dim` (~11.2M vs ~23.6M params with a 62-class head), which
    matters directly here since mask generation, sketching and range
    proofs are all O(model_dim). `build_resnet50` is kept, not
    removed -- it is still the intended large-model scalability
    ablation once F2 (mask PRG) and F5 (memory) land (FIX.md).
    """
    _require_torch()
    import torch.nn as nn
    import torchvision

    model = torchvision.models.resnet18(weights=None)  # from scratch, no ImageNet weights
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = _replace_batchnorm_with_groupnorm(model)
    return model


def build_mobilenet_v2(num_classes: int, in_channels: int = 1):
    """MobileNetV2, from-scratch weights, 1-channel stem, GroupNorm
    throughout. The lighter of the two MVP backbones -- chosen
    (alongside ResNet-50) specifically to keep per-client compute
    inside the ~12 GPU-hr MVP budget (ROADMAP.md §0.1); EfficientNet-B0
    is deferred to the camera-ready run.
    """
    _require_torch()
    import torch.nn as nn
    import torchvision

    model = torchvision.models.mobilenet_v2(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels, first_conv.out_channels,
        kernel_size=first_conv.kernel_size, stride=first_conv.stride,
        padding=first_conv.padding, bias=False,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = _replace_batchnorm_with_groupnorm(model)
    return model


def build_tiny_debug_model(num_classes: int, in_channels: int = 1):
    """A tiny (~8K-param) single-linear-layer model for validating the
    ROUND-LOOP WIRING (fl/round_loop.py) quickly -- NOT a real MVP
    backbone, and deliberately NOT in MVP_MODEL_BUILDERS so it can
    never be mistaken for one or accidentally used in a real
    CASE1/CASE2 experiment run.

    Why this exists: ResNet-50/MobileNetV2's real d~23.5M is fixed by
    the architecture regardless of how few clients/rounds a "smoke
    test" uses -- crypto/sketch.py's SketchProjector AND
    crypto/vector_commit.py's weighted_digest both do an O(d) pass in
    slow NumPy/Python reference code (see each module's own
    PERFORMANCE NOTE), so a round-loop smoke test against the real
    model dimension can still take many minutes to hours per client.
    This model's d is small enough (784*10+10 = 7850 for MNIST's
    28x28x1 input) that the SAME O(d) passes finish in well under a
    second, making it possible to validate the PROTOCOL correctness
    (V1-V5, the independent V5 audit, B_t evolution, apply_update) in
    seconds rather than discovering a wiring bug after a 40-minute
    wait. Swap back to a real MVP_MODEL_BUILDERS entry once the smoke
    test passes clean against this.
    """
    _require_torch()
    import torch.nn as nn

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28 * in_channels, num_classes),
    )


MVP_MODEL_BUILDERS = {
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,  # kept for the later large-model scalability ablation, see build_resnet18's docstring
    "mobilenet_v2": build_mobilenet_v2,
    # "efficientnet_b0" intentionally omitted -- see module docstring
    # and ROADMAP.md §0.1's 3-day scope cut.
}


def build_model(name: str, num_classes: int, in_channels: int = 1):
    if name not in MVP_MODEL_BUILDERS:
        raise ValueError(f"unknown or deferred model '{name}'; MVP set is {list(MVP_MODEL_BUILDERS)}")
    return MVP_MODEL_BUILDERS[name](num_classes, in_channels)


def flatten_state_dict(model) -> "tuple":
    """Flattens a model's state_dict into one 1D float32 tensor (the
    `w_t` / `Delta_k` representation the round protocol operates on)
    plus the shape metadata needed to unflatten it later. TORCH-
    DEPENDENT like the rest of this file.
    """
    _require_torch()
    import torch

    shapes = []
    pieces = []
    for key, tensor in model.state_dict().items():
        shapes.append((key, tuple(tensor.shape)))
        pieces.append(tensor.detach().reshape(-1).float())
    flat = torch.cat(pieces)
    return flat, shapes


def unflatten_to_state_dict(flat, shapes) -> dict:
    _require_torch()
    import torch

    state_dict = {}
    cursor = 0
    for key, shape in shapes:
        numel = 1
        for s in shape:
            numel *= s
        state_dict[key] = flat[cursor : cursor + numel].reshape(shape)
        cursor += numel
    return state_dict
