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


def build_small_resnet(num_classes: int, in_channels: int = 1):
    """A small residual network (~263K params for MNIST/10-class),
    2026-11-15, added as the SECOND backbone -- replacing
    `mobilenet_v2` in `MVP_MODEL_BUILDERS` -- for the same reason
    `build_tiny_cnn` replaced `resnet18`: matching this paper's actual
    peer group's model scale, not an arbitrary size.

    Specifically anchored to WhiteCloak (NDSS 2026, this repo's closest
    peer -- same anonymous-accountability-in-secure-aggregation problem
    as F7f/SuspectTracker)'s OWN second model: their CIFAR-10-L config
    uses a ResNet at 273K parameters (WhiteCloak Table II / Section
    VIII). This builder lands at ~263K for MNIST (num_classes=10), within ~4%
    of that number -- not a different paper's unrelated size, but the
    *same* paper's own second configuration. Combined with
    `build_tiny_cnn` (~20.6K, anchored to WhiteCloak's MNIST CNN at 19K
    and PQS-BFL's MNIST CNN at ~21.8K), this repo's two MVP backbones
    now mirror WhiteCloak's own two reported model sizes end to end --
    a stronger citation story than defending two unrelated choices
    against two different papers.

    Why a residual network specifically, not just "something at 263K
    params": `build_tiny_cnn` is a plain CNN, so pairing it with another
    plain CNN would only vary SIZE, not architecture family. A model
    with real residual (skip) connections demonstrates the protocol
    works across architecture classes, not just across parameter
    counts -- which is what ROADMAP.md §0.1's "2 backbones" was for.

    Architecture (GroupNorm throughout, not BatchNorm, same reasoning
    as every other builder in this file -- see module docstring):
    stem conv(in->32,3x3) -> residual block (32->32, two 3x3 convs) ->
    maxpool(2x2) -> residual block (32->64, 1x1 projection on the skip
    path since channels change) -> maxpool(2x2) -> residual block
    (64->112, 1x1 projection) -> global average pool -> linear(->num_classes).
    For MNIST (28x28x1 in, num_classes=10): stem 384 + block1 18,624 +
    block2 57,792 + block3 185,248 + head 1,130 = ~263K.
    """
    _require_torch()
    import torch
    import torch.nn as nn

    def groupnorm(channels, max_groups=8):
        g = max_groups
        while channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, channels)

    class ResidualBlock(nn.Module):
        """Two 3x3 convs + GroupNorm on the main path; a 1x1 conv on
        the skip path only when in_ch != out_ch (so the add is always
        shape-valid), matching the standard ResNet basic-block pattern
        this repo already follows for resnet18/resnet50 (just built by
        hand here instead of via torchvision, since torchvision has no
        "tiny ResNet" preset at this param scale)."""

        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv_a = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.gn_a = groupnorm(out_ch)
            self.conv_b = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
            self.gn_b = groupnorm(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.proj = None if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            skip = x if self.proj is None else self.proj(x)
            out = self.relu(self.gn_a(self.conv_a(x)))
            out = self.gn_b(self.conv_b(out))
            return self.relu(out + skip)

    class SmallResNet(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                groupnorm(32),
                nn.ReLU(inplace=True),
            )
            self.block1 = ResidualBlock(32, 32)
            self.pool1 = nn.MaxPool2d(2)   # 28x28 -> 14x14
            self.block2 = ResidualBlock(32, 64)
            self.pool2 = nn.MaxPool2d(2)   # 14x14 -> 7x7
            self.block3 = ResidualBlock(64, 112)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(112, num_classes),
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.pool1(self.block1(x))
            x = self.pool2(self.block2(x))
            x = self.block3(x)
            return self.head(x)

    return SmallResNet(in_channels, num_classes)


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


def build_tiny_cnn(num_classes: int, in_channels: int = 1):
    """A small real CNN (NOT the wiring-smoke-test `build_tiny_debug_model`
    above -- this one is meant to go in the paper), 2026-11-15, added to
    match the model scale this paper's actual peer group uses, not
    ResNet-18/50-scale backbones: a literature check across recent
    accountable/anonymous secure-aggregation FL papers (not plain-FedAvg
    scalability papers) found every one of them deliberately uses a
    SMALL model (so the measured cost is the crypto's, not the model's),
    with MNIST specifically landing at ~19-22K params in TWO independent,
    recent papers -- WhiteCloak (NDSS 2026, the closest peer: anonymous
    accountability in secure aggregation, same problem as this repo's
    F7f/SuspectTracker) uses a 19K-param CNN on MNIST; PQS-BFL (2025,
    post-quantum blockchain FL) independently lands on a ~21.8K-param
    LeNet-style CNN on MNIST. Two unrelated groups converging on the
    same scale for the same dataset is as strong a citation anchor as
    this sub-field offers -- so this model is sized to land in that same
    band, not chosen arbitrarily.

    Architecture: conv(1->16,3x3)-GroupNorm-ReLU-maxpool(2x2) ->
    conv(16->32,3x3)-GroupNorm-ReLU-maxpool(2x2) -> flatten -> linear(->num_classes).
    For MNIST (28x28x1 in, num_classes=10): conv1 160 + conv2 4640 +
    GroupNorm 96 + fc (32*7*7=1568 -> 10) 15690 = ~20.6K params --
    landing directly between WhiteCloak's 19K and PQS-BFL's 21.8K.
    GroupNorm kept (not BatchNorm) for the same cross-client-statistic
    reason as every other builder in this file, even though this run is
    IID MNIST and wouldn't itself expose the non-IID BatchNorm failure
    mode -- consistency with the rest of this file's models, not a new
    requirement for this one.
    """
    _require_torch()
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
        nn.GroupNorm(4, 16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 28x28 -> 14x14
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.GroupNorm(8, 32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 14x14 -> 7x7
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, num_classes),
    )


MVP_MODEL_BUILDERS = {
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,  # kept for the later large-model scalability ablation, see build_resnet18's docstring
    "tiny_cnn": build_tiny_cnn,  # 2026-11-15: matches WhiteCloak/PQS-BFL's MNIST CNN scale (19K/~21.8K), see build_tiny_cnn's docstring
    "small_resnet": build_small_resnet,  # 2026-11-15: replaces mobilenet_v2, matches WhiteCloak's CIFAR-10-L ResNet scale (273K), see build_small_resnet's docstring
    # "mobilenet_v2" removed from the MVP set (2026-11-15) -- no
    # comparable paper in this sub-field uses it, and standard
    # mobilenet_v2 is ~2-2.5M params, ~8-35x bigger than any model this
    # paper's peer group actually reports. build_mobilenet_v2 itself is
    # kept below, just not registered as a current MVP backbone -- same
    # treatment as resnet50 above.
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
