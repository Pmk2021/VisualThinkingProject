import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None


def _get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


class UNetKeypointModel(nn.Module):
    """Mirrors the layout of astra/models/keypoint_model.py:UNETEmbeddingExtractor so that pretrained
    ASTRA weights load directly, and our trained weights load back into ASTRAEDMDiffusionModel's
    RGBBoxContextEncoder via the same key prefixes."""

    def __init__(self, config):
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required for UNetKeypointModel. "
                "Install with: pip install segmentation_models_pytorch"
            )
        encoder_name = _get(config, "feature_extractor", "resnet18")
        encoder_weights = _get(config, "encoder_weights", None)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        in_channels = self.unet.segmentation_head[0].in_channels
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.unet.segmentation_head = nn.Identity()
        self.feature_extractor.apply(self._init_weights)
        self.seg_head.apply(self._init_weights)
        self.register_buffer("rgb_norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight)

    def normalize(self, x):
        return (x - self.rgb_norm_mean) / self.rgb_norm_std

    def forward(self, x):
        x = self.normalize(x)
        decoder_features = self.unet(x)
        features = self.feature_extractor(decoder_features)
        heatmap = self.seg_head(features)
        return heatmap


def load_unet_keypoint_state(model, weights_path, strict=False):
    """Load weights into a UNetKeypointModel, tolerating common state_dict prefixes."""
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    return missing, unexpected
