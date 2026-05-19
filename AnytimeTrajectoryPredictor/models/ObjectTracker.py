"""YOLO nano + ByteTrack object tracker with configurable feature output.

This module exposes a lightweight tracker that accepts a single RGB image frame
and returns only dataset-style tensors:

- features: (1, O, F)
- mask: (1, O, 1)

The feature tensor is assembled from the requested components in this order:
1. bounding boxes as [center_x, center_y, width, height]
2. confidences
3. class IDs
4. latent features
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from ultralytics import YOLO


class ObjectTracker:
    """YOLO nano + ByteTrack tracker with configurable feature components."""

    FEATURES: Dict[str, Any] = {
        "bboxes": "_get_bboxes_component",
        "confidences": "_get_confidences_component",
        "object_ids": "_get_object_ids_component",
        "class_ids": "_get_class_ids_component",
        "latent_features": "_get_latent_features_component",
        "local_latent_features": "_get_local_latent_features_component",
    }

    def __init__(
        self,
        model_name: Union[str, Path] = "yolo26n.pt",
        tracker_cfg: Union[str, Path] = "bytetrack.yaml",
        device: Optional[Union[str, int, torch.device]] = None,
        conf: float = 0.25,
        imgsz: int = 640,
        classes: Optional[Sequence[int]] = None,
        max_det: int = 300,
        feature_components: Sequence[str] = ("bboxes",),
        latent_features_layers: Optional[Sequence[str]] = [
            "model.16",
            "model.19",
        ],  # Defaults to P3 and P4 features
        local_latent_feature_fraction: float = 0.10,
        verbose: bool = False,
        additional_track_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        An object tracker that extracts object and scene features from RGB image using YOLO and ByteTrack.

        Parameters
        ----------
        model_name : Union[str, Path], optional
            The name or path of the YOLO model to use for tracking
        tracker_cfg : Union[str, Path], optional
            The name or path of the ByteTrack tracker config file, by default "bytetrack.yaml"
        device : Optional[Union[str, int, torch.device]], optional
            The device to run the model on (e.g., "cpu", "cuda:0"), by default None (automatically selects CUDA if available)
        conf : float, optional
            Confidence threshold for detections, by default 0.25
        imgsz : int, optional
            Inference image size for the model, by default 640
        classes : Optional[Sequence[int]], optional
            Filter detections to only these class IDs, by default None (no filtering)
        max_det : int, optional
            Maximum number of detections per frame, by default 300
        feature_components : Sequence[str], optional
            Which feature components to include in the output features tensor. Allowed values are "bboxes", "confidences", "class_ids", "latent_features". By default ("bboxes",)
        latent_features_layers : Optional[Sequence[str]], optional
            If "latent_features" is included in feature_components, this specifies which model layers to capture as latent_features. Each layer's output will be averaged spatially and concatenated together. By default ["model.16", "model.19"] (P3 and P4 features)
        local_latent_feature_fraction : float, optional
            Spatial fraction of each latent feature map to keep for local object crops. For example, 0.10 on a 1920x1080 feature map yields a 192x108 local representation per layer.
        verbose : bool, optional
            If True, prints additional information about the model and tracking process, by default False
        additional_track_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments to pass to the track() method of the YOLO model. This allows for further customization of the tracking behavior beyond the main parameters exposed by ObjectTracker. Overrides any conflicting parameters set by the main arguments (e.g., conf, imgsz, classes, max_det) if specified.
        """
        self.model_name = str(model_name)
        self.tracker_cfg = str(tracker_cfg)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.conf = conf
        self.imgsz = imgsz
        self.classes = list(classes) if classes is not None else None
        self.max_det = max_det
        self.latent_features_layers = latent_features_layers
        self.local_latent_feature_fraction = float(local_latent_feature_fraction)
        self.feature_components = tuple(feature_components)
        self.verbose = verbose

        invalid_components = [
            component
            for component in self.feature_components
            if component not in self.FEATURES
        ]
        if invalid_components:
            raise ValueError(
                f"Invalid feature_components: {invalid_components}. Allowed values are {self.FEATURES}."
            )
        if (
            "latent_features" in self.feature_components
            or "local_latent_features" in self.feature_components
        ) and (
            self.latent_features_layers is None or len(self.latent_features_layers) == 0
        ):
            raise ValueError(
                "feature_components includes 'latent_features' but latent_features_layers is None."
            )
        if self.local_latent_feature_fraction <= 0:
            raise ValueError("local_latent_feature_fraction must be strictly positive.")

        self.track_kwargs: Dict[str, Any] = {
            "persist": True,
            "tracker": self.tracker_cfg,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device is not None:
            self.track_kwargs["device"] = self.device
        if self.classes is not None:
            self.track_kwargs["classes"] = self.classes
        if additional_track_kwargs is not None:
            self.track_kwargs.update(additional_track_kwargs)

        self._tracker_model = YOLO(self.model_name)

        # Optimize for inference:
        self._tracker_model.fuse()
        self._tracker_model.eval()
        torch.set_grad_enabled(False)

        self._last_latent_features: Optional[Dict[str, torch.Tensor]] = None

        self._seen = []

        if self.latent_features_layers is not None:
            self._register_latent_features_hooks()

    def _register_latent_features_hooks(self) -> None:
        """Register a persistent forward hook that captures one latent feature set per frame."""

        def get_hook_fn(layer_name: str):
            def hook_fn(module: Any, input: Any, output: Any) -> None:
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        f"Expected output of layer '{layer_name}' to be a torch.Tensor, but got {type(output)}."
                    )

                latent_features = output.detach().cpu().float()
                self._last_latent_features[layer_name] = latent_features

            return hook_fn

        self._last_latent_features = {}
        hooked_layers = {}
        backbone = self._tracker_model.model
        for layer_name in self.latent_features_layers:
            layer = backbone.get_submodule(layer_name)
            layer.register_forward_hook(get_hook_fn(layer_name))
            hooked_layers[layer_name] = layer.__class__.__name__
        if self.verbose:
            print(f"Hooked layers for latent features: {hooked_layers}")

    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image, str],
        jpeg: bool = False,
    ) -> Dict[str, Union[torch.Tensor, list[tuple[str, int]]]]:
        """
        Track one RGB frame and return only features and mask.

        Features are assembled in the order declared by feature_components.
        """

        for layer_name in self.latent_features_layers:
            self._last_latent_features[layer_name] = None

        results = self._tracker_model.track(source=image, **self.track_kwargs)

        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        confidences = np.zeros((0,), dtype=np.float32)
        class_ids = np.zeros((0,), dtype=np.int64)
        object_ids = np.zeros((0,), dtype=np.int64)

        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                boxes_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                confidences = (
                    boxes.conf.detach().cpu().numpy().astype(np.float32)
                    if boxes.conf is not None
                    else np.zeros((len(boxes_xyxy),), dtype=np.float32)
                )
                object_ids = (
                    boxes.id.detach().cpu().numpy().astype(np.int64)
                    if boxes.id is not None
                    else np.zeros((len(boxes_xyxy),), dtype=np.int64)
                )
                class_ids = (
                    boxes.cls.detach().cpu().numpy().astype(np.int64)
                    if boxes.cls is not None
                    else np.zeros((len(boxes_xyxy),), dtype=np.int64)
                )

        for object_id in object_ids:
            if object_id not in self._seen:
                self._seen.append(object_id)

        components: list[np.ndarray] = []
        lengths: list[tuple[str, int]] = []
        valid_objects = len(boxes_xyxy)

        for feature in self.feature_components:
            component, component_length = getattr(self, self.FEATURES[feature])(
                image=image,
                boxes_xyxy=boxes_xyxy,
                confidences=confidences,
                object_ids=object_ids,
                class_ids=class_ids,
                latent_features=self._last_latent_features,
            )
            components.append(component)
            lengths.append((feature, component_length))

        if components:
            features_arr = np.concatenate(components, axis=1).astype(np.float32)
        else:
            features_arr = np.zeros((valid_objects, 0), dtype=np.float32)

        mask = np.zeros((len(self._seen), 1), dtype=bool)
        for i, object_id in enumerate(self._seen):
            if object_id in object_ids:
                mask[i] = True

        features = torch.from_numpy(features_arr)
        mask_tensor = torch.from_numpy(mask)

        features_cst = torch.zeros((1, self.max_det, features.shape[1]), dtype=torch.float32)
        mask_cst = torch.zeros((1, self.max_det, 1), dtype=torch.float32)
        
        features_cst[:, :features.shape[0], :] = features
        mask_cst[:, :mask_tensor.shape[0], :] = mask_tensor
        
        return {
            "features": features_cst.unsqueeze(0).float(),
            "lengths": lengths,
            "mask": mask_cst.unsqueeze(0).float(),
        }

    def _get_bboxes_component(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image, str],
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        object_ids: np.ndarray,
        class_ids: np.ndarray,
        latent_features: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, int]:
        if len(boxes_xyxy) > 0:
            x1, y1, x2, y2 = (
                boxes_xyxy[:, 0],
                boxes_xyxy[:, 1],
                boxes_xyxy[:, 2],
                boxes_xyxy[:, 3],
            )
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            return (
                np.stack([center_x, center_y, width, height], axis=1).astype(
                    np.float32
                ),
                4,
            )
        else:
            return np.zeros((0, 4), dtype=np.float32), 4

    def _get_confidences_component(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image, str],
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        object_ids: np.ndarray,
        class_ids: np.ndarray,
        latent_features: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, int]:
        return confidences.reshape(-1, 1).astype(np.float32), 1

    def _get_object_ids_component(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image, str],
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        object_ids: np.ndarray,
        class_ids: np.ndarray,
        latent_features: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, int]:
        return object_ids.reshape(-1, 1).astype(np.float32), 1

    def _get_class_ids_component(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image, str],
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        object_ids: np.ndarray,
        class_ids: np.ndarray,
        latent_features: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, int]:
        return class_ids.reshape(-1, 1).astype(np.float32), 1

    def _get_latent_features_component(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image, str],
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        object_ids: np.ndarray,
        class_ids: np.ndarray,
        latent_features: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, int]:
        if latent_features is None or len(latent_features) == 0:
            raise RuntimeError(
                "Embeddings were requested but no latent feature hooks have produced outputs yet. "
                "Check that latent_features_layers points to valid layers."
            )

        missing_layers = [
            layer for layer, value in latent_features.items() if value is None
        ]
        if missing_layers:
            raise RuntimeError(
                f"Embeddings were requested but no latent features were captured for {missing_layers[0]}. "
                "Check that latent_features_layers points to valid layers."
            )

        curr_latent_features = []
        for layer_name in self.latent_features_layers:
            layer_latent_features = latent_features[layer_name]

            # Turn the latent features into (1, C)
            layer_latent_features = layer_latent_features.mean(dim=[2, 3])

            curr_latent_features.append(layer_latent_features)

        concatenated_latent_features = (
            torch.cat(curr_latent_features, dim=1).cpu().numpy()
        )
        latent_features_row = np.repeat(
            concatenated_latent_features.reshape(1, -1), len(boxes_xyxy), axis=0
        )
        return latent_features_row.astype(np.float32), latent_features_row.shape[1]

    def _get_local_latent_features_component(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image, str],
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        object_ids: np.ndarray,
        class_ids: np.ndarray,
        latent_features: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, int]:
        """
        Returns local latent features for each detected object by cropping the feature maps to the bounding
        box regions and applying adaptive average pooling to get a fixed-size feature vector per object.
        This allows the model to capture more localized information about each detected object,
        rather than using the same global latent features for all objects.
        """
        if latent_features is None or len(latent_features) == 0:
            raise RuntimeError(
                "Local embeddings were requested but no latent feature hooks have produced outputs yet. "
                "Check that latent_features_layers points to valid layers."
            )

        if isinstance(image, Image.Image):
            image_width, image_height = image.size
        elif isinstance(image, np.ndarray):
            image_height, image_width = image.shape[:2]
        elif isinstance(image, torch.Tensor):
            image_height = int(image.shape[-2])
            image_width = int(image.shape[-1])
        elif isinstance(image, str):
            with Image.open(image) as image_file:
                image_width, image_height = image_file.size
        else:
            raise TypeError(
                f"Unsupported image type for local feature extraction: {type(image)}"
            )

        image_width = max(int(image_width), 1)
        image_height = max(int(image_height), 1)

        local_latent_features_list = []
        for layer_name in self.latent_features_layers:
            layer_latent_features = latent_features[layer_name]
            feature_map_height = layer_latent_features.shape[-2]
            feature_map_width = layer_latent_features.shape[-1]
            target_height, target_width = self._get_local_latent_target_size(
                feature_map_height, feature_map_width
            )

            pooled_features = []
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box.astype(int)

                # Scale to feature map space
                def scale(coord, max_coord, feature_map_size):
                    return max(
                        0,
                        min(
                            feature_map_size,
                            int(round(coord * feature_map_size / max_coord)),
                        ),
                    )

                x1 = scale(x1, image_width, feature_map_width)
                x2 = scale(x2, image_width, feature_map_width)
                y1 = scale(y1, image_height, feature_map_height)
                y2 = scale(y2, image_height, feature_map_height)

                # Expand to the minimum target size, centered on the box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                crop_width = max(x2 - x1, target_width)
                crop_height = max(y2 - y1, target_height)
                x1, x2 = self._centered_bounds(cx, crop_width, feature_map_width)
                y1, y2 = self._centered_bounds(cy, crop_height, feature_map_height)

                cropped_feature = layer_latent_features[:, :, y1:y2, x1:x2]
                pooled_feature = torch.nn.functional.adaptive_avg_pool2d(
                    cropped_feature, (target_height, target_width)
                )
                # pooled_feature: (1, C, Th, Tw) -> squeeze batch dim to (C, Th, Tw)
                pooled_features.append(pooled_feature.squeeze(0))

            if pooled_features:
                # stack -> (O, C, Th, Tw)
                layer_tensor = torch.stack(pooled_features, dim=0)
                local_latent_features_list.append(layer_tensor)
            else:
                n_channels = layer_latent_features.shape[1]
                local_latent_features_list.append(
                    torch.zeros(
                        (len(boxes_xyxy), n_channels, target_height, target_width),
                        dtype=torch.float32,
                    )
                )

        # Plot local/global features and image+bbox for each detected object (debug only).
        if self.verbose:
            n_layers = len(self.latent_features_layers)
            for obj_idx, track_idx, box in zip(
                range(len(boxes_xyxy)), object_ids, boxes_xyxy
            ):
                fig, axes = plt.subplots(n_layers, 3, figsize=(18, 5 * n_layers))
                axes = np.atleast_2d(axes)

                for layer_idx, layer_name in enumerate(self.latent_features_layers):
                    row_axes = axes[layer_idx]
                    layer_latent_map = latent_features[layer_name][0]
                    global_map = layer_latent_map.mean(dim=0).cpu().numpy()
                    target_height, target_width = self._get_local_latent_target_size(
                        layer_latent_map.shape[-2], layer_latent_map.shape[-1]
                    )

                    layer_local_features = local_latent_features_list[layer_idx]
                    n_channels = layer_latent_map.shape[0]

                    if layer_local_features.shape[0] > obj_idx:
                        # layer_local_features[obj_idx]: (C, Th, Tw)
                        local_tensor = layer_local_features[obj_idx]
                        # average over channels for spatial heatmap
                        local_map = local_tensor.mean(dim=0).cpu().numpy()
                        row_axes[0].imshow(
                            local_map,
                            cmap="viridis",
                            vmin=global_map.min(),
                            vmax=global_map.max(),
                        )
                        row_axes[0].set_title(
                            f"object {obj_idx}: local avg ({layer_name}, {target_height}x{target_width})"
                        )
                        row_axes[0].axis("off")
                    else:
                        row_axes[0].text(
                            0.5,
                            0.5,
                            "No local features",
                            ha="center",
                            va="center",
                        )
                        row_axes[0].axis("off")

                    row_axes[1].imshow(
                        global_map,
                        cmap="viridis",
                        vmin=global_map.min(),
                        vmax=global_map.max(),
                    )
                    row_axes[1].set_title(f"{layer_name} global avg")
                    row_axes[1].axis("off")

                    frame_to_plot = image
                    row_axes[2].imshow(frame_to_plot)
                    x1, y1, x2, y2 = box
                    rect = Rectangle(
                        (x1, y1),
                        max(1.0, x2 - x1),
                        max(1.0, y2 - y1),
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                    row_axes[2].add_patch(rect)
                    row_axes[2].set_title(
                        f"id={track_idx}, cls={int(class_ids[obj_idx])}, conf={float(confidences[obj_idx]):.2f}"
                    )
                    row_axes[2].axis("off")

                plt.tight_layout()
                plt.show()

        # For return: for each layer tensor (O, C, Th, Tw) compute spatial mean -> (O, C)
        per_layer_feats = [
            (
                layer_tensor.mean(dim=[2, 3])
                if isinstance(layer_tensor, torch.Tensor)
                else torch.tensor(layer_tensor).mean(dim=[2, 3])
            )
            for layer_tensor in local_latent_features_list
        ]
        # concatenate channel dims across layers -> (O, D_local)
        concatenated_local_latent_features = (
            torch.cat(per_layer_feats, dim=1).cpu().numpy()
        )

        return (
            concatenated_local_latent_features.astype(np.float32),
            concatenated_local_latent_features.shape[1],
        )

    def _get_local_latent_target_size(
        self,
        feature_map_height: int,
        feature_map_width: int,
    ) -> tuple[int, int]:
        target_height = max(
            1, int(round(feature_map_height * self.local_latent_feature_fraction))
        )
        target_width = max(
            1, int(round(feature_map_width * self.local_latent_feature_fraction))
        )
        return target_height, target_width

    @staticmethod
    def _centered_bounds(center: float, size: int, max_size: int) -> tuple[int, int]:
        size = max(1, min(int(size), int(max_size)))
        start = int(round(center - size / 2.0))
        start = max(0, min(start, max_size - size))
        end = start + size
        return start, end
