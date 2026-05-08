"""YOLO nano + ByteTrack object tracker with configurable feature output.

This module exposes a lightweight tracker that accepts a single RGB image frame
and returns only dataset-style tensors:

- features: (1, O, F)
- mask: (1, O, 1)

The feature tensor is assembled from the requested components in this order:
1. bounding boxes as [center_x, center_y, width, height]
2. confidences
3. class IDs
4. embeddings
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO


class ObjectTracker:
    """YOLO nano + ByteTrack tracker with configurable feature components."""

    FEATURES: Dict[str, Any] = {
        "bboxes": "_get_bboxes_component",
        "confidences": "_get_confidences_component",
        "class_ids": "_get_class_ids_component",
        "embeddings": "_get_embeddings_component",
    }

    def __init__(
        self,
        model_name: Union[str, Path] = "yolo11n.pt",
        tracker_cfg: Union[str, Path] = "bytetrack.yaml",
        device: Optional[Union[str, int, torch.device]] = None,
        conf: float = 0.25,
        imgsz: int = 640,
        classes: Optional[Sequence[int]] = None,
        max_det: int = 300,
        feature_components: Sequence[str] = ("bboxes",),
        embedding_layers: Optional[Sequence[str]] = ["model.16", "model.19"], # Defaults to P3 and P4 features
        verbose: bool = False,
    ) -> None:
        self.model_name = str(model_name)
        self.tracker_cfg = str(tracker_cfg)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.classes = list(classes) if classes is not None else None
        self.max_det = max_det
        self.embedding_layers = embedding_layers
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
        if "embeddings" in self.feature_components and (self.embedding_layers is None or len(self.embedding_layers) == 0):
            raise ValueError(
                "feature_components includes 'embeddings' but embedding_layers is None."
            )

        self._tracker_model = YOLO(self.model_name)
        self._last_embeddings: Optional[Dict[str, torch.Tensor]] = None

        if self.embedding_layers is not None:
            self._register_embedding_hooks()

    def _register_embedding_hooks(self) -> None:
        """Register a persistent forward hook that captures one embedding per frame."""

        def get_hook_fn(layer_name: str):
            def hook_fn(module: Any, input: Any, output: Any) -> None:
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        f"Expected output of layer '{layer_name}' to be a torch.Tensor, but got {type(output)}."
                    )

                embedding = output.detach().cpu().float()
                self._last_embeddings[layer_name] = embedding
            return hook_fn

        self._last_embeddings = {}
        hooked_layers = {}
        backbone = self._tracker_model.model
        for layer_name in self.embedding_layers:
            layer = self._tracker_model.model.get_submodule(layer_name)
            layer.register_forward_hook(get_hook_fn(layer_name))
            hooked_layers[layer_name] = layer.__class__.__name__
        if self.verbose:
            print(f"Hooked layers for embeddings: {hooked_layers}")

    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        num_objects: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Track one RGB frame and return only features and mask.

        Features are assembled in the order declared by feature_components.
        """

        for layer_name in self.embedding_layers:
            self._last_embeddings[layer_name] = None

        track_kwargs: Dict[str, Any] = {
            "source": image,
            "persist": True,
            "tracker": self.tracker_cfg,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device is not None:
            track_kwargs["device"] = self.device
        if self.classes is not None:
            track_kwargs["classes"] = self.classes

        results = self._tracker_model.track(**track_kwargs)

        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        confidences = np.zeros((0,), dtype=np.float32)
        class_ids = np.zeros((0,), dtype=np.int64)

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
                class_ids = (
                    boxes.cls.detach().cpu().numpy().astype(np.int64)
                    if boxes.cls is not None
                    else np.zeros((len(boxes_xyxy),), dtype=np.int64)
                )

        components: list[np.ndarray] = []
        valid_objects = int(len(boxes_xyxy))

        for feature in self.feature_components:
            component = getattr(self, self.FEATURES[feature])(
                boxes_xyxy=boxes_xyxy,
                confidences=confidences,
                class_ids=class_ids,
                embedding=self._last_embeddings,
            )
            components.append(component)

        if components:
            features_arr = np.concatenate(components, axis=1).astype(np.float32)
        else:
            features_arr = np.zeros((valid_objects, 0), dtype=np.float32)

        if num_objects is not None:
            feature_dim = features_arr.shape[1]
            padded = np.zeros((num_objects, feature_dim), dtype=np.float32)
            copy_count = min(valid_objects, num_objects)
            if copy_count > 0:
                padded[:copy_count] = features_arr[:copy_count]
            features_arr = padded
            mask = np.zeros((1, num_objects, 1), dtype=np.float32)
            mask[:, :copy_count, :] = 1.0
        else:
            features_arr = features_arr[:valid_objects]
            mask = np.zeros((1, valid_objects, 1), dtype=np.float32)
            if valid_objects > 0:
                mask[:] = 1.0

        features = torch.from_numpy(features_arr).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).float()

        return {
            "features": features,
            "mask": mask_tensor,
        }

    def _get_bboxes_component(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        embedding: Optional[torch.Tensor],
    ) -> np.ndarray:
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
            return np.stack([center_x, center_y, width, height], axis=1).astype(
                np.float32
            )
        else:
            return np.zeros((0, 4), dtype=np.float32)

    def _get_confidences_component(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        embedding: Optional[torch.Tensor],
    ) -> np.ndarray:
        return confidences.reshape(-1, 1).astype(np.float32)

    def _get_class_ids_component(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        embedding: Optional[torch.Tensor],
    ) -> np.ndarray:
        return class_ids.reshape(-1, 1).astype(np.float32)

    def _get_embeddings_component(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        embedding: Optional[torch.Tensor],
    ) -> np.ndarray:
        if self._last_embeddings is None or len(self._last_embeddings) == 0:
            raise RuntimeError(
                "Embeddings were requested but no embedding hooks have produced outputs yet. "
                "Check that embedding_layers points to valid layers."
            )

        missing_layers = [layer for layer, value in self._last_embeddings.items() if value is None]
        if missing_layers:
            raise RuntimeError(
                f"Embeddings were requested but no embedding was captured for {missing_layers[0]}. "
                "Check that embedding_layers points to valid layers."
            )

        embeddings = []
        for layer_name in self.embedding_layers:
            layer_embedding = self._last_embeddings[layer_name]

            # Turn the embedding into (1, C)
            layer_embedding = layer_embedding.mean(dim=[2, 3])

            embeddings.append(layer_embedding)

        concatenated_embedding = torch.cat(embeddings, dim=1).cpu().numpy()
        embedding_row = np.repeat(concatenated_embedding.reshape(1, -1), len(boxes_xyxy), axis=0)
        return embedding_row.astype(np.float32)
