"""Weak / strong TorchVision detectors for in-repo detection regeneration.

Only the two models the paper uses are supported:
  - weak   : fasterrcnn_mobilenet_v3_large_fpn   (full-res FPN)
  - strong : fasterrcnn_resnet50_fpn_v2

Both load COCO-pretrained weights (auto-downloaded on first run) and use the
weights' default transform (no manual resize). Class filtering is applied per
detection by 1-based COCO id.
"""

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

from config import COCO_CLASSES, get_device

# model_key -> (constructor, weights enum)
_MODEL_BUILDERS = {
    "weak":   (fasterrcnn_mobilenet_v3_large_fpn,
               FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    "strong": (fasterrcnn_resnet50_fpn_v2,
               FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
}


class Detector:
    """Run a TorchVision Faster R-CNN and return filtered detections.

    Args:
        model_key: "weak" or "strong".
        conf_threshold: minimum confidence to keep (0.0 keeps all that the model
            itself emits; torchvision floors at ~0.05 internally).
        class_filter: set of 1-based COCO class ids to keep, or None for all.
    """

    def __init__(self, model_key, conf_threshold, class_filter=None):
        if model_key not in _MODEL_BUILDERS:
            raise ValueError(
                f"Unknown model '{model_key}'. Choose from: {list(_MODEL_BUILDERS)}"
            )
        self.device = get_device()
        self.conf_threshold = conf_threshold
        self.class_filter = class_filter

        builder, weights = _MODEL_BUILDERS[model_key]
        self.model = builder(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.transform = weights.transforms()

        print(f"Using device: {self.device.upper()}")
        print(f"Model: {model_key} ({builder.__name__})")

    def _filter(self, pred):
        """Convert one model prediction dict to a list of standard detections.

        Returns list of (left, top, width, height, conf, class_name).
        """
        boxes = pred["boxes"].cpu()
        labels = pred["labels"].cpu()
        scores = pred["scores"].cpu()

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            label_id = int(label)
            conf = float(score)
            if conf < self.conf_threshold:
                continue
            if self.class_filter is not None and label_id not in self.class_filter:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                (x1, y1, x2 - x1, y2 - y1, conf, COCO_CLASSES[label_id])
            )
        return detections

    def detect_batch(self, image_paths):
        """Run detection on a batch of images.

        Returns one list of detections per image, in the same order.
        """
        img_tensors = [
            self.transform(Image.open(p).convert("RGB")) for p in image_paths
        ]
        batch = torch.stack(img_tensors).to(self.device)

        with torch.no_grad():
            predictions = self.model(batch)

        return [self._filter(pred) for pred in predictions]
