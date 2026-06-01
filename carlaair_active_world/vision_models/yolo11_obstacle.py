from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


class UltralyticsObstacleDetector:
    """Optional YOLO11-compatible forward-obstacle detector for RGB frames."""

    obstacle_classes = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    traffic_classes = {"traffic light", "stop sign"}

    def __init__(self, model_path: str, confidence: float = 0.35) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = float(confidence)

    def _class_name(self, class_id: int) -> str:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return str(names.get(int(class_id), int(class_id)))
        if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
            return str(names[int(class_id)])
        return str(class_id)

    def _iter_boxes(self, result: Any) -> Iterable[tuple[str, float, tuple[float, float, float, float]]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        cls_values = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
        conf_values = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
        xyxy_values = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
        return [
            (
                self._class_name(int(cls_id)),
                float(conf),
                tuple(float(v) for v in xyxy),
            )
            for cls_id, conf, xyxy in zip(cls_values, conf_values, xyxy_values)
        ]

    @staticmethod
    def _is_forward_obstacle(width: int, height: int, xyxy: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = xyxy
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        image_area = max(1.0, float(width * height))
        central = width * 0.30 <= center_x <= width * 0.70
        ahead = center_y >= height * 0.45 or box_area / image_area >= 0.035
        return bool(central and ahead)

    def predict(self, rgb: np.ndarray) -> Dict[str, Any]:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[0] < 8 or image.shape[1] < 8:
            return {"available": True, "obstacle": False, "reason": "invalid_rgb"}

        height, width = image.shape[:2]
        results = self.model.predict(image, conf=self.confidence, verbose=False, device="cpu")
        best: Dict[str, Any] = {
            "available": True,
            "obstacle": False,
            "traffic": False,
            "detections": 0,
            "traffic_detections": 0,
        }
        for result in results:
            for label, conf, xyxy in self._iter_boxes(result):
                if label in self.traffic_classes:
                    best["traffic"] = True
                    best["traffic_detections"] = int(best["traffic_detections"]) + 1
                    if conf >= float(best.get("traffic_confidence", 0.0)):
                        best.update(
                            {
                                "traffic_label": label,
                                "traffic_confidence": float(conf),
                                "traffic_bbox_xyxy": [float(v) for v in xyxy],
                            }
                        )
                    continue
                if label not in self.obstacle_classes:
                    continue
                best["detections"] = int(best["detections"]) + 1
                if self._is_forward_obstacle(width, height, xyxy) and conf >= float(best.get("confidence", 0.0)):
                    best.update(
                        {
                            "obstacle": True,
                            "label": label,
                            "confidence": float(conf),
                            "bbox_xyxy": [float(v) for v in xyxy],
                        }
                    )
        return best
