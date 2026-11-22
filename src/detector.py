"""
AI Detection Engine for Road Damage.
Wraps YOLOv8 trained on the RDD2022 (Road Damage Detection) benchmark,
with automatic weight management and CV heuristic fallback.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

# Default Hugging Face repository and weights
DEFAULT_HF_REPO = "dronefreak/rdd2022-yolov8s"
DEFAULT_WEIGHTS_FILENAME = "best.pt"
LOCAL_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# Standard RDD2022 class mappings
RDD_CLASS_MAP = {
    0: "D00",  # Longitudinal Crack
    1: "D10",  # Transverse Crack
    2: "D20",  # Alligator Crack
    3: "D40",  # Pothole
    4: "D43",  # Marking blur / crosswalk
}

RDD_HUMAN_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "D43": "Surface Marking Blur",
}


class RoadDamageDetector:
    """Detects road distress (potholes, alligator cracks, linear cracks) using YOLOv8."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        hf_repo: str = DEFAULT_HF_REPO,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Local path to .pt weights. If None, auto-downloads from HF.
            hf_repo: HuggingFace repository ID if model_path is None.
            conf_threshold: Minimum confidence score to accept detection.
            iou_threshold: Non-Maximum Suppression (NMS) IoU overlap threshold.
            device: 'cpu' or 'cuda'.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.hf_repo = hf_repo
        self.model = None
        self.model_path = model_path
        self.is_cv_fallback = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads YOLOv8 weights or sets up CV fallback if unavailable."""
        LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        target_path: Optional[Path] = None

        if self.model_path and Path(self.model_path).is_file():
            target_path = Path(self.model_path)
        else:
            # Check for existing weights in local weights dir
            direct_default = LOCAL_WEIGHTS_DIR / DEFAULT_WEIGHTS_FILENAME
            local_default = LOCAL_WEIGHTS_DIR / f"{self.hf_repo.replace('/', '_')}_{DEFAULT_WEIGHTS_FILENAME}"
            if direct_default.is_file():
                target_path = direct_default
            elif local_default.is_file():
                target_path = local_default
            else:
                # Try downloading from Hugging Face
                try:
                    from huggingface_hub import hf_hub_download
                    print(f"[Rovex] Fetching weights from Hugging Face: {self.hf_repo}...")
                    downloaded = hf_hub_download(
                        repo_id=self.hf_repo,
                        filename=DEFAULT_WEIGHTS_FILENAME,
                        local_dir=str(LOCAL_WEIGHTS_DIR),
                        local_dir_use_symlinks=False,
                    )
                    target_path = Path(downloaded)
                    print(f"[Rovex] Model weights successfully cached at: {target_path}")
                except Exception as e:
                    print(f"[Rovex] Warning: Could not download weights ({e}).")

        # Attempt to load Ultralytics YOLO
        if target_path and target_path.is_file():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(target_path))
                self.model_path = str(target_path)
                print(f"[Rovex] YOLOv8 model loaded successfully from {target_path}")
                return
            except Exception as e:
                print(f"[Rovex] Warning: Failed to initialize YOLO ({e}). Engaging CV fallback.")

        # Fallback to computer vision heuristic detector if YOLO or weights aren't loaded
        self.is_cv_fallback = True
        print("[Rovex] Operating in Computer Vision Heuristic Fallback mode.")

    def detect(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        mode: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Runs road damage detection on an image.
        Args:
            image_input: image array or filepath.
            conf: confidence threshold.
            iou: IoU NMS threshold.
            mode: 'auto' (YOLO with fallback/supplement), 'yolo' (YOLO only), 
                  'cv' (Computer Vision edge/darkness only), 'hybrid' (ensemble).
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold

        # Load image array (RGB for PIL, BGR for CV)
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image file: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_bgr = image_input
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if self.is_cv_fallback or mode == "cv":
            return self._detect_cv_heuristic(img_bgr, conf)

        if mode == "yolo":
            return self._detect_yolo(img_rgb, conf, iou)

        # Mode is 'auto' or 'hybrid'
        yolo_dets = self._detect_yolo(img_rgb, conf, iou)
        if mode == "auto":
            if yolo_dets:
                return yolo_dets
            # If YOLO detected nothing (e.g. synthetic road), try CV heuristic
            return self._detect_cv_heuristic(img_bgr, conf)

        # Hybrid: combine both without overlaps
        cv_dets = self._detect_cv_heuristic(img_bgr, conf)
        combined = list(yolo_dets)
        for cd in cv_dets:
            cx1, cy1, cx2, cy2 = cd["box"]
            c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            overlap = False
            for yd in yolo_dets:
                yx1, yy1, yx2, yy2 = yd["box"]
                ix1, iy1 = max(cx1, yx1), max(cy1, yy1)
                ix2, iy2 = min(cx2, yx2), min(cy2, yy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / float(c_area) > 0.35:
                        overlap = True
                        break
            if not overlap:
                combined.append(cd)
        return combined

    def _detect_yolo(
        self,
        img_rgb: np.ndarray,
        conf: float,
        iou: float,
    ) -> List[Dict[str, Any]]:
        """Runs inference using Ultralytics YOLOv8."""
        results = self.model.predict(
            source=img_rgb,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Inspect model class names mapping
        names = getattr(self.model, "names", {})

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(box.cls[0].cpu().item())
            confidence = float(box.conf[0].cpu().item())

            # Map class id
            raw_class = names.get(cls_id, RDD_CLASS_MAP.get(cls_id, f"Class_{cls_id}"))

            # Determine human-friendly name
            if raw_class in RDD_HUMAN_NAMES:
                display_name = RDD_HUMAN_NAMES[raw_class]
            elif "pothole" in str(raw_class).lower():
                display_name = "Pothole"
            elif "crack" in str(raw_class).lower():
                display_name = "Road Crack"
            else:
                display_name = str(raw_class)

            detections.append(
                {
                    "box": xyxy,
                    "confidence": confidence,
                    "class": str(raw_class),
                    "class_name": display_name,
                }
            )

        return detections

    def _detect_cv_heuristic(
        self,
        img_bgr: np.ndarray,
        conf: float,
    ) -> List[Dict[str, Any]]:
        """
        Computer Vision heuristic fallback for detecting road distress when DL weights
        are unavailable or during offline testing.
        Uses adaptive multi-scale edge filtering & shadow depression analysis.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Focus analysis on the lower 75% of the frame (road surface)
        road_roi_y = int(h * 0.25)
        gray_road = gray[road_roi_y:h, 0:w]

        # 1. Pothole depression detection: dark concave regions
        blurred = cv2.GaussianBlur(gray_road, (11, 11), 0)
        # Threshold for dark depressions significantly lower than mean pavement
        mean_lum = np.mean(blurred)
        thresh_dark = max(20, int(mean_lum * 0.62))
        _, dark_mask = cv2.threshold(blurred, thresh_dark, 255, cv2.THRESH_BINARY_INV)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dark_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_clean = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, kernel)

        contours_pothole, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        min_area = (w * h) * 0.0015  # At least 0.15% of image
        max_area = (w * h) * 0.25

        for cnt in contours_pothole:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / float(bh)
                # Potholes are generally elliptical / rounded (aspect 0.4 to 3.0)
                if 0.35 < aspect < 3.2:
                    abs_y1 = by + road_roi_y
                    abs_y2 = abs_y1 + bh
                    score = min(0.92, max(0.55, 0.60 + (area / (w * h)) * 10.0))
                    if score >= conf:
                        detections.append(
                            {
                                "box": [bx, abs_y1, bx + bw, abs_y2],
                                "confidence": round(score, 3),
                                "class": "D40",
                                "class_name": "Pothole",
                            }
                        )

        # 2. Crack detection: high-gradient edges
        edges = cv2.Canny(gray_road, 50, 150)
        crack_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_edges = cv2.dilate(edges, crack_kernel, iterations=1)

        contours_cracks, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_cracks:
            area = cv2.contourArea(cnt)
            if (w * h) * 0.002 < area < (w * h) * 0.15:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Check if it overlaps existing pothole detection
                abs_y1 = by + road_roi_y
                abs_y2 = abs_y1 + bh
                overlaps = False
                for d in detections:
                    dx1, dy1, dx2, dy2 = d["box"]
                    if not (bx + bw < dx1 or bx > dx2 or abs_y2 < dy1 or abs_y1 > dy2):
                        overlaps = True
                        break

                if not overlaps:
                    aspect = bw / float(bh)
                    if aspect > 2.0:
                        cls_name = "Transverse Crack"
                        cls_code = "D10"
                    elif aspect < 0.5:
                        cls_name = "Longitudinal Crack"
                        cls_code = "D00"
                    else:
                        cls_name = "Alligator Crack"
                        cls_code = "D20"

                    score = min(0.85, max(0.45, 0.50 + (area / (w * h)) * 8.0))
                    if score >= conf:
                        detections.append(
                            {
                                "box": [bx, abs_y1, bx + bw, abs_y2],
                                "confidence": round(score, 3),
                                "class": cls_code,
                                "class_name": cls_name,
                            }
                        )

        return detections
