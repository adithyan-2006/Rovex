"""
Utility helpers and High-Level Pipeline for Road Damage Detection.
Bundles sample generator, image I/O, and one-stop RoadDamagePipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from .detector import RoadDamageDetector
from .severity import SeverityAnalyzer, RoadAssessmentReport
from .visualizer import RoadDamageVisualizer
from .reporter import InspectionReporter


class RoadDamagePipeline:
    """One-stop pipeline uniting detection, severity assessment, and visualization."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        perspective_weight: float = 1.5,
        device: str = "cpu",
        show_hud: bool = True,
    ):
        self.detector = RoadDamageDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        self.analyzer = SeverityAnalyzer(perspective_weight=perspective_weight)
        self.visualizer = RoadDamageVisualizer(show_hud=show_hud)
        self.reporter = InspectionReporter()

    def process_image(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Executes end-to-end detection, severity assessment, and visual overlay.

        Returns dictionary with:
        - "report": RoadAssessmentReport instance
        - "annotated_image": np.ndarray (BGR format)
        - "annotated_pil": PIL Image (RGB format)
        - "detections": list of raw detection dictionaries
        - "dataframe": pandas DataFrame of defects
        """
        # Load standard numpy BGR
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_input}")
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        h, w = img_bgr.shape[:2]

        # 1. Run detection
        detections = self.detector.detect(img_bgr, conf=conf, iou=iou, mode=mode)

        # 2. Run severity assessment
        report = self.analyzer.assess_image(detections, (h, w), image=img_bgr)

        # 3. Create visual annotations
        annotated_bgr = self.visualizer.annotate(img_bgr, report)
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

        # 4. Tabular DataFrame
        df = self.reporter.to_dataframe(report)

        return {
            "report": report,
            "annotated_image": annotated_bgr,
            "annotated_pil": annotated_pil,
            "detections": detections,
            "dataframe": df,
            "is_cv_fallback": self.detector.is_cv_fallback,
        }


def generate_synthetic_road_sample(
    output_path: Union[str, Path],
    scenario: str = "pothole_and_cracks",
) -> Path:
    """
    Synthesizes a realistic asphalt road image with defects for instant local testing
    without requiring internet downloads.
    Scenarios: 'severe_pothole', 'fatigue_cracks', 'pothole_and_cracks', 'clean_road'
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = 960, 640
    # Generate realistic asphalt road gradient & noise
    # Horizon is around y = 200
    base_color = 75  # dark asphalt gray
    road = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky/background
    road[0:180, :] = [180, 160, 140]  # muted sky

    # Road surface with perspective shading (darker near camera)
    for y in range(180, h):
        intensity = int(base_color - 15 * (y - 180) / (h - 180))
        road[y, :] = [intensity, intensity, intensity + 2]

    # Asphalt grain texture noise
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    road_noise = np.clip(road.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    road[180:h, :] = road_noise[180:h, :]

    # Paint lane marking lines (perspective)
    cv2.line(road, (480, 180), (320, h), (210, 210, 215), 4)  # Left white stripe
    cv2.line(road, (480, 180), (640, h), (40, 180, 220), 4)   # Right yellow stripe

    if scenario in ("severe_pothole", "pothole_and_cracks"):
        # Draw a realistic dark pothole depression with broken edge texture
        px, py = 450, 430
        axes = (70, 45)
        # Outer rough rim
        cv2.ellipse(road, (px, py), (axes[0] + 8, axes[1] + 6), -5, 0, 360, (40, 40, 42), -1)
        # Dark inner depression
        cv2.ellipse(road, (px, py), axes, -5, 0, 360, (18, 18, 20), -1)
        # Deep shadow center
        cv2.ellipse(road, (px - 5, py + 4), (int(axes[0] * 0.6), int(axes[1] * 0.5)), -5, 0, 360, (8, 8, 10), -1)

    if scenario in ("fatigue_cracks", "pothole_and_cracks"):
        # Draw longitudinal / transverse crack networks
        crack_color = (25, 25, 28)
        pts1 = np.array([[280, 400], [290, 430], [285, 470], [305, 510], [295, 550]], np.int32)
        cv2.polylines(road, [pts1], False, crack_color, 3)

        # Branching cracks
        pts2 = np.array([[290, 430], [330, 440], [360, 465], [350, 490]], np.int32)
        cv2.polylines(road, [pts2], False, crack_color, 2)

        pts3 = np.array([[285, 470], [260, 485], [245, 510]], np.int32)
        cv2.polylines(road, [pts3], False, crack_color, 2)

    if scenario == "severe_pothole":
        # Additional secondary pothole
        cv2.ellipse(road, (610, 370), (40, 25), 8, 0, 360, (22, 22, 24), -1)
        cv2.ellipse(road, (608, 372), (28, 16), 8, 0, 360, (10, 10, 12), -1)

    # Slight blur to blend naturally
    road = cv2.GaussianBlur(road, (3, 3), 0)
    cv2.imwrite(str(out), road)
    return out
