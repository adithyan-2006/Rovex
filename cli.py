"""
Command-Line Interface (CLI) for Rovex Road Damage Detection & Severity Assessment.
Supports single image, batch directory processing, and report exports.
"""

import argparse
from pathlib import Path
import sys
import time

from src.utils import RoadDamagePipeline
from src.reporter import InspectionReporter


def setup_sample_data(sample_dir: Path) -> None:
    """Ensures sample road directory exists."""
    sample_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rovex - Road Damage Detection and Severity Assessment CLI"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/samples",
        help="Path to an image file or directory of road images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory to save inspection outputs and reports",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to custom YOLOv8 model weights (.pt)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "yolo", "cv", "hybrid"],
        help="Detection mode: 'auto' (YOLO+supplement), 'yolo' (AI only), 'cv' (edge/darkness only), 'hybrid' (ensemble)",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable road condition HUD banner on annotated images",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate synthetic benchmark road sample images and exit",
    )

    args = parser.parse_args()

    # If requested to generate sample images
    sample_dir = Path("data/samples")
    if args.generate_samples:
        setup_sample_data(sample_dir)
        print("[Rovex] Sample images generation complete.")
        return

    # Check if source exists, if default data/samples is missing, auto-create
    source_path = Path(args.source)
    if not source_path.exists():
        if args.source == "data/samples":
            setup_sample_data(sample_dir)
        else:
            print(f"Error: Source path '{args.source}' not found.", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("      ROVEX - ROAD DAMAGE DETECTION & SEVERITY ASSESSMENT      ")
    print("=" * 65)
    print(f" Source:     {source_path}")
    print(f" Output:     {out_dir}")
    print(f" Confidence: {args.conf}")
    print("=" * 65 + "\n")

    # Initialize Pipeline
    print("[Rovex] Initializing AI models and severity engine...")
    t0 = time.time()
    pipeline = RoadDamagePipeline(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        show_hud=not args.no_hud,
    )
    init_time = time.time() - t0
    print(f"[Rovex] Pipeline ready in {init_time:.2f}s\n")

    # Collect images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = [f for f in source_path.iterdir() if f.suffix.lower() in image_exts]

    if not image_files:
        print(f"[Rovex] No image files found in {source_path}. Creating demo samples...")
        setup_sample_data(sample_dir)
        image_files = [f for f in sample_dir.iterdir() if f.suffix.lower() in image_exts]

    print(f"[Rovex] Processing {len(image_files)} image(s)...\n")

    batch_records = []

    for idx, img_path in enumerate(image_files, start=1):
        t_start = time.time()
        res = pipeline.process_image(img_path, mode=args.mode)
        elapsed = time.time() - t_start

        report = res["report"]
        batch_records.append({"filename": img_path.name, "report": report})

        # Save annotated image
        import cv2
        annotated_path = annotated_dir / f"scan_{img_path.name}"
        cv2.imwrite(str(annotated_path), res["annotated_image"])

        # Save individual JSON report
        json_path = out_dir / f"{img_path.stem}_report.json"
        InspectionReporter.save_json(report, json_path, metadata={"filename": img_path.name})

        # Print per-image console summary
        print(f"[{idx}/{len(image_files)}] {img_path.name} ({elapsed:.2f}s)")
        print(f"  Condition: {report.road_condition.value} | PCI: {report.pavement_condition_index:.1f}/100")
        print(f"  Defects:   {report.total_defects} (Potholes: {report.pothole_count}, Cracks: {report.crack_count})")
        print(f"  Severity:  Critical={report.critical_defects_count}, High={report.high_defects_count}, Med={report.medium_defects_count}, Low={report.low_defects_count}")
        print(f"  Action:    {report.action_summary}")
        print(f"  Output:    {annotated_path}\n")

    # Save Aggregate Batch CSV
    summary_df = InspectionReporter.generate_batch_summary(batch_records)
    summary_csv = out_dir / "batch_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("=" * 65)
    print("INSPECTION SUMMARY:")
    print(f"  Total Images Processed: {len(image_files)}")
    print(f"  Batch Summary CSV:      {summary_csv}")
    print(f"  Annotated Images:       {annotated_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
