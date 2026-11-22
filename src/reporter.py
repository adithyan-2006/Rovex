"""
Reporting and Export Module for Road Damage Assessment.
Formats assessment reports into structured dictionaries, pandas DataFrames,
JSON exports, and CSV summaries.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from .severity import RoadAssessmentReport, DefectAssessment


class InspectionReporter:
    """Handles tabular formatting, JSON generation, and CSV export of inspections."""

    @staticmethod
    def to_dataframe(report: RoadAssessmentReport) -> pd.DataFrame:
        """Converts defects list of a report into a pandas DataFrame."""
        if not report.defects:
            return pd.DataFrame(
                columns=[
                    "Defect ID",
                    "Damage Type",
                    "Confidence (%)",
                    "Severity Grade",
                    "Severity Score",
                    "Relative Area (%)",
                    "Perspective Factor",
                    "Bounding Box",
                    "Recommended Action",
                ]
            )

        rows = []
        for d in report.defects:
            rows.append(
                {
                    "Defect ID": d.defect_id,
                    "Damage Type": d.display_name,
                    "Confidence (%)": round(d.confidence * 100, 1),
                    "Severity Grade": d.severity_grade.value,
                    "Severity Score": round(d.severity_score, 1),
                    "Relative Area (%)": round(d.relative_area_pct, 2),
                    "Perspective Factor": round(d.perspective_factor, 2),
                    "Bounding Box": f"[{d.box[0]}, {d.box[1]}, {d.box[2]}, {d.box[3]}]",
                    "Recommended Action": d.repair_recommendation,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def save_json(
        report: RoadAssessmentReport,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves detailed assessment report to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = report.to_dict()
        if metadata:
            data["metadata"] = metadata

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out

    @staticmethod
    def save_csv(
        report: RoadAssessmentReport,
        output_path: Union[str, Path],
    ) -> Path:
        """Exports defect list to a CSV file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        df = InspectionReporter.to_dataframe(report)
        df.to_csv(out, index=False)
        return out

    @staticmethod
    def generate_batch_summary(
        batch_results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Creates aggregate summary DataFrame from batch inspection results."""
        rows = []
        for item in batch_results:
            rep: RoadAssessmentReport = item["report"]
            filename = item.get("filename", "unknown")
            rows.append(
                {
                    "Image Name": filename,
                    "Road Condition": rep.road_condition.value,
                    "PCI (0-100)": round(rep.pavement_condition_index, 1),
                    "Total Defects": rep.total_defects,
                    "Potholes": rep.pothole_count,
                    "Cracks": rep.crack_count,
                    "Critical": rep.critical_defects_count,
                    "High": rep.high_defects_count,
                    "Medium": rep.medium_defects_count,
                    "Low": rep.low_defects_count,
                    "Action Summary": rep.action_summary,
                }
            )
        return pd.DataFrame(rows)
