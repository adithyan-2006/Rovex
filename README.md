# 🛣️ Rovex: Road Damage Detection & Severity Assessment System

An end-to-end, production-ready computer vision system to detect potholes and road cracks (longitudinal, transverse, alligator) and quantify their structural severity using civil-engineering-grounded metrics (ASTM D6433 Pavement Condition Index).

---

## 🌟 Key Features

- **Multi-Class Road Distress Detection**:
  - **D40**: Potholes & surface depressions (high road safety hazard)
  - **D20**: Alligator / Fatigue cracking (indicates subgrade base failure)
  - **D10**: Transverse cracks (thermal / joint expansion)
  - **D00**: Longitudinal cracks (lane & wheel-path distress)
- **Civil Engineering Severity Assessment**:
  - **Perspective Correction**: Compensates for road camera angles so distant defects are proportionally scaled to their true pavement footprint.
  - **Depth / Shadow Contrast**: Distinguishes shallow abrasions from deep, axle-damaging pothole cavities.
  - **ASTM D6433 PCI Metric**: Computes a calibrated **0–100 Pavement Condition Index (PCI)** and categorizes road health into *Good*, *Fair*, *Poor*, or *Critical*.
  - **Actionable Maintenance Guidance**: Generates specific civil engineering repair recommendations (e.g. *Polymer crack sealing*, *Full-depth asphalt patching*, *Base reconstruction*).
- **Interactive Streamlit Web Dashboard**:
  - Single image inspection with side-by-side original vs. annotated view.
  - Pre-bundled realistic benchmark samples for immediate 1-click testing.
  - Batch road survey tool with aggregate network statistics and CSV export.
  - Dashcam & mobile video inspection with real-time frame-by-frame PCI tracking charts.
- **Headless Command-Line Interface (CLI)**:
  - Batch process folders of road imagery or single files with automated report generation.
- **Robust & Autonomous**:
  - Uses fine-tuned **YOLOv8** weights from the international **RDD2022** benchmark (`dronefreak/rdd2022-yolov8s`).
  - Automatic model weight caching and built-in Computer Vision heuristic fallback for offline or zero-configuration environments.

---

## 🏗️ System Architecture

```
Rovex/
├── app.py                  # Interactive Streamlit Web Dashboard
├── cli.py                  # Command-Line batch processing tool
├── requirements.txt        # Python package dependencies
├── src/
│   ├── __init__.py         # Package initialization
│   ├── detector.py         # YOLOv8 & CV fallback road damage detector
│   ├── severity.py         # Perspective-aware ASTM D6433 severity engine
│   ├── visualizer.py       # Color-coded bounding boxes & telemetry HUD
│   ├── reporter.py         # DataFrame, JSON, and CSV export handlers
│   └── utils.py            # RoadDamagePipeline and benchmark generator
├── data/
│   └── samples/            # Pre-bundled road benchmark images
├── tests/
│   ├── __init__.py
│   └── test_severity.py    # Unit tests for severity math & logic
└── README.md               # System documentation
```

---

## 🚀 Quick Start

### 1. Installation

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard

```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`. You can test immediately using the built-in preset benchmark images or upload your own road photos!

### 3. Run the CLI Tool

Inspect the default sample images and save results to `results/`:
```bash
python cli.py --source data/samples --output results
```

Inspect a specific custom road image:
```bash
python cli.py --source path/to/road_photo.jpg --output results --conf 0.25
```

---

## 📊 Severity Classification & Engineering Rubric

| Severity Grade | Severity Score (0-100) | Defect Footprint | Recommended Civil Engineering Action |
|:---|:---:|:---|:---|
| **Low (Minor)** | 0 – 31 | Hairline cracks, shallow distress (<1% area) | Routine monitoring; preventative surface seal |
| **Medium (Moderate)** | 32 – 54 | Noticeable crack or small pothole (1-3% area) | Rubberized asphalt crack fill / mastic patch |
| **High (Severe)** | 55 – 74 | Large pothole, deep depression, dense fatigue grid | Full-depth patch compaction within 48h |
| **Critical** | 75 – 100 | Deep tire-hazard pothole (>3.5% effective area) | Immediate road closure / emergency excavation |

### Pavement Condition Index (PCI) Scale:
- **85 – 100 (Good)**: Pristine to minor surface aging; routine maintenance.
- **60 – 84 (Fair)**: Moderate surface distress; scheduled repair.
- **35 – 59 (Poor)**: Significant distress; prioritized resurfacing/patching.
- **0 – 34 (Critical)**: Severe structural deterioration; urgent intervention required.

---

## 🧪 Testing

Run automated unit and functional tests:

```bash
python -m unittest discover -s tests
```

---

## 📄 License
Open-source under the Apache 2.0 / MIT License. Pretrained models from RDD2022 are subject to their respective open research licenses.
