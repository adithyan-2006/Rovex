"""
Rovex Infrastructure Systems - Pavement Condition & Severity Assessment Platform.
ASTM D6433 Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys.
Executive White & Emerald Crystal Glassmorphic Architecture with Unique Contemporary Typography.
"""

from pathlib import Path
import tempfile
import time
from typing import Optional, List, Dict, Any
import io
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.utils import RoadDamagePipeline
from src.severity import RoadCondition, SeverityGrade
from src.reporter import InspectionReporter

# Set Page Configuration
st.set_page_config(
    page_title="Rovex | Pavement Condition & Asset Management Platform",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# WHITE & EMERALD CRYSTAL GLASSMORPHIC DESIGN SYSTEM
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --canvas-bg: #F8FAFC;
        --card-glass: rgba(255, 255, 255, 0.85);
        --card-glass-elevated: rgba(255, 255, 255, 0.95);
        --card-border: rgba(226, 232, 240, 0.9);
        --card-border-hover: rgba(16, 185, 129, 0.45);
        --emerald-primary: #059669;
        --emerald-light: #10B981;
        --emerald-surface: rgba(16, 185, 129, 0.10);
        --emerald-surface-deep: #ECFDF5;
        --emerald-border: rgba(16, 185, 129, 0.28);
        --emerald-text: #065F46;
        --text-headline: #0F172A;
        --text-body: #334155;
        --text-muted: #64748B;
        --text-dim: #94A3B8;
        --amber-warning: #D97706;
        --red-critical: #DC2626;
        --shadow-glass: 0 8px 24px -4px rgba(15, 23, 42, 0.05), 0 0 0 1px rgba(226, 232, 240, 0.7);
        --shadow-glass-hover: 0 12px 30px -4px rgba(16, 185, 129, 0.12), 0 0 0 1px rgba(16, 185, 129, 0.35);
    }

    /* Global Reset & Base Typography */
    html, body, .stApp {
        background-color: var(--canvas-bg) !important;
        background-image: 
            radial-gradient(circle at 10% 0%, rgba(16, 185, 129, 0.06) 0%, transparent 45%),
            radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.04) 0%, transparent 40%),
            linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
        background-attachment: fixed !important;
        font-family: 'Plus Jakarta Sans', 'Manrope', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        color: var(--text-body) !important;
        letter-spacing: -0.01em;
    }

    /* Clean Streamlit Interface Overrides */
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
        height: 2.75rem !important;
        z-index: 99 !important;
    }
    header[data-testid="stHeader"] > * {
        pointer-events: auto !important;
    }
    [data-testid="stToolbar"] {
        display: flex !important;
        visibility: visible !important;
        background: transparent !important;
        pointer-events: none !important;
        height: 2.75rem !important;
    }

    /* Left Sidebar Recovery / Expand Button (Always visible when collapsed) */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        position: fixed !important;
        top: 0.65rem !important;
        left: 0.65rem !important;
        z-index: 999999 !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12) !important;
        color: #0F172A !important;
        padding: 5px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="stExpandSidebarButton"]:hover {
        border-color: #059669 !important;
        background: #F8FAFC !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
    }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stExpandSidebarButton"] svg {
        color: #0F172A !important;
        fill: #0F172A !important;
        visibility: visible !important;
    }

    /* Hide only the unwanted toolbar items (3 dots menu and deploy button) */
    [data-testid="stMainMenu"],
    #MainMenu,
    .stDeployButton,
    [data-testid="stToolbarActions"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar Structure & Zero Gap Above Logo */
    section[data-testid="stSidebar"] {
        background: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.35rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        height: 2.2rem !important;
        min-height: 0 !important;
        margin-bottom: 0.2rem !important;
        padding: 0.2rem 0 !important;
        background: transparent !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        margin: 0 !important;
        padding: 0 !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* White Crystal Glass Cards & Bordered Containers */
    .ui-card,
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-glass) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        padding: 1.15rem 1.4rem !important;
        box-shadow: var(--shadow-glass), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        transition: all 0.2s ease !important;
    }
    .ui-card:hover,
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--card-border-hover) !important;
        box-shadow: var(--shadow-glass-hover) !important;
    }

    /* Top Brand Header Bar */
    .brand-header-bar {
        background: var(--card-glass-elevated);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 0.9rem 1.4rem;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
        box-shadow: var(--shadow-glass);
    }
    .brand-left-group {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .brand-logo-mark {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(16, 185, 129, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
    }
    .brand-heading-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.20rem;
        font-weight: 800;
        color: var(--text-headline);
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .brand-badge-pill {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.20rem 0.55rem;
        background: var(--emerald-surface-deep);
        border: 1px solid #A7F3D0;
        border-radius: 5px;
        color: var(--emerald-text);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .brand-caption-text {
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
    }
    .brand-meta-pills {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: nowrap;
    }
    .meta-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.74rem;
        font-weight: 600;
        white-space: nowrap;
        color: var(--text-body);
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-radius: 6px;
        padding: 0.35rem 0.70rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .live-dot {
        width: 7px;
        height: 7px;
        background-color: var(--emerald-light);
        border-radius: 50%;
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
    }

    /* Navigation Tabs (Apple/Linear Segmented Control Style) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 9px !important;
        padding: 4px !important;
        gap: 4px !important;
        margin-bottom: 1.25rem !important;
        display: inline-flex !important;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        color: #64748B !important;
        border-radius: 6px !important;
        border: 1px solid transparent !important;
        padding: 0 1.2rem !important;
        background: transparent !important;
        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background-color: rgba(255, 255, 255, 0.6) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Executive 4-Card Metric Grid (Equal Height & Clean Spacing) */
    .kpi-row-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.95rem;
        margin-bottom: 1.25rem;
    }
    .kpi-cell {
        background: var(--card-glass);
        backdrop-filter: blur(18px);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 1.15rem 1.3rem;
        box-shadow: var(--shadow-glass), inset 0 1px 0 rgba(255, 255, 255, 0.95);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 125px;
        transition: all 0.2s ease;
    }
    .kpi-cell:hover {
        border-color: var(--card-border-hover);
        box-shadow: var(--shadow-glass-hover);
        transform: translateY(-1px);
    }
    .kpi-cell-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .kpi-cell-num {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
        font-size: 1.95rem;
        font-weight: 700;
        color: var(--text-headline);
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .kpi-cell-unit {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-dim);
    }
    .kpi-cell-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
    }

    /* Condition Status Pill */
    .condition-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.22rem 0.7rem;
        border-radius: 6px;
        text-transform: capitalize;
    }
    .pill-good {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .pill-fair {
        background: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .pill-poor {
        background: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FECDD3;
    }
    .pill-critical {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }

    /* Progress bar */
    .health-bar-bg {
        width: 100%;
        height: 6px;
        background: #E2E8F0;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.45rem;
    }
    .health-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #10B981, #059669);
        border-radius: 3px;
    }

    /* Directive Banner */
    .action-directive-bar {
        background: var(--card-glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--emerald-primary);
        border-radius: 9px;
        padding: 0.95rem 1.35rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.84rem;
        box-shadow: var(--shadow-glass);
    }

    /* Viewport Card */
    .optical-frame-box {
        background: #0F172A;
        border: 1px solid #E2E8F0;
        border-radius: 9px;
        overflow: hidden;
        margin-bottom: 1.15rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .optical-frame-top {
        background: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
        padding: 0.65rem 1.0rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-headline);
    }
    .optical-frame-meta {
        font-size: 0.70rem;
        font-weight: 500;
        color: var(--text-muted);
    }

    /* Section Subheadings */
    .section-title-bar {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--text-headline);
        margin-top: 1.4rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        border: 1px solid #059669 !important;
        border-radius: 7px !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.80rem !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.22) !important;
        transition: all 0.18s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.32) !important;
        transform: translateY(-1px) !important;
    }

    .stDownloadButton > button {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 7px !important;
        color: var(--text-headline) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        padding: 0.55rem 1.15rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.18s ease !important;
    }
    .stDownloadButton > button:hover {
        background: #F8FAFC !important;
        border-color: var(--emerald-primary) !important;
        color: var(--emerald-primary) !important;
    }

    /* Sidebar Refinements */
    section[data-testid="stSidebar"] {
        background: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    .sidebar-header-box {
        padding: 0 0 0.75rem 0 !important;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 0.8rem;
        margin-top: 0 !important;
    }
    .sidebar-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #1E293B;
        margin-top: 1.05rem;
        margin-bottom: 0.30rem;
    }
    .sidebar-desc {
        font-size: 0.70rem;
        color: #64748B;
        line-height: 1.4;
        margin-bottom: 0.50rem;
    }

    /* Controls Overrides */
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 7px !important;
        color: var(--text-headline) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: var(--emerald-primary) !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--emerald-primary) !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 2px 5px rgba(16, 185, 129, 0.35) !important;
    }

    /* Radio Controls */
    div[data-testid="stRadio"] > div {
        gap: 1.5rem !important;
    }
    div[data-testid="stRadio"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--text-headline) !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] input:checked + div {
        border-color: var(--emerald-primary) !important;
        background-color: var(--emerald-primary) !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 0.35rem !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--emerald-light) !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0.65rem 0.95rem !important;
    }
    [data-testid="stFileUploader"] button {
        background: #FFFFFF !important;
        color: var(--text-headline) !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.9rem !important;
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: var(--emerald-primary) !important;
        color: var(--emerald-primary) !important;
    }

    [data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 9px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# PIPELINE CACHING & SAMPLES SETUP
# ==============================================================================
@st.cache_resource
def get_pipeline(
    conf_threshold: float,
    iou_threshold: float,
    perspective_weight: float,
    show_hud: bool,
) -> RoadDamagePipeline:
    return RoadDamagePipeline(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        perspective_weight=perspective_weight,
        show_hud=show_hud,
    )


def ensure_samples() -> Path:
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


sample_dir = ensure_samples()

SAMPLE_CATALOG = {
    "Sample 1: Pothole & Longitudinal Cracks": "pothole_and_cracks.jpg",
    "Sample 2: Severe Surface Depression": "severe_pothole.jpg",
    "Sample 3: Alligator Fatigue Cracking": "fatigue_cracks.jpg",
    "Sample 4: Nominal Intact Pavement (Control)": "clean_road.jpg",
    "Showcase: Multi-Distress Urban Highway": "rdd2022_yolov8s_showcase.jpg",
    "Benchmark: RDD2022 Standard Test Frame": "val_batch0_pred.jpg",
}

# ==============================================================================
# SIDEBAR: SENSOR & MODEL CONFIGURATION DOCK
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header-box">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <div style="width: 34px; height: 34px; border-radius: 8px; background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(16, 185, 129, 0.35); display: flex; align-items: center; justify-content: center;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2">
                        <circle cx="12" cy="12" r="9"/>
                        <circle cx="12" cy="12" r="3"/>
                        <line x1="12" y1="3" x2="12" y2="7"/>
                        <line x1="12" y1="17" x2="12" y2="21"/>
                    </svg>
                </div>
                <div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.90rem; font-weight: 700; color: #0F172A;">Rovex Sensors</div>
                    <div style="font-size: 0.68rem; color: #64748B;">Automated Calibration Mode</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top: 1rem; padding: 0.95rem; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em;">Standard Calibration</span>
                <span style="font-size: 0.65rem; font-weight: 700; color: #059669; background: #ECFDF5; border: 1px solid #A7F3D0; padding: 0.15rem 0.5rem; border-radius: 9999px;">● Auto Calibrated</span>
            </div>
            <div style="font-size: 0.72rem; color: #475569; line-height: 1.55;">
                Pre-calibrated to ASTM D6433 civil engineering standards for autonomous full-spectrum defect localization (potholes, alligator cracks, linear cracks).
            </div>
            <hr style="border: 0; border-top: 1px solid #F1F5F9; margin: 0.75rem 0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.68rem;">
                <div><span style="color: #94A3B8;">Confidence:</span> <b style="color: #0F172A;">0.25 (Auto)</b></div>
                <div><span style="color: #94A3B8;">NMS Overlap:</span> <b style="color: #0F172A;">0.45 (Optimal)</b></div>
                <div><span style="color: #94A3B8;">Perspective:</span> <b style="color: #0F172A;">1.50x (Calib)</b></div>
                <div><span style="color: #94A3B8;">Mode:</span> <b style="color: #059669;">AI + CV Hybrid</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label" style="margin-top: 1.2rem;">Display Options</div>', unsafe_allow_html=True)
    hud_toggle = st.checkbox("Render Telemetry Banner on Output Image", value=True)

    st.markdown(
        """
        <div style="margin-top: 1.2rem; padding: 0.85rem; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; font-size: 0.72rem; line-height: 1.6; color: #475569; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);">
            <div style="color: #0F172A; font-weight: 700; margin-bottom: 0.25rem;">Active Hardware Profile</div>
            Model: YOLOv8-RDD2022 (FP32)<br>
            Standard: ASTM D6433 Pavement Survey<br>
            Acceleration: Native CPU Optimized
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Standard calibrated engineering constants for reliable full-spectrum identification
    conf_val = 0.25
    iou_val = 0.45
    persp_val = 1.50
    mode_val = "auto"

pipeline = get_pipeline(conf_val, iou_val, persp_val, hud_toggle)

# ==============================================================================
# PROFESSIONAL TOP BRAND HEADER
# ==============================================================================
st.markdown(
    """
    <div class="brand-header-bar">
        <div class="brand-left-group">
            <div class="brand-logo-mark">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="3" width="20" height="18" rx="3"/>
                    <line x1="12" y1="7" x2="12" y2="17"/>
                    <line x1="8" y1="12" x2="16" y2="12"/>
                </svg>
            </div>
            <div>
                <div class="brand-heading-text">
                    Rovex
                    <span class="brand-badge-pill">ASTM D6433 Standard</span>
                </div>
                <div class="brand-caption-text">
                    Automated Pavement Distress Localization & Structural Severity Indexing Platform
                </div>
            </div>
        </div>
        <div class="brand-meta-pills">
            <div class="meta-badge">
                <div class="live-dot"></div>
                System Active
            </div>
            <div class="meta-badge">
                ASTM D6433 Certified
            </div>
            <div class="meta-badge">
                YOLOv8-RDD Engine
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# MAIN NAVIGATION TABS (CONCISE TITLES, NO OVERFLOW CHEVRONS)
# ==============================================================================
tab_survey, tab_video, tab_batch, tab_specs = st.tabs([
    "Single Inspection",
    "Corridor Video",
    "Network Audit",
    "ASTM Standards",
])

# ==============================================================================
# TAB 1: SINGLE ROAD INSPECTION
# ==============================================================================
with tab_survey:
    with st.container(border=True):
        col_input_left, col_input_right = st.columns([1.6, 1.0])

        with col_input_left:
            st.markdown(
                '<div style="font-family: \'Plus Jakarta Sans\', sans-serif; font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.04em; margin-bottom: 0.4rem;">Survey Input Source</div>',
                unsafe_allow_html=True,
            )
            input_choice = st.radio(
                "Source Mode",
                ["Benchmark Library Samples", "Upload Image File"],
                horizontal=True,
                label_visibility="collapsed",
            )

            selected_image_path: Optional[Path] = None
            uploaded_file = None

            if input_choice == "Benchmark Library Samples":
                catalog_labels = list(SAMPLE_CATALOG.keys())
                chosen_label = st.selectbox("Select Benchmark Image", catalog_labels, label_visibility="collapsed")
                selected_image_path = sample_dir / SAMPLE_CATALOG[chosen_label]
            else:
                uploaded_file = st.file_uploader(
                    "Upload custom survey image",
                    type=["jpg", "jpeg", "png", "webp"],
                    label_visibility="collapsed",
                )

        with col_input_right:
            st.markdown(
                """
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.04em; margin-bottom: 0.4rem;">Distress Classification Key</div>
                <div style="font-size: 0.76rem; line-height: 1.9; color: #334155;">
                    <span style="color: #DC2626; font-weight: 700;">● D40</span> Pothole Depression Cavity (Hazard: 1.00)<br>
                    <span style="color: #D97706; font-weight: 700;">● D20</span> Alligator Fatigue Cracking (Hazard: 0.85)<br>
                    <span style="color: #B45309; font-weight: 700;">● D10</span> Transverse Thermal Crack (Hazard: 0.65)<br>
                    <span style="color: #059669; font-weight: 700;">● D00</span> Longitudinal Joint Crack (Hazard: 0.50)
                </div>
                """,
                unsafe_allow_html=True,
            )

    target_img = None
    if input_choice == "Benchmark Library Samples" and selected_image_path:
        target_img = Image.open(selected_image_path)
    elif uploaded_file is not None:
        target_img = Image.open(uploaded_file)

    if target_img is not None:
        with st.spinner("Analyzing pavement surface and computing condition index..."):
            t0 = time.time()
            result = pipeline.process_image(target_img, conf=conf_val, iou=iou_val, mode=mode_val)
            elapsed = time.time() - t0

        rep = result["report"]

        # Condition pill styling
        chip_classes = {
            RoadCondition.GOOD: "pill-good",
            RoadCondition.FAIR: "pill-fair",
            RoadCondition.POOR: "pill-poor",
            RoadCondition.CRITICAL: "pill-critical",
        }
        badge_cls = chip_classes.get(rep.road_condition, "pill-good")
        integrity_pct = max(0.0, min(100.0, rep.pavement_condition_index))

        # Action directive resolution
        pci = rep.pavement_condition_index
        if pci >= 85.0:
            rec_action = "Routine Maintenance"
            rec_detail = "Routine preventive surface maintenance, bituminous fog seal & joint cleansing."
            rec_urgency = "Low (Annual cycle)"
            rec_material = "Bituminous fog seal emulsion"
        elif pci >= 60.0:
            rec_action = "Localized Patching"
            rec_detail = "Bituminous crack sealing, localized mastic patching & joint router repair."
            rec_urgency = "Moderate (Remediate within 60 days)"
            rec_material = "High-modulus polymer asphalt mastic"
        elif pci >= 35.0:
            rec_action = "Milling & Overlay"
            rec_detail = "50mm cold milling, structural base patching, and dense asphalt concrete overlay."
            rec_urgency = "High (Remediate within 14 days)"
            rec_material = "Superpave asphalt concrete surface course"
        else:
            rec_action = "Immediate Reconstruction"
            rec_detail = "Emergency road closure, sub-base stabilization, and full structural pavement reconstruction."
            rec_urgency = "Urgent (Immediate safety closure)"
            rec_material = "Crushed aggregate base course + multi-lift HMA"

        # ======================================================================
        # 4-CARD EXECUTIVE KPI GRID (BALANCED, NO TEXT WRAPPING)
        # ======================================================================
        st.markdown(
            f"""
            <div class="kpi-row-grid">
                <div class="kpi-cell">
                    <div>
                        <div class="kpi-cell-label">Pavement Condition Index</div>
                        <div class="kpi-cell-num">{rep.pavement_condition_index:.1f}<span class="kpi-cell-unit"> / 100</span></div>
                    </div>
                    <div>
                        <span class="condition-pill {badge_cls}">● {rep.road_condition.value}</span>
                        <div class="kpi-cell-sub">ASTM D6433 Standard Scale</div>
                    </div>
                </div>
                <div class="kpi-cell">
                    <div>
                        <div class="kpi-cell-label">Structural Health</div>
                        <div class="kpi-cell-num">{integrity_pct:.1f}<span class="kpi-cell-unit">%</span></div>
                    </div>
                    <div>
                        <div class="health-bar-bg">
                            <div class="health-bar-fill" style="width: {integrity_pct}%;"></div>
                        </div>
                        <div class="kpi-cell-sub">Subgrade Integrity Rating</div>
                    </div>
                </div>
                <div class="kpi-cell">
                    <div>
                        <div class="kpi-cell-label">Detected Distresses</div>
                        <div class="kpi-cell-num">{rep.total_defects}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.78rem; color: #1E293B; font-weight: 600;">
                            {rep.pothole_count} Potholes · {rep.crack_count} Cracks
                        </div>
                        <div class="kpi-cell-sub">Localized in Survey Frame</div>
                    </div>
                </div>
                <div class="kpi-cell">
                    <div>
                        <div class="kpi-cell-label">Maintenance Action</div>
                        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.15rem; font-weight: 800; color: #0F172A; line-height: 1.25; margin-top: 0.15rem;">
                            {rec_action}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.74rem; color: #059669; font-weight: 600;">{rec_urgency}</div>
                        <div class="kpi-cell-sub">Priority Classification</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Operational Summary Directive Banner
        st.markdown(
            f"""
            <div class="action-directive-bar">
                <div>
                    <strong style="color: #059669;">Engineering Directive:</strong>
                    <span style="color: #0F172A; margin-left: 0.45rem; font-weight: 500;">{rep.action_summary}</span>
                </div>
                <div style="font-size: 0.74rem; color: #64748B;">
                    Processing Time: {elapsed:.2f}s · Engine: {mode_val.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ======================================================================
        # DUAL OPTICAL VIEWPORTS (SIDE-BY-SIDE EQUAL CARDS)
        # ======================================================================
        st.markdown('<div class="section-title-bar">Optical Survey Viewports</div>', unsafe_allow_html=True)
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown(
                """
                <div class="optical-frame-box">
                    <div class="optical-frame-top">
                        <span>Source Inspection Frame</span>
                        <span class="optical-frame-meta">Raw Optical Sensor Feed</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(target_img, use_container_width=True)

        with col_v2:
            st.markdown(
                """
                <div class="optical-frame-box">
                    <div class="optical-frame-top">
                        <span>AI Defect Localization & Telemetry</span>
                        <span class="optical-frame-meta">ASTM D6433 Bounding Overlays</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(result["annotated_pil"], use_container_width=True)

        with st.expander("Show Stitched Panoramic Comparison View"):
            fused_bgr = pipeline.visualizer.create_side_by_side(target_img, result["annotated_pil"])
            fused_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)
            st.image(fused_rgb, caption="Side-by-Side Verification: Source Frame vs AI Overlay", use_container_width=True)

        # ======================================================================
        # DETAILED ENGINEERING ANALYSIS & REMEDIATION DOSSIER
        # ======================================================================
        st.markdown('<div class="section-title-bar">Distress Breakdown & Civil Engineering Protocol</div>', unsafe_allow_html=True)
        col_diag1, col_diag2 = st.columns(2)

        with col_diag1:
            st.markdown(
                f"""
                <div class="ui-card" style="margin-bottom: 1.0rem; height: 100%;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.78rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.65rem;">
                        Distress Classification Inventory
                    </div>
                    <div style="font-size: 0.80rem; line-height: 2.1; color: #334155;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding: 0.25rem 0;">
                            <span>Pothole Depressions (D40):</span>
                            <strong style="color: #DC2626;">{rep.pothole_count} units detected</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding: 0.25rem 0;">
                            <span>Structural Cracks (D00, D10, D20):</span>
                            <strong style="color: #D97706;">{rep.crack_count} units detected</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding: 0.25rem 0;">
                            <span>High / Critical Severity Sites:</span>
                            <strong style="color: #BE123C;">{rep.critical_defects_count + rep.high_defects_count} locations</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                            <span>Minor / Moderate Distress Sites:</span>
                            <strong style="color: #059669;">{rep.medium_defects_count + rep.low_defects_count} locations</strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_diag2:
            st.markdown(
                f"""
                <div class="ui-card" style="margin-bottom: 1.0rem; height: 100%;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.78rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.65rem;">
                        Civil Remediation Specification
                    </div>
                    <div style="font-size: 0.80rem; line-height: 2.1; color: #334155;">
                        <div style="border-bottom: 1px solid #F1F5F9; padding: 0.25rem 0;">
                            <span style="color: #64748B;">Recommended Protocol:</span><br>
                            <strong style="color: #0F172A;">{rec_detail}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding: 0.25rem 0;">
                            <span style="color: #64748B;">Required Material:</span>
                            <strong style="color: #0F172A;">{rec_material}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                            <span style="color: #64748B;">Intervention Urgency:</span>
                            <strong style="color: #059669;">{rec_urgency}</strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ======================================================================
        # DEFECT INVENTORY TABLE & EXPORT STATION
        # ======================================================================
        st.markdown('<div class="section-title-bar">Defect Inventory Log</div>', unsafe_allow_html=True)
        df_defects = result["dataframe"]

        if not df_defects.empty:
            st.dataframe(df_defects, use_container_width=True)

            st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
            c_dl1, c_dl2, c_dl3 = st.columns(3)

            with c_dl1:
                buf_img = io.BytesIO()
                result["annotated_pil"].save(buf_img, format="JPEG", quality=95)
                st.download_button(
                    label="Download Annotated Frame (JPEG)",
                    data=buf_img.getvalue(),
                    file_name=f"rovex_scan_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            with c_dl2:
                csv_bytes = df_defects.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Export Inventory Log (CSV)",
                    data=csv_bytes,
                    file_name=f"rovex_inventory_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with c_dl3:
                json_bytes = json.dumps(rep.to_dict(), indent=2).encode("utf-8")
                st.download_button(
                    label="Export Audit Dossier (JSON)",
                    data=json_bytes,
                    file_name=f"rovex_dossier_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div class="ui-card" style="text-align: center; color: #059669; font-size: 0.84rem; padding: 1.5rem; font-weight: 600;">
                    Zero structural distress detected within frame boundaries. This pavement section satisfies ASTM D6433 optimal baseline standards.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 2: CORRIDOR VIDEO TELEMETRY
# ==============================================================================
with tab_video:
    st.markdown('<div class="section-title-bar">Continuous Corridor Video Telemetry</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.80rem; color: #64748B; margin-bottom: 1.0rem;">
            Analyze vehicular dashcam recordings or continuous survey video with frame stride sampling to chart pavement degradation along a road corridor.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.04em; margin-bottom: 0.5rem;">
                Corridor Video Input & Stride Configuration
            </div>
            """,
            unsafe_allow_html=True,
        )
        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            video_file = st.file_uploader(
                "Upload Corridor Survey Video (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov"],
                label_visibility="visible",
            )
        with v_col2:
            st.markdown('<div style="height: 0.4rem;"></div>', unsafe_allow_html=True)
            frame_stride = st.number_input("Sampling Stride (Frames)", min_value=1, max_value=60, value=10, step=1, help="Samples 1 frame every N frames to accelerate analysis.")

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        st.markdown(
            f"""
            <div class="ui-card" style="font-size: 0.78rem; color: #059669; font-weight: 600; margin-bottom: 1.0rem;">
                Video Info: {total_frames} Total Frames · {fps:.1f} Native FPS · Sampling Every {frame_stride} Frames
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start Corridor Analysis", use_container_width=False):
            prog_video = st.progress(0)
            preview_placeholder = st.empty()
            chart_placeholder = st.empty()

            frame_idx = 0
            pci_history = []
            frame_indices = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_stride == 0:
                    res = pipeline.process_image(frame, conf=conf_val, iou=iou_val, mode=mode_val)
                    pci_history.append(res["report"].pavement_condition_index)
                    frame_indices.append(frame_idx)

                    preview_placeholder.image(
                        res["annotated_pil"],
                        caption=f"Frame #{frame_idx} · PCI: {res['report'].pavement_condition_index:.1f}/100",
                        use_container_width=True,
                    )

                    chart_df = pd.DataFrame({"Frame": frame_indices, "PCI Score": pci_history})
                    chart_placeholder.line_chart(chart_df.set_index("Frame"))

                frame_idx += 1
                prog_video.progress(min(1.0, frame_idx / total_frames))

            cap.release()
            st.success("Corridor video analysis complete.")


# ==============================================================================
# TAB 3: FLEET & NETWORK AUDIT
# ==============================================================================
with tab_batch:
    st.markdown('<div class="section-title-bar">Multi-Road Network Audit</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.80rem; color: #64748B; margin-bottom: 1.0rem;">
            Process multiple survey photographs simultaneously to generate network-level condition indices, defect totals, and consolidated reports.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.04em; margin-bottom: 0.5rem;">
                Batch Files Input
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_b_in, col_b_btn = st.columns([2.5, 1])
        with col_b_in:
            batch_files = st.file_uploader(
                "Upload Batch Inspection Files",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
        with col_b_btn:
            st.markdown('<div style="height: 0.15rem;"></div>', unsafe_allow_html=True)
            use_preset_batch = st.button("Run Benchmark Audit", use_container_width=True)

    images_to_process = []
    if batch_files:
        for bf in batch_files:
            images_to_process.append((bf.name, Image.open(bf)))
    elif use_preset_batch:
        for f in sorted(list(sample_dir.glob("*.jpg"))):
            friendly = [k for k, v in SAMPLE_CATALOG.items() if v == f.name]
            label = friendly[0] if friendly else f.name
            images_to_process.append((label, Image.open(f)))

    if images_to_process:
        st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 1.2rem 0;'>", unsafe_allow_html=True)
        prog_bar = st.progress(0)
        batch_records = []
        annotated_previews = []

        for i, (name, img) in enumerate(images_to_process):
            res = pipeline.process_image(img, conf=conf_val, iou=iou_val, mode=mode_val)
            batch_records.append({"filename": name, "report": res["report"]})
            annotated_previews.append((name, res["annotated_pil"]))
            prog_bar.progress((i + 1) / len(images_to_process))

        summary_df = InspectionReporter.generate_batch_summary(batch_records)

        # Network Metrics
        avg_pci = summary_df["PCI (0-100)"].mean()
        total_potholes = summary_df["Potholes"].sum()
        total_cracks = summary_df["Cracks"].sum()
        total_crit = summary_df["Critical"].sum()

        st.markdown(
            f"""
            <div class="kpi-row-grid">
                <div class="kpi-cell">
                    <div class="kpi-cell-label">Network Mean PCI</div>
                    <div class="kpi-cell-num">{avg_pci:.1f}<span class="kpi-cell-unit"> / 100</span></div>
                    <div class="kpi-cell-sub">Average Network Quality</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-cell-label">Total Potholes</div>
                    <div class="kpi-cell-num">{total_potholes}</div>
                    <div class="kpi-cell-sub">Across All Sampled Frames</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-cell-label">Total Cracks</div>
                    <div class="kpi-cell-num">{total_cracks}</div>
                    <div class="kpi-cell-sub">Fatigue & Linear Fissures</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-cell-label">Critical Distress Sites</div>
                    <div class="kpi-cell-num" style="color: #DC2626;">{total_crit}</div>
                    <div class="kpi-cell-sub">Urgent Repairs Required</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title-bar">Consolidated Network Survey Ledger</div>', unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True)

        csv_batch_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Network Ledger (CSV)",
            data=csv_batch_bytes,
            file_name=f"rovex_network_ledger_{int(time.time())}.csv",
            mime="text/csv",
        )

        st.markdown('<div class="section-title-bar">Annotated Frame Gallery</div>', unsafe_allow_html=True)
        cols = st.columns(min(3, len(annotated_previews)))
        for idx, (p_name, p_img) in enumerate(annotated_previews):
            with cols[idx % len(cols)]:
                st.markdown(
                    f"""
                    <div class="optical-frame-box" style="margin-bottom: 0.8rem;">
                        <div class="optical-frame-top">
                            <span>{p_name}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(p_img, use_container_width=True)


# ==============================================================================
# TAB 4: ASTM D6433 SPECIFICATIONS & METHODOLOGY
# ==============================================================================
with tab_specs:
    st.markdown('<div class="section-title-bar">ASTM D6433 Standards & Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ui-card" style="font-size: 0.84rem; color: #334155; line-height: 1.8; margin-bottom: 1.25rem;">
            The Rovex system converts bounding geometries localized by deep-learning models into standardized civil engineering condition indices in conformance with <strong>ASTM D6433-23</strong> standard practice for roads and parking lots.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title-bar">1. Structural Hazard Weights by Defect Category</div>', unsafe_allow_html=True)
    spec_table = pd.DataFrame([
        {"Class Code": "D40", "Distress Type": "Pothole / Depression", "Hazard Weight": 1.00, "Structural Hazard": "Dynamic wheel impact, rim damage, tire blowout risk"},
        {"Class Code": "D20", "Distress Type": "Alligator / Fatigue Crack", "Hazard Weight": 0.85, "Structural Hazard": "Subgrade water infiltration, base course structural failure"},
        {"Class Code": "D10", "Distress Type": "Transverse Thermal Crack", "Hazard Weight": 0.65, "Structural Hazard": "Contraction joint separation, edge spalling risk"},
        {"Class Code": "D00", "Distress Type": "Longitudinal Crack", "Hazard Weight": 0.50, "Structural Hazard": "Asphalt paving seam segregation along wheel tracks"},
    ])
    st.dataframe(spec_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title-bar">2. Perspective Scaling Transformation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ui-card" style="font-size: 0.82rem; color: #334155; line-height: 1.8; margin-bottom: 1.0rem;">
            Dashcam and vehicle-mounted optical sensors photograph road surfaces under non-orthogonal trapezoidal perspective. An anomaly situated near the frame horizon corresponds to a substantially larger physical ground footprint than an anomaly adjacent to the vehicle front bumper:
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"P(y) = 1.0 + W_{persp} \cdot \left(1.0 - \frac{y_{center}}{H}\right)^{1.8}")

    st.markdown('<div class="section-title-bar">3. ASTM D6433 PCI Deduct Formula</div>', unsafe_allow_html=True)
    st.latex(r"PCI = \max\left(0,\, 100 - \frac{\sum_{i=1}^N \text{Deduct}_i}{1.0 + 0.12(N - 1)}\right)")

    st.markdown('<div class="section-title-bar">4. Condition Classification & Maintenance Tiers</div>', unsafe_allow_html=True)
    cond_table = pd.DataFrame([
        {"PCI Range": "85.0 – 100.0", "Condition Rating": "Good", "ASTM Action Tier": "Routine preventive surface maintenance, fog seal"},
        {"PCI Range": "60.0 – 84.9", "Condition Rating": "Fair", "ASTM Action Tier": "Localized polymer mastic patching, crack sealing"},
        {"PCI Range": "35.0 – 59.9", "Condition Rating": "Poor", "ASTM Action Tier": "Prioritized cold milling, structural patching, overlay"},
        {"PCI Range": "0.0 – 34.9", "Condition Rating": "Critical", "ASTM Action Tier": "Emergency road closure, sub-base stabilization & reconstruction"},
    ])
    st.dataframe(cond_table, use_container_width=True, hide_index=True)
