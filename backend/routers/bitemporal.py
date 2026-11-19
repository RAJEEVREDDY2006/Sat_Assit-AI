"""
Bi-Temporal Satellite Change Detection Router
Author: Sat_Assist Team

Provides endpoints for uploading:
1. Single composite Before/After image (auto-split into T1 and T2).
2. Dual temporal images (T1: Before, T2: After).
Runs co-registered structural change heatmaps and automated VLM change reports.
"""

import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from backend.model.change_detector import BiTemporalChangeDetector
from backend.model.vlm_engine import vlm_engine

router = APIRouter(prefix="/api/bitemporal", tags=["Bi-Temporal Change Detection"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class ChangeAnalysisRequest(BaseModel):
    t1_filename: Optional[str] = None
    t2_filename: Optional[str] = None
    single_filename: Optional[str] = None
    include_ai_report: Optional[bool] = True


@router.post("/upload-single")
async def upload_single_bitemporal_image(
    file: UploadFile = File(..., description="Single composite Before/After satellite image")
):
    """
    Uploads a single composite satellite image containing both Before and After scenes
    (side-by-side or stacked). Automatically splits it into T1 and T2 and generates change heatmap.
    """
    img_id = uuid.uuid4().hex[:8]
    ext = Path(file.filename).suffix or ".jpg"
    saved_name = f"bitemp_single_{img_id}{ext}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)

        # Run automated split and detection
        heatmap_name = f"change_{img_id}_heatmap.png"
        cv_result = BiTemporalChangeDetector.detect_changes_from_single_image(
            single_image_path=str(saved_path),
            output_heatmap_name=heatmap_name
        )

        # Generate VLM Intelligence Brief with structured categories
        ai_report = ""
        change_categories = {"appeared": [], "disappeared": [], "changed": [], "unchanged": [], "summary": ""}
        try:
            vlm_res = vlm_engine.analyze_bitemporal_change(
                image_path_t1=cv_result["t1_path"],
                image_path_t2=cv_result["t2_path"],
                change_stats=cv_result
            )
            ai_report = vlm_res.get("report", "")
            change_categories = vlm_res.get("categories", change_categories)
        except Exception as e:
            print(f"[BiTemporal] VLM report warning: {e}")

        return {
            "status": "success",
            "mode": "single_composite",
            "single_filename": saved_name,
            "t1_url": cv_result["t1_url"],
            "t2_url": cv_result["t2_url"],
            "heatmap_url": f"/annotated/{heatmap_name}",
            "change_percentage": cv_result["change_percentage"],
            "structural_similarity": cv_result["structural_similarity"],
            "change_boxes": cv_result["change_boxes"],
            "summary_counts": cv_result.get("summary_counts", {}),
            "ai_report": ai_report,
            "change_categories": change_categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process single change image: {str(e)}")


@router.post("/upload")
async def upload_bitemporal_pair(
    file_t1: UploadFile = File(..., description="T1 (Before) satellite image"),
    file_t2: UploadFile = File(..., description="T2 (After) satellite image")
):
    """
    Uploads a pair of bi-temporal satellite images for comparative analysis.
    """
    pair_id = uuid.uuid4().hex[:8]

    ext_t1 = Path(file_t1.filename).suffix or ".jpg"
    ext_t2 = Path(file_t2.filename).suffix or ".jpg"

    t1_name = f"bitemp_{pair_id}_T1_before{ext_t1}"
    t2_name = f"bitemp_{pair_id}_T2_after{ext_t2}"

    t1_path = UPLOAD_DIR / t1_name
    t2_path = UPLOAD_DIR / t2_name

    try:
        content_t1 = await file_t1.read()
        with open(t1_path, "wb") as f:
            f.write(content_t1)

        content_t2 = await file_t2.read()
        with open(t2_path, "wb") as f:
            f.write(content_t2)

        return {
            "status": "success",
            "pair_id": pair_id,
            "t1": {
                "filename": t1_name,
                "url": f"/uploads/{t1_name}"
            },
            "t2": {
                "filename": t2_name,
                "url": f"/uploads/{t2_name}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image pair: {str(e)}")


@router.post("/analyze")
async def analyze_bitemporal_pair(payload: ChangeAnalysisRequest):
    """
    Runs computer vision alignment + SSIM change detection + VLM reasoning on T1 and T2 images.
    """
    t1_path = UPLOAD_DIR / payload.t1_filename
    t2_path = UPLOAD_DIR / payload.t2_filename

    if not t1_path.exists() or not t2_path.exists():
        raise HTTPException(status_code=404, detail="One or both temporal images were not found on server.")

    try:
        heatmap_name = f"change_{uuid.uuid4().hex[:8]}_heatmap.png"

        cv_result = BiTemporalChangeDetector.detect_changes(
            image_path_t1=str(t1_path),
            image_path_t2=str(t2_path),
            output_heatmap_name=heatmap_name
        )

        ai_report = ""
        change_categories = {"appeared": [], "disappeared": [], "changed": [], "unchanged": [], "summary": ""}
        if payload.include_ai_report:
            vlm_res = vlm_engine.analyze_bitemporal_change(
                image_path_t1=str(t1_path),
                image_path_t2=str(t2_path),
                change_stats=cv_result
            )
            ai_report = vlm_res.get("report", "")
            change_categories = vlm_res.get("categories", change_categories)

        return {
            "status": "success",
            "change_percentage": cv_result["change_percentage"],
            "structural_similarity": cv_result["structural_similarity"],
            "heatmap_url": f"/annotated/{heatmap_name}",
            "change_boxes": cv_result["change_boxes"],
            "summary_counts": cv_result.get("summary_counts", {}),
            "ai_report": ai_report,
            "change_categories": change_categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Change analysis failed: {str(e)}")
