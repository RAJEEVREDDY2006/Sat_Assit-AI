"""
Annotated Image Export Router
Author: Sat_Assist Team

POST /api/annotate  → Draws bounding boxes on the session image, saves PNG, returns URL.
GET  /api/annotate/{filename} → Download the annotated PNG directly.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path

from backend.storage.session_store import session_store
from backend.model.image_annotator import ImageAnnotator, ANNOTATED_DIR

router = APIRouter(prefix="/api", tags=["Annotate"])


class AnnotateRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    bounding_boxes: List[Dict[str, Any]] = Field(..., description="List of bounding box dicts")
    label: str = Field("Detected Objects", description="Overall label suffix for filename")


@router.post("/annotate")
async def annotate_image(payload: AnnotateRequest):
    """
    Draws bounding boxes on the session's satellite image server-side using PIL.
    Returns a URL to the saved annotated PNG for display and download.
    """
    session = session_store.get_session(payload.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{payload.session_id}' not found. Upload an image first."
        )

    if not payload.bounding_boxes:
        raise HTTPException(
            status_code=400,
            detail="No bounding boxes provided. Run a grounding query first."
        )

    image_path = session["image_path"]
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Original image file not found on disk.")

    try:
        output_filename = f"{payload.session_id}_annotated.png"
        saved_path = ImageAnnotator.annotate(
            image_path=image_path,
            bounding_boxes=payload.bounding_boxes,
            output_filename=output_filename,
        )
        annotated_url = f"/annotated/{output_filename}"
        return {
            "status": "success",
            "annotated_image_url": annotated_url,
            "download_url": f"/api/annotate/download/{output_filename}",
            "box_count": len(payload.bounding_boxes),
            "session_id": payload.session_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")


@router.get("/annotate/download/{filename}")
async def download_annotated_image(filename: str):
    """Downloads a previously generated annotated PNG."""
    # Safety: allow only safe filenames (no path traversal)
    safe_name = Path(filename).name
    file_path = ANNOTATED_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Annotated file '{safe_name}' not found.")
    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=safe_name,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'}
    )

