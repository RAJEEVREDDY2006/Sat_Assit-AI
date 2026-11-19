"""
Satellite Image Upload Route Handler
Author: Sat_Assist Team
"""

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.model.image_processor import SatelliteImageProcessor
from backend.storage.session_store import session_store

router = APIRouter(prefix="/api", tags=["Upload"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

@router.post("/upload")
async def upload_satellite_image(file: UploadFile = File(...)):
    """
    Accepts satellite image uploads (JPG, PNG, TIFF, GeoTIFF).
    Processes image, creates thumbnail, and initializes conversation session.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension '{ext}'. Supported formats: JPG, PNG, WebP, TIFF."
        )

    session_id = f"sat_{uuid.uuid4().hex[:12]}"
    raw_filename = f"{session_id}_{Path(file.filename).name}"
    save_path = UPLOADS_DIR / raw_filename

    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        # Process and validate image with SatelliteImageProcessor
        img = SatelliteImageProcessor.load_image(save_path)
        meta = SatelliteImageProcessor.get_image_metadata(img)

        # Generate and save thumbnail
        thumb_filename = f"thumb_{session_id}.jpg"
        thumb_path = UPLOADS_DIR / thumb_filename
        thumb_bytes = SatelliteImageProcessor.generate_thumbnail_bytes(img)
        with open(thumb_path, "wb") as f:
            f.write(thumb_bytes)

        # Register session in store
        session_data = session_store.create_session(
            session_id=session_id,
            image_path=str(save_path),
            image_filename=file.filename,
            image_meta=meta,
            is_sample=False
        )

        return {
            "status": "success",
            "session_id": session_id,
            "filename": file.filename,
            "image_url": f"/uploads/{raw_filename}",
            "thumbnail_url": f"/uploads/{thumb_filename}",
            "metadata": meta,
            "message": "Satellite image processed successfully."
        }

    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process satellite image: {str(e)}")
