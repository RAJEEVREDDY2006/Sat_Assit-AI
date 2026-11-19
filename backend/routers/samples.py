"""
Sample Satellite Images Route Handler
Author: Sat_Assist Team

Provides curated sample satellite images from the downloaded VRSBench dataset.
"""

import json
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.storage.session_store import session_store
from backend.model.image_processor import SatelliteImageProcessor

router = APIRouter(prefix="/api", tags=["Samples"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VAL_JSON_PATH = BASE_DIR / "data" / "vrsbench_5k" / "vrsbench_5k_val.json"
TRAIN_JSON_PATH = BASE_DIR / "data" / "vrsbench_5k" / "vrsbench_5k_train.json"

@router.get("/samples")
async def get_sample_images(count: int = 12):
    """
    Returns a list of curated satellite images from the local VRSBench dataset
    with pre-associated questions and captions for instant 1-click testing.
    """
    data_path = VAL_JSON_PATH if VAL_JSON_PATH.exists() else TRAIN_JSON_PATH
    if not data_path.exists():
        return {"status": "ok", "samples": []}

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        curated = []
        for s in all_samples:
            img_file = s.get("image_file")
            img_path = Path(s.get("image_path", ""))
            
            # Verify image file exists on disk
            if not img_path.exists():
                alt_path = BASE_DIR / "data" / "vrsbench_5k" / "images" / img_file
                if alt_path.exists():
                    img_path = alt_path
                else:
                    continue

            # Extract suggested prompt questions
            suggested_questions = []
            if s.get("vqa_pairs"):
                suggested_questions = [q["question"] for q in s["vqa_pairs"][:4]]
            
            if not suggested_questions:
                suggested_questions = [
                    "Describe this satellite image.",
                    "What are the main structures visible in this area?",
                    "Are there any aircraft or vehicles present?",
                    "Where are the key facilities located?"
                ]

            curated.append({
                "id": s.get("image_id", img_file),
                "image_file": img_file,
                "image_url": f"/dataset_images/{img_file}",
                "caption": s.get("caption", ""),
                "suggested_questions": suggested_questions,
                "vqa_count": len(s.get("vqa_pairs", []))
            })

            if len(curated) >= count:
                break

        return {"status": "ok", "samples": curated, "total_available": len(all_samples)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset samples: {str(e)}")

@router.post("/samples/select/{image_file}")
async def select_sample_image(image_file: str):
    """
    Selects a sample dataset image and initializes a new chat session for it.
    """
    img_path = BASE_DIR / "data" / "vrsbench_5k" / "images" / image_file
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample image '{image_file}' not found.")

    try:
        img = SatelliteImageProcessor.load_image(img_path)
        meta = SatelliteImageProcessor.get_image_metadata(img)

        session_id = f"sample_{uuid.uuid4().hex[:10]}"
        session_data = session_store.create_session(
            session_id=session_id,
            image_path=str(img_path),
            image_filename=image_file,
            image_meta=meta,
            is_sample=True
        )

        return {
            "status": "success",
            "session_id": session_id,
            "filename": image_file,
            "image_url": f"/dataset_images/{image_file}",
            "metadata": meta
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize sample session: {str(e)}")
