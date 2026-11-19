"""
Conversational Remote Sensing Chat & VQA Router
Author: Sat_Assist Team
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.model.vlm_engine import vlm_engine
from backend.storage.session_store import session_store

router = APIRouter(prefix="/api", tags=["Chat"])

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID for the active satellite image")
    message: str = Field(..., description="Natural language question or command")
    task: Optional[str] = Field("auto", description="Task type: 'auto', 'caption', 'vqa', or 'grounding'")

class ChatResponse(BaseModel):
    status: str
    session_id: str
    answer: str
    bounding_boxes: List[dict]
    task_performed: str
    device: str
    annotated_image_url: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_with_satellite_image(payload: ChatRequest):
    """
    Submits a natural language query for the active satellite image session.
    Invokes the VLM Inference Engine and returns natural answer + bounding boxes.
    """
    session = session_store.get_session(payload.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{payload.session_id}' not found. Please upload a satellite image first."
        )

    image_path = session["image_path"]
    user_query = payload.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    # Get conversation history for multi-turn context
    history = session_store.get_history(payload.session_id)

    try:
        # Determine conversational greetings vs image analysis tasks
        q_clean = user_query.lower().strip("!.,? ")
        
        GREETINGS = ["hi", "hello", "hey", "hola", "namaste", "greetings", "good morning", "good afternoon", "good evening", "howdy", "hey there", "hi there"]
        HELP_PROMPTS = ["help", "who are you", "what can you do", "what are you", "how does this work", "what is satassist"]
        THANKS_PROMPTS = ["thank you", "thanks", "thx", "thank you so much", "appreciate it"]

        if q_clean in GREETINGS or any(q_clean.startswith(g + " ") for g in ["hi", "hello", "hey"]):
            answer = "Hello! I am SatAssist AI, your remote sensing and satellite image analysis assistant. How can I help you analyze this image today?"
            bboxes = []
            task_type = "chat"
            result = {"answer": answer, "bounding_boxes": bboxes, "device": "system"}
        elif q_clean in HELP_PROMPTS or "what can you do" in q_clean:
            answer = "I am SatAssist AI, an expert vision-language model for overhead satellite and aerial imagery. You can ask me to identify roads, buildings, vehicles, aircraft, vegetation, or describe any geographic and structural details in this image."
            bboxes = []
            task_type = "chat"
            result = {"answer": answer, "bounding_boxes": bboxes, "device": "system"}
        elif q_clean in THANKS_PROMPTS:
            answer = "You're welcome! Feel free to ask any other questions about this satellite image."
            bboxes = []
            task_type = "chat"
            result = {"answer": answer, "bounding_boxes": bboxes, "device": "system"}
        elif payload.task == "caption" or (payload.task == "auto" and any(k in q_clean for k in ["describe", "caption", "overview", "what is this image"])):
            result = vlm_engine.generate_caption(image_path)
            task_type = "caption"
        elif payload.task == "grounding" or (payload.task == "auto" and any(k in q_clean for k in ["where is", "where are", "locate", "highlight", "find the", "find", "show me", "bound", "detect"])):
            result = vlm_engine.visual_grounding(image_path, user_query)
            task_type = "visual_grounding"
        else:
            result = vlm_engine.answer_question(image_path, user_query, chat_history=history)
            task_type = "vqa"

        answer = result.get("answer", "No answer generated.")
        bboxes = result.get("bounding_boxes", [])

        # Automatically render annotated image with bounding boxes for chat display
        annotated_url = None
        if bboxes and len(bboxes) > 0:
            try:
                import uuid
                from backend.model.image_annotator import ImageAnnotator
                out_name = f"{payload.session_id}_{uuid.uuid4().hex[:6]}_annotated.png"
                ImageAnnotator.annotate(image_path, bboxes, out_name)
                annotated_url = f"/annotated/{out_name}"
            except Exception as e:
                print(f"[ChatRouter] Auto-annotation error: {e}")

        # Record turns in session store
        session_store.add_message(payload.session_id, role="user", content=user_query)
        session_store.add_message(payload.session_id, role="assistant", content=answer, boxes=bboxes)

        return {
            "status": "success",
            "session_id": payload.session_id,
            "answer": answer,
            "bounding_boxes": bboxes,
            "task_performed": task_type,
            "device": result.get("device", "unknown"),
            "annotated_image_url": annotated_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Retrieves full conversation history for a given session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "status": "success",
        "session_id": session_id,
        "history": session.get("history", []),
        "image_filename": session.get("image_filename"),
        "image_meta": session.get("image_meta")
    }

@router.delete("/session/{session_id}")
async def clear_chat_session(session_id: str):
    """Clears an active chat session."""
    success = session_store.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "success", "message": f"Session '{session_id}' cleared."}
