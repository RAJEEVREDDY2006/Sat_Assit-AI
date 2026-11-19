"""
In-Memory Session & History Store for SatAssist Chatbot
Author: Sat_Assist Team
"""

import time
from typing import Dict, List, Any, Optional

class SessionStore:
    def __init__(self, max_idle_seconds: int = 3600 * 4):
        # session_id -> session_data
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_idle_seconds = max_idle_seconds

    def create_session(
        self,
        session_id: str,
        image_path: str,
        image_filename: str,
        image_meta: Dict[str, Any],
        is_sample: bool = False
    ) -> Dict[str, Any]:
        """Initializes a new session for an uploaded or preset satellite image."""
        session_data = {
            "session_id": session_id,
            "image_path": image_path,
            "image_filename": image_filename,
            "image_meta": image_meta,
            "is_sample": is_sample,
            "created_at": time.time(),
            "last_active": time.time(),
            "history": []  # List of {"role": "user" | "assistant", "content": str, "boxes": []}
        }
        self.sessions[session_id] = session_data
        return session_data

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(session_id)
        if session:
            session["last_active"] = time.time()
        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        boxes: Optional[List[Dict[str, Any]]] = None
    ):
        session = self.get_session(session_id)
        if session:
            session["history"].append({
                "role": role,
                "content": content,
                "boxes": boxes or [],
                "timestamp": time.time()
            })

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        return session.get("history", []) if session else []

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

# Global singleton
session_store = SessionStore()
