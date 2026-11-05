"""
app/routers/chat.py — Document chat and analysis endpoints.

Endpoints
---------
POST /chat/     — conversational Q&A against an OCR'd document
POST /analyze/  — quick single-shot analysis (summary, key points, etc.)
"""

import json
import traceback
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, Form

from app.config import CHAT_CONTEXT_MAX_CHARS
from app.utils.responses import create_error_response
from app.utils.text_processing import smart_truncate_text
from ocr_olm import run_plain_text_inference
from task_manager import get_task_manager

router = APIRouter(tags=["Chat"])


# ---------------------------------------------------------------------------
# /chat/
# ---------------------------------------------------------------------------

@router.post("/chat/")
async def chat_with_document(
    message: str = Form(...),
    extracted_text: str = Form(default=""),
    conversation_history: str = Form(default=""),
    document_name: str = Form(default=""),
    session_id: str = Form(...),
):
    """
    Conversational Q&A endpoint.

    Answers questions strictly from the provided document content.
    Maintains short-term conversation history (last 4 turns).
    """
    try:
        task_manager = get_task_manager()
        if not task_manager.touch_session(session_id):
            return create_error_response(
                "Invalid or expired session", status_code=401
            )

        if not message or not message.strip():
            return create_error_response("Message cannot be empty", status_code=400)

        print(f"[CHAT] Session {session_id}: {message[:100]}...")

        # Truncate the document context to fit the model's effective window.
        context_text = (
            smart_truncate_text(extracted_text, max_chars=CHAT_CONTEXT_MAX_CHARS)
            if extracted_text
            else "No document content available."
        )

        # Parse conversation history (JSON array of {role, content} dicts).
        history: List[Dict[str, str]] = []
        if conversation_history:
            try:
                parsed = json.loads(conversation_history)
                if isinstance(parsed, list):
                    history = parsed
            except json.JSONDecodeError as e:
                print(f"[CHAT] Warning: Failed to parse conversation_history: {e}")

        # Build history text (last 4 turns to keep the prompt lean).
        history_text = ""
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n"

        system_prompt = f"""You are an AI Document Assistant.
Document: {document_name if document_name else "Untitled"}
Content: {context_text}

Previous conversation:
{history_text if history_text else "No previous conversation"}

Current question: {message}

Instructions:
1. Answer strictly based on the provided document content.
2. If the question is unrelated to the document, reply: "I can only answer questions related to the provided document."
3. Do not provide coding assistance unless the document is about code.
4. Provide much more detailed, comprehensive answers.
5. If your answer references tabular data or you are extracting information from a table, you MUST format that information as a Markdown table in your response.

Response:"""

        response = run_plain_text_inference(
            prompt=system_prompt,
            max_new_tokens=2000,
            temperature=0.7,
        )

        print(f"[CHAT] Response generated for session {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "document_context": bool(extracted_text),
            "context_truncated": (
                len(extracted_text) > CHAT_CONTEXT_MAX_CHARS if extracted_text else False
            ),
        }

    except Exception as e:
        print(f"[CHAT] Error: {str(e)}")
        traceback.print_exc()
        return create_error_response(str(e))


# ---------------------------------------------------------------------------
# /analyze/
# ---------------------------------------------------------------------------

# Maps analysis_type -> instruction appended to the system prompt.
_ANALYSIS_PROMPTS: Dict[str, str] = {
    "summary":    "Provide a comprehensive summary of this document.",
    "key_points": "Extract the main key points as a list.",
    "entities":   "Identify and list all important entities (names, dates, amounts).",
    "dates":      "Find and list all dates and deadlines.",
    "questions":  "Suggest 5-7 relevant questions about this document.",
}


@router.post("/analyze/")
async def analyze_document(
    extracted_text: str = Form(...),
    document_name: str = Form(default=""),
    analysis_type: str = Form(default="summary"),
):
    """
    Quick single-shot document analysis.

    analysis_type options: summary | key_points | entities | dates | questions
    """
    try:
        prompt_instruction = _ANALYSIS_PROMPTS.get(
            analysis_type, _ANALYSIS_PROMPTS["summary"]
        )
        context_text = extracted_text[:3500] if extracted_text else ""

        system_prompt = f"""Analyze the following document and {prompt_instruction}

Document: {document_name}
Content: {context_text}

Analysis:"""

        response = run_plain_text_inference(
            prompt=system_prompt,
            max_new_tokens=600,
            temperature=0.6,
        )

        return {
            "success": True,
            "analysis": response,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "document_name": document_name,
        }

    except Exception as e:
        print(f"[ANALYSIS] Error: {str(e)}")
        return create_error_response(str(e))
