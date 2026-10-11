"""
FastAPI app for AI Video Assistant

Endpoints
---------
POST /summarize  -> download audio, transcribe, summarize, build vector store
POST /chat       -> ask questions about a previously summarized video (RAG)
GET  /health     -> simple health check

Run:
    uvicorn api:app --reload
Docs:
    http://127.0.0.1:7900/docs
"""

import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Literal


from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain
from core.vector_store import build_vector_store

from dotenv import load_dotenv
load_dotenv()


# 1. app
app = FastAPI(title="AI Video Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=".")

# In-memory session store: session_id -> {"transcript": str, "vector_store": obj}
SESSIONS: Dict[str, dict] = {}

# 2. Pydantic Schema
class SummarizeRequest(BaseModel):
    source: str = Field(..., description="YouTube URL or local file path")
    language: Literal["english", "hinglish"] = "english"


class SummarizeResponse(BaseModel):
    session_id: str
    title: str
    summary: str
    action_items: str
    key_decisions: str
    open_questions: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Copy session_id from summary response")
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    answer: str


# 3. Endpoints
# Plain `def` (not async) on purpose: the pipeline is blocking, so FastAPI
# runs it in a worker thread and the server stays responsive.
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_video(req: SummarizeRequest):
    try:
        chunks = process_input(req.source)
        translate = True if req.language != "english" else False
        transcript = transcribe_all(chunks, translate)

        if not transcript or not transcript.strip():
            raise HTTPException(status_code=422, detail="Transcript is empty.")

        summary = summarize(transcript)
        title = generate_title(summary)
        action_items = extract_action_items(summary)
        key_decisions = extract_key_decisions(summary)
        open_questions = extract_questions(summary)

        # Build the vector store now so /chat is instant later
        vector_store = build_vector_store(transcript)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"transcript": transcript, "vector_store": vector_store}

    return SummarizeResponse(
        session_id=session_id,
        title=title,
        summary=summary,
        action_items=action_items,
        key_decisions=key_decisions,
        open_questions=open_questions,
    )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /summarize first and use its session_id.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    

    try:
        answer = build_rag_chain(question=question, vector_store=session["vector_store"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
    

    return ChatResponse(session_id=req.session_id, answer=answer)

app.mount('/static', StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request : Request):
    return templates.TemplateResponse(request, "static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(SESSIONS)}






