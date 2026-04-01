"""
main.py — FastAPI backend for the ML News Classifier.

Run:
    cd backend
    uvicorn api.main:app --reload --port 8000

Swagger docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parents[2]))

from ml_pipeline.models.ensemble import EnsemblePredictor

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML News Classifier API",
    description="Hybrid ensemble: TF-IDF + BiLSTM + DistilBERT",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor once on startup
predictor: Optional[EnsemblePredictor] = None


@app.on_event("startup")
async def load_models():
    global predictor
    print("Loading ensemble models …")
    predictor = EnsemblePredictor()
    print("Models ready ✅")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "Apple launches new AI-powered iPhone with record battery life"}
        }


class ClassifyResponse(BaseModel):
    category:      str
    label:         int
    confidence:    float
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    texts: list[str]


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "ML News Classifier API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return predictor.predict(req.text)


@app.post("/classify/batch")
def classify_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts list is empty")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return {"results": predictor.predict_batch(req.texts)}


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Simple rule-based chat that answers questions about the classifier.
    In Phase 2 this will be wired to a HuggingFace model.
    """
    msg = req.message.lower()

    if any(w in msg for w in ["classify", "what is", "what's"]):
        # Try to classify the user's text
        text = req.message
        if predictor:
            result = predictor.predict(text)
            return {
                "reply": (
                    f"I'd classify that as **{result['category']}** "
                    f"with {result['confidence']*100:.1f}% confidence."
                )
            }

    if "accuracy" in msg or "how good" in msg:
        return {
            "reply": (
                "The ensemble achieves ~94% accuracy on the AG News test set. "
                "Baseline (TF-IDF): ~87%, LSTM: ~92%, DistilBERT: ~94%."
            )
        }

    return {
        "reply": (
            "I can classify news headlines into World, Sports, Business, or Technology. "
            "Try pasting a headline!"
        )
    }


@app.get("/categories")
def categories():
    return {
        "categories": [
            {"id": 0, "name": "World",      "color": "#3B82F6"},
            {"id": 1, "name": "Sports",     "color": "#10B981"},
            {"id": 2, "name": "Business",   "color": "#F59E0B"},
            {"id": 3, "name": "Technology", "color": "#8B5CF6"},
        ]
    }
