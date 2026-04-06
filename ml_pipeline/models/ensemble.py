"""
ensemble.py — Combines TF-IDF, LSTM, and DistilBERT predictions via weighted voting.

Usage:
    from ml_pipeline.models.ensemble import EnsemblePredictor
    predictor = EnsemblePredictor()
    result = predictor.predict("Apple launches new iPhone with AI features")
    # {"category": "Technology", "confidence": 0.94, "probabilities": {...}}
"""

import json
import pickle
import numpy as np
import torch
import joblib
from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

import sys
sys.path.append(str(Path(__file__).parents[2]))
PROJECT_ROOT = Path(__file__).parents[2].resolve()
from ml_pipeline.data.preprocessor import TextPreprocessor

CATEGORIES = ["World", "Sports", "Business", "Technology"]

# Weights: higher = more trust in that model
WEIGHTS = {
    "baseline": 0.20,
    "lstm":     0.30,
    "distilbert": 0.50,
}


class EnsemblePredictor:
    """
    Loads all three trained models and combines their probability outputs.
    Falls back gracefully if some models are not yet trained.
    """

    def __init__(self):
        self.prep   = TextPreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._baseline    = self._load_baseline()
        self._lstm        = self._load_lstm()
        self._distilbert  = self._load_distilbert()

        loaded = [k for k, v in {
            "baseline": self._baseline,
            "lstm": self._lstm,
            "distilbert": self._distilbert,
        }.items() if v is not None]
        print(f"✅ Ensemble loaded: {loaded}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        probs = self._get_weighted_probs(text)
        pred_idx = int(np.argmax(probs))
        return {
            "category":    CATEGORIES[pred_idx],
            "label":       pred_idx,
            "confidence":  float(round(probs[pred_idx], 4)),
            "probabilities": {
                cat: float(round(p, 4))
                for cat, p in zip(CATEGORIES, probs)
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_weighted_probs(self, text: str) -> np.ndarray:
        probs_list, weight_sum = [], 0.0

        if self._baseline:
            p = self._baseline_probs(text)
            probs_list.append(p * WEIGHTS["baseline"])
            weight_sum += WEIGHTS["baseline"]

        if self._lstm:
            p = self._lstm_probs(text)
            probs_list.append(p * WEIGHTS["lstm"])
            weight_sum += WEIGHTS["lstm"]

        if self._distilbert:
            p = self._distilbert_probs(text)
            probs_list.append(p * WEIGHTS["distilbert"])
            weight_sum += WEIGHTS["distilbert"]

        if not probs_list:
            raise RuntimeError("No models loaded!")

        combined = np.sum(probs_list, axis=0) / weight_sum
        return combined / combined.sum()   # normalise

    def _baseline_probs(self, text: str) -> np.ndarray:
        clean = self.prep.clean(text)
        probs = self._baseline.predict_proba([clean])[0]
        return probs

    def _lstm_probs(self, text: str) -> np.ndarray:
        model, vocab = self._lstm
        clean   = self.prep.clean(text)
        ids     = torch.tensor([vocab.encode(clean)], dtype=torch.long).to(self.device)
        model.eval()
        with torch.no_grad():
            logits = model(ids)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    def _distilbert_probs(self, text: str) -> np.ndarray:
        model, tokenizer = self._distilbert
        enc = tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )
        model.eval()
        with torch.no_grad():
            logits = model(
                enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            ).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # ── Loaders (return None if model not yet trained) ────────────────────────

    def _load_baseline(self):
        path = PROJECT_ROOT /"ml_pipeline/models/baseline/tfidf_lr_pipeline.joblib"
        if path.exists():
            return joblib.load(path)
        print(f"⚠️  Baseline model not found at {path}  (run train_baseline.py first)")
        return None

    # def _load_lstm(self):
    #     from ml_pipeline.training.train_lstm import BiLSTMClassifier, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_CLASSES
    #     pt_path   = PROJECT_ROOT /"ml_pipeline/models/lstm/lstm_best.pt"
    #     vocab_path = PROJECT_ROOT /"ml_pipeline/models/lstm/vocab.pkl"
    #     if not pt_path.exists():
    #         print(f"⚠️  LSTM model not found  (run train_lstm.py first)")
    #         return None
    #     with open(vocab_path, "rb") as f:
    #         vocab = pickle.load(f)
    #     model = BiLSTMClassifier(
    #         vocab_size=len(vocab.word2idx),
    #         embed_dim=EMBED_DIM,
    #         hidden_dim=HIDDEN_DIM,
    #         num_layers=NUM_LAYERS,
    #         num_classes=NUM_CLASSES,
    #         dropout=DROPOUT,
    #     ).to(self.device)
    #     model.load_state_dict(torch.load(pt_path, map_location=self.device))
    #     return model, vocab
    def _load_lstm(self):
        from ml_pipeline.training.train_lstm import BiLSTMClassifier, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_CLASSES
        
        pt_path    = PROJECT_ROOT / "ml_pipeline/models/lstm/lstm_best.pt"
        vocab_path = PROJECT_ROOT / "ml_pipeline/models/lstm/vocab.pkl"
        
        if not pt_path.exists() or not vocab_path.exists():
            print(f"⚠️  LSTM model not found (run train_lstm.py first)")
            return None
            
        # === QUICK FIX FOR PICKLE VOCABULARY ERROR ===
        import sys
        from ml_pipeline.training.train_lstm import Vocabulary   # Import the class from where it was defined
        sys.modules['__main__'].Vocabulary = Vocabulary
        # =============================================

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            
        model = BiLSTMClassifier(
            vocab_size=len(vocab.word2idx),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        ).to(self.device)
        
        model.load_state_dict(torch.load(pt_path, map_location=self.device))
        return model, vocab

    def _load_distilbert(self):
        path = PROJECT_ROOT /"ml_pipeline/models/distilbert/best_model"
        if not path.exists():
            print(f"⚠️  DistilBERT model not found  (run train_distilbert.py first)")
            return None
        tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        model = DistilBertForSequenceClassification.from_pretrained(path).to(self.device)
        return model, tokenizer


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = EnsemblePredictor()

    test_headlines = [
        "Manchester United beats Liverpool in dramatic 3-2 comeback",
        "Federal Reserve raises interest rates amid inflation concerns",
        "Apple unveils new AI chip for next-generation MacBooks",
        "UN Security Council meets to discuss humanitarian crisis",
    ]

    print("\n" + "=" * 55)
    for headline in test_headlines:
        result = predictor.predict(headline)
        print(f"\nHeadline : {headline}")
        print(f"Predicted: {result['category']}  ({result['confidence']*100:.1f}%)")
        for cat, p in result["probabilities"].items():
            bar = "█" * int(p * 30)
            print(f"  {cat:<12} {p*100:5.1f}%  {bar}")
