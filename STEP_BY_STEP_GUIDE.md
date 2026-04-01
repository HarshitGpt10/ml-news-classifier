# ML News Classifier — Complete Step-by-Step Guide
## For PyCharm on Windows

---

## WHAT YOU'RE BUILDING

A production-grade news classifier that uses THREE machine learning models together:

| Model | Type | Expected Accuracy |
|-------|------|-------------------|
| TF-IDF + Logistic Regression | Classical ML | ~87% |
| Bidirectional LSTM | Deep Learning | ~92% |
| DistilBERT | Transformer (AI) | ~94% |
| **Ensemble (all 3 combined)** | **Hybrid** | **~94-95%** |

Plus a FastAPI backend, chatbot, and React frontend.

---

## PHASE 1 — Project Setup & Baseline Model

### Step 1: Create a GitHub repository

1. Go to **github.com** and sign in (create account if needed)
2. Click the **"+"** button (top right) → **"New repository"**
3. Fill in:
   - Repository name: `ml-news-classifier`
   - Description: `Production ML news classification system`
   - Visibility: **Public**
   - ✅ Check **"Add a README file"**
4. Click **"Create repository"**
5. Copy the URL shown (looks like: `https://github.com/YOUR_NAME/ml-news-classifier.git`)

---

### Step 2: Download and extract the project files

1. Download the ZIP file attached to this guide
2. Extract it to: `C:\Users\YourName\Projects\ml-news-classifier`
3. Make sure the folder structure looks like this:
   ```
   ml-news-classifier/
   ├── ml_pipeline/
   │   ├── data/
   │   │   ├── preprocessor.py
   │   │   └── download_dataset.py
   │   ├── training/
   │   │   ├── train_baseline.py
   │   │   ├── train_lstm.py
   │   │   └── train_distilbert.py
   │   └── models/
   │       └── ensemble.py
   ├── backend/
   │   ├── api/
   │   │   └── main.py
   │   └── config/
   │       └── schema.sql
   ├── tests/
   │   └── unit/
   │       └── test_preprocessor.py
   ├── .env.example
   ├── .gitignore
   └── requirements.txt  ← (inside ml_pipeline/)
   ```

---

### Step 3: Open in PyCharm

1. Open **PyCharm**
2. Click **File → Open**
3. Navigate to `C:\Users\YourName\Projects\ml-news-classifier`
4. Click **OK**
5. If a popup says **"Trust this project?"** → click **Trust Project**

---

### Step 4: Check Python version

1. In PyCharm, press **Alt + F12** to open the Terminal (black panel at bottom)
2. Type this and press Enter:
   ```
   python --version
   ```
3. You should see **Python 3.10** or **3.11** or **3.12**
4. If you see 3.8 or 3.9, you need to upgrade Python from python.org

---

### Step 5: Create a virtual environment

In the PyCharm terminal, run these commands **one at a time**:

```bash
# Navigate into the ml_pipeline folder
cd ml_pipeline

# Create a virtual environment (isolated Python for this project)
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
```

After activation, you'll see **(venv)** at the start of your terminal line. This means it worked! ✅

---

### Step 6: Install all dependencies

Still in the terminal (with venv active):

```bash
pip install -r requirements.txt
```

⏳ **This takes 5–15 minutes** — it's downloading PyTorch, HuggingFace, scikit-learn, etc.
You'll see lots of text scrolling — that's normal. Wait for it to finish.

When done, you'll see something like:
```
Successfully installed torch-2.3.0 transformers-4.41.1 ...
```

---

### Step 7: Download the AG News dataset

```bash
cd ..
python ml_pipeline/data/download_dataset.py
```

This downloads ~120,000 news articles. You'll see:
```
⬇️  Downloading AG News from HuggingFace (≈ 30 MB) …
✅ Saved 84,000 records → ml_pipeline/data/raw/train.json
✅ Saved 18,000 records → ml_pipeline/data/raw/val.json
✅ Saved 18,000 records → ml_pipeline/data/raw/test.json
🎉 Dataset ready!  Total: 120,000 articles
```

---

### Step 8: Train your first model (Baseline)

```bash
python ml_pipeline/training/train_baseline.py
```

This trains the TF-IDF + Logistic Regression model. Takes ~2–3 minutes.

When done you'll see:
```
  Test Accuracy : 87.20%
  Test F1       : 87.15%
✅ Baseline training complete!
```

A confusion matrix image is saved at `ml_pipeline/models/baseline/confusion_matrix.png` — open it in PyCharm to see how well each category was classified.

---

### Step 9: Push to GitHub

In the terminal:

```bash
git init
git add .
git commit -m "feat: Phase 1 — baseline model trained"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ml-news-classifier.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## PHASE 2 — Deep Learning Models

### Step 10: Train the LSTM model

```bash
python ml_pipeline/training/train_lstm.py
```

⏳ Takes 10–30 minutes (faster with GPU).

Expected output at the end:
```
  Test Accuracy: 91.80%
✅ LSTM training complete!
```

---

### Step 11: Train DistilBERT (the most powerful model)

```bash
python ml_pipeline/training/train_distilbert.py
```

⏳ **This is the slowest step** — 30–90 minutes on CPU, 10–20 min on GPU.

If you get an **out-of-memory error**, open `train_distilbert.py` and change:
```python
BATCH_SIZE = 16   # change this to...
BATCH_SIZE = 8    # smaller number
```

Expected output:
```
  Test Accuracy: 94.20%
✅ DistilBERT fine-tuning complete!
```

---

### Step 12: Test the ensemble

```bash
python ml_pipeline/models/ensemble.py
```

You'll see all 3 models combining their predictions:
```
Headline : Manchester United beats Liverpool in dramatic 3-2 comeback
Predicted: Sports  (97.3%)
  World        2.1%  ▌
  Sports      97.3%  ██████████████████████████████
  Business     0.4%
  Technology   0.2%
```

---

## PHASE 3 — Backend API

### Step 13: Run the FastAPI backend

```bash
pip install fastapi uvicorn
cd backend
uvicorn api.main:app --reload --port 8000
```

Open your browser and go to: **http://localhost:8000/docs**

You'll see an interactive API page where you can test classification!

Try clicking on `/classify` → **Try it out** → paste a headline → **Execute**.

---

### Step 14: Test the API

In a NEW terminal (keep the first one running):

```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Apple launches revolutionary new AI chip\"}"
```

Response:
```json
{
  "category": "Technology",
  "label": 3,
  "confidence": 0.9421,
  "probabilities": {
    "World": 0.012,
    "Sports": 0.008,
    "Business": 0.038,
    "Technology": 0.942
  }
}
```

---

## PHASE 4 — Frontend (Optional, requires Node.js)

### Step 15: Install Node.js

Download from **nodejs.org** → install LTS version.

### Step 16: Create the React frontend

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install axios
npm run dev
```

Open: **http://localhost:5173**

---

## RUNNING TESTS

```bash
pytest tests/ -v
```

Expected:
```
tests/unit/test_preprocessor.py::TestTextPreprocessor::test_lowercase PASSED
tests/unit/test_preprocessor.py::TestTextPreprocessor::test_removes_html PASSED
...
11 passed in 2.30s
```

---

## COMMON ERRORS AND FIXES

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` again with venv active |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in the training script |
| `FileNotFoundError: train.json` | Run `download_dataset.py` first |
| `venv\Scripts\activate` not recognized | Run in PowerShell as Administrator |
| `python` not found | Install Python from python.org and add to PATH |

---

## PROJECT SUMMARY

After completing all phases, you'll have:
- ✅ 3 trained ML models with ~94% accuracy
- ✅ FastAPI REST API with swagger docs
- ✅ Database schema for storing predictions
- ✅ Unit tests
- ✅ GitHub repository with full history

This is a **portfolio-grade project** demonstrating:
Classical ML → Deep Learning → Transformers → MLOps → API → Frontend
