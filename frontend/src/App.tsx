import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  // const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<any[]>([]);

  const API_URL = 'http://localhost:8000';   // Change to Render URL later

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // setImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/classify`, { text });
      const prediction = response.data;

      setResult(prediction);

      // Save to history
      setHistory(prev => [prediction, ...prev.slice(0, 4)]);
    } catch (error) {
      alert("❌ Cannot connect to backend. Is FastAPI running on port 8000?");
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <header>
        <div className="logo">📰 NewsMind</div>
        <h1>AI News Classifier</h1>
        <p>Instantly categorize news with 94%+ accuracy</p>
      </header>

      <div className="main-container">
        <form onSubmit={handleSubmit} className="input-card">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your news headline here..."
            rows={4}
          />

          {/* Image Upload */}
          <div className="image-upload">
            <label>
              📸 Upload news image (optional - future OCR feature)
              <input type="file" accept="image/*" onChange={handleImageUpload} />
            </label>
            {imagePreview && <img src={imagePreview} alt="preview" className="preview" />}
          </div>

          {/* Language Translation (Future Feature) */}
          <div className="language-selector">
            <label>Translate news to:</label>
            <select>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
            </select>
          </div>

          <button type="submit" disabled={loading} className="classify-btn">
            {loading ? "Classifying..." : "🔍 Classify News"}
          </button>
        </form>

        {/* Result Section */}
        {result && (
          <div className="result-card">
            <h2>Prediction</h2>
            <div className={`category-badge ${result.category.toLowerCase()}`}>
              {result.category}
            </div>
            <p className="confidence">
              Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
            </p>

            <div className="probabilities">
              {Object.entries(result.probabilities).map(([cat, prob]) => (
                <div key={cat} className="prob-bar">
                  <span className="label">{cat}</span>
                  <div className="bar-bg">
                    <div className="bar-fill" style={{ width: `${(prob as number) * 100}%` }}></div>
                  </div>
                  <span className="value">{((prob as number) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="history">
            <h3>Recent Classifications</h3>
            {history.map((item, index) => (
              <div key={index} className="history-item">
                <span className="history-text">{item.text.substring(0, 60)}...</span>
                <span className="history-category">{item.category}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <footer>
        <p>Built with ❤️ using FastAPI + React + DistilBERT</p>
      </footer>
    </div>
  );
}

export default App;