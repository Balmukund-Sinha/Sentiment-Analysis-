
import React, { useState, useEffect } from 'react';
import InputCard from './components/InputCard';
import ResultCard from './components/ResultCard';
import ThemeToggle from './components/ThemeToggle';
import BatchProcessor from './components/BatchProcessor';

function App() {
  const [text, setText] = useState('');
  const [explain, setExplain] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [view, setView] = useState('single'); // 'single' or 'batch'

  // Load theme preference on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  const analyzeSentiment = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, explain }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze sentiment');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Analysis Engine Offline. Please check your backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="glass-panel" style={{ padding: '1.5rem 2rem', borderTop: 'none', borderLeft: 'none', borderRight: 'none', borderRadius: '0 0 24px 24px', position: 'sticky', top: 0, zIndex: 1000 }}>
        <div>
          <h1 className="title" style={{ margin: 0, fontSize: '1.75rem' }}>Sentiment Engine <span style={{ fontSize: '0.8rem', verticalAlign: 'middle', opacity: 0.7 }}>v4.0</span></h1>
          <p className="subtitle" style={{ fontSize: '0.85rem' }}>Production-Grade DeBERTa-v3 NLP</p>
        </div>

        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <nav style={{ background: 'rgba(150,150,150,0.1)', padding: '4px', borderRadius: '12px', display: 'flex', gap: '4px' }}>
            <button
              onClick={() => setView('single')}
              style={{ background: view === 'single' ? 'var(--card-bg)' : 'transparent', border: 'none', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', color: 'inherit', fontWeight: view === 'single' ? '700' : '400' }}
            >
              Single
            </button>
            <button
              onClick={() => setView('batch')}
              style={{ background: view === 'batch' ? 'var(--card-bg)' : 'transparent', border: 'none', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', color: 'inherit', fontWeight: view === 'batch' ? '700' : '400' }}
            >
              Batch
            </button>
          </nav>
          <div style={{ width: '1px', height: '20px', background: 'var(--border)' }}></div>
          <ThemeToggle />
        </div>
      </header>

      <main style={{ marginTop: '3rem', width: '100%' }}>
        {error && (
          <div className="glass-panel" style={{ padding: '1rem', marginBottom: '2rem', borderColor: 'var(--danger)', color: 'var(--danger)', textAlign: 'center' }}>
            ⚠️ {error}
          </div>
        )}

        {view === 'single' ? (
          <div className="main-layout">
            <InputCard
              text={text}
              setText={setText}
              analyze={analyzeSentiment}
              loading={loading}
              explain={explain}
              setExplain={setExplain}
            />
            <ResultCard result={result} />
          </div>
        ) : (
          <BatchProcessor />
        )}
      </main>

      <footer className="footer">
        <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'center', gap: '2rem' }}>
          <span>📦 DeBERTa-v3-Base</span>
          <span>🛡️ SHAP XAI</span>
          <span>🚀 BF16 Optimized</span>
        </div>
        <p>© 2026 Mukund MLOps • Built for Performance & Interpretability</p>
      </footer>
    </div>
  );
}

export default App;
