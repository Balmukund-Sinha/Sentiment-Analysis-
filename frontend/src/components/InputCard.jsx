
import React from 'react';

const InputCard = ({ text, setText, analyze, loading, explain, setExplain }) => {
    const charLimit = 512;
    const charsUsed = text.length;
    const isTooLong = charsUsed > 2000; // Character approximation for token limit

    return (
        <div className="glass-panel action-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '1.25rem', fontWeight: '700' }}>Review Input</h3>
                <span className={`badge ${isTooLong ? 'badge-negative' : ''}`} style={{ fontSize: '0.7rem' }}>
                    {charsUsed} characters {isTooLong && '(Approaching API Limit)'}
                </span>
            </div>

            <textarea
                className="input-area"
                placeholder="Paste a movie review here (e.g., 'The screenplay was brilliant and the acting was top-notch...')"
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={loading}
            />

            <div className="controls-row">
                <label className="switch-group">
                    <input
                        type="checkbox"
                        checked={explain}
                        onChange={(e) => setExplain(e.target.checked)}
                        style={{ width: '1.2rem', height: '1.2rem', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.95rem', fontWeight: '500' }}>Enable Explainability (SHAP) 🧠</span>
                </label>

                <button
                    className="btn-primary"
                    onClick={analyze}
                    disabled={loading || !text.trim()}
                >
                    {loading ? (
                        <>
                            <div className="spinner" style={{ border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%' }}></div>
                            Analyzing...
                        </>
                    ) : (
                        <>
                            <svg style={{ width: 20, height: 20 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            Analyze Sentiment
                        </>
                    )}
                </button>
            </div>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem' }}>
                <span>⚡ Real-time Inference</span>
                <span>•</span>
                <span>🔒 Privacy Guaranteed</span>
            </div>
        </div>
    );
};

export default InputCard;
