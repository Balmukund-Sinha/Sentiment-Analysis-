
import React from 'react';

const ResultCard = ({ result }) => {
    if (!result) {
        return (
            <div className="glass-panel" style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}>
                <p>Awaiting Analysis...</p>
            </div>
        );
    }

    const isPositive = result.sentiment === 'Positive';
    const confidencePercent = Math.round(result.confidence * 100);
    const isReliable = result.status === 'Reliable';

    return (
        <div className="glass-panel action-card result-card-container">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                <span className="subtitle">Prediction Result</span>
                <span className="subtitle" style={{ fontSize: '0.8rem' }}>Model: {result.model_version}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', marginBottom: '2rem' }}>
                <div className={`result-label ${isPositive ? 'result-positive' : 'result-negative'}`} style={{ fontSize: '2rem' }}>
                    {result.sentiment}
                    <span className={`badge ${isReliable ? 'badge-reliable' : 'badge-uncertain'}`} style={{ marginLeft: '1rem', fontSize: '0.7rem' }}>
                        {result.status}
                    </span>
                </div>

                <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                        <span>Confidence</span>
                        <span style={{ fontWeight: '800' }}>{confidencePercent}%</span>
                    </div>
                    <div className="confidence-bg">
                        <div
                            className="confidence-fill"
                            style={{
                                width: `${confidencePercent}%`,
                                backgroundColor: isPositive ? 'var(--success)' : 'var(--danger)'
                            }}
                        />
                    </div>
                </div>
            </div>

            <div style={{ padding: '1rem', background: 'rgba(150,150,150,0.05)', borderRadius: '12px', marginBottom: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>Probability Distribution:</span>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <span style={{ borderLeft: '3px solid var(--success)', paddingLeft: '5px' }}>Pos: {Math.round(result.probabilities.positive * 100)}%</span>
                        <span style={{ borderLeft: '3px solid var(--danger)', paddingLeft: '5px' }}>Neg: {Math.round(result.probabilities.negative * 100)}%</span>
                    </div>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                    ⚡ Latecy: {result.latency_ms}ms
                </div>
            </div>

            {result.explanation && (
                <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                        <h4 style={{ fontWeight: '700' }}>Model Evidence (SHAP)</h4>
                        <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.7rem' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <div style={{ width: 8, height: 8, background: 'rgba(0, 120, 255, 0.6)', borderRadius: '2px' }}></div> Positive
                            </span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <div style={{ width: 8, height: 8, background: 'rgba(255, 60, 60, 0.6)', borderRadius: '2px' }}></div> Negative
                            </span>
                        </div>
                    </div>

                    <div style={{ lineHeight: '2', fontSize: '1.05rem' }}>
                        {result.explanation.map((item, idx) => {
                            const val = item.normalized_score || 0;
                            const isPos = val > 0;
                            const opacity = Math.abs(val) * 0.7;
                            const color = isPos ? `rgba(0, 120, 255, ${opacity})` : `rgba(255, 60, 60, ${opacity})`;

                            return (
                                <span
                                    key={idx}
                                    data-tooltip={`Impact: ${(item.score * 100).toFixed(2)}`}
                                    style={{
                                        backgroundColor: color,
                                        padding: '2px 0px',
                                        borderRadius: '2px',
                                        display: 'inline-block',
                                        whiteSpace: 'pre-wrap',
                                        minWidth: item.token === ' ' ? '0.5em' : 'auto',
                                        borderBottom: opacity > 0.2 ? `2px solid ${isPos ? '#0078FF' : '#FF3C3C'}` : 'none'
                                    }}
                                >
                                    {item.token || '\u00A0'}
                                </span>
                            );
                        })}
                    </div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem', fontStyle: 'italic' }}>
                        *Hover over words to see exact contribution values.
                    </p>
                </div>
            )}
        </div>
    );
};

export default ResultCard;
