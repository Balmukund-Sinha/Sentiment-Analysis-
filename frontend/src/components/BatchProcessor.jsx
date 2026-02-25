
import React, { useState } from 'react';

const BatchProcessor = () => {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [stats, setStats] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const processBatch = async () => {
        if (!file) return;
        setLoading(true);
        setResults([]);

        const reader = new FileReader();
        reader.onload = async (e) => {
            const text = e.target.result;
            // Simple line-based splitting for batch processing
            const lines = text.split('\n').filter(l => l.trim().length > 10).slice(0, 500); // Increased limit to 500 for production use

            try {
                const response = await fetch('http://localhost:8000/predict_batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ texts: lines, explain: false })
                });
                const data = await response.json();
                setResults(data.map((res, i) => ({ ...res, text: lines[i] })));

                // Calculate stats
                const posCount = data.filter(r => r.sentiment === 'Positive').length;
                setStats({
                    total: data.length,
                    positive: posCount,
                    negative: data.length - posCount,
                    avgConf: (data.reduce((acc, r) => acc + r.confidence, 0) / data.length).toFixed(2)
                });
            } catch (err) {
                console.error(err);
                alert("Batch processing failed. Check console.");
            } finally {
                setLoading(false);
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="glass-panel" style={{ padding: '2.5rem' }}>
            <div style={{ marginBottom: '2rem' }}>
                <h3 style={{ fontSize: '1.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>Batch Review Analysis</h3>
                <p className="subtitle">Upload a .txt or .csv file to analyze multiple reviews at once (Up to 500 reviews).</p>
            </div>

            <div style={{
                border: '2px dashed var(--border)',
                borderRadius: '16px',
                padding: '3rem',
                textAlign: 'center',
                background: file ? 'rgba(99, 102, 241, 0.05)' : 'transparent',
                transition: 'all 0.3s ease'
            }}>
                <input
                    type="file"
                    accept=".txt,.csv"
                    onChange={handleFileChange}
                    id="batch-upload"
                    style={{ display: 'none' }}
                />
                <label htmlFor="batch-upload" style={{ cursor: 'pointer' }}>
                    <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📁</div>
                    <div style={{ fontWeight: '700', fontSize: '1.1rem' }}>{file ? file.name : 'Choose a file or drop it here'}</div>
                    <div className="subtitle" style={{ marginTop: '0.5rem' }}>Supports .txt and .csv (one review per line)</div>
                </label>
            </div>

            {file && !loading && results.length === 0 && (
                <button
                    className="btn-primary"
                    onClick={processBatch}
                    style={{ marginTop: '2rem', width: 'auto' }}
                >
                    Start Batch Analysis
                </button>
            )}

            {loading && (
                <div style={{ textAlign: 'center', marginTop: '3rem' }}>
                    <div className="spinner" style={{ margin: '0 auto 1rem', width: '40px', height: '40px', borderWidth: '4px' }}></div>
                    <p className="subtitle">Processing bulk inference on GPU...</p>
                </div>
            )}

            {stats && (
                <div style={{ marginTop: '3rem', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div className="subtitle">Total</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '800' }}>{stats.total}</div>
                    </div>
                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div className="subtitle">Positive</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--success)' }}>{stats.positive}</div>
                    </div>
                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div className="subtitle">Negative</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--danger)' }}>{stats.negative}</div>
                    </div>
                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div className="subtitle">Avg Conf</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '800' }}>{stats.avgConf}</div>
                    </div>
                </div>
            )}

            {results.length > 0 && (
                <div style={{ marginTop: '2rem', overflowX: 'auto' }}>
                    <table className="batch-table">
                        <thead>
                            <tr>
                                <th>Review Snippet</th>
                                <th>Sentiment</th>
                                <th>Confidence</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results.map((res, i) => (
                                <tr key={i}>
                                    <td style={{ maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.9rem' }}>
                                        {res.text}
                                    </td>
                                    <td>
                                        <span className={`badge ${res.sentiment === 'Positive' ? 'badge-positive' : 'badge-negative'}`}>
                                            {res.sentiment}
                                        </span>
                                    </td>
                                    <td style={{ fontWeight: '700' }}>{Math.round(res.confidence * 100)}%</td>
                                    <td>
                                        <span className={`badge ${res.status === 'Reliable' ? 'badge-reliable' : 'badge-uncertain'}`}>
                                            {res.status}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default BatchProcessor;
