import React, { useRef, useEffect } from 'react';

export default function Results({ results }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  const latest = results.length > 0 ? results[results.length - 1] : null;

  useEffect(() => {
    if (!latest || !canvasRef.current || !imgRef.current) return;

    const img = imgRef.current;
    const canvas = canvasRef.current;

    const drawBoxes = () => {
      const ctx = canvas.getContext('2d');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      const detections = latest.detection?.detections || [];
      const classifications = latest.classification?.classifications || [];

      detections.forEach((det, i) => {
        if (det.category !== 'animal') return;

        const x1 = det.x1 * canvas.width;
        const y1 = det.y1 * canvas.height;
        const x2 = det.x2 * canvas.width;
        const y2 = det.y2 * canvas.height;
        const w = x2 - x1;
        const h = y2 - y1;

        ctx.strokeStyle = '#2d5f2d';
        ctx.lineWidth = Math.max(2, canvas.width * 0.003);
        ctx.strokeRect(x1, y1, w, h);

        const cornerLen = Math.min(w, h) * 0.15;
        ctx.strokeStyle = '#2d5f2d';
        ctx.lineWidth = Math.max(3, canvas.width * 0.005);

        ctx.beginPath();
        ctx.moveTo(x1, y1 + cornerLen);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + cornerLen, y1);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x2 - cornerLen, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + cornerLen);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x1, y2 - cornerLen);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1 + cornerLen, y2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x2 - cornerLen, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - cornerLen);
        ctx.stroke();

        const cls = classifications[i];
        const label = cls ? `${cls.species} ${(cls.confidence * 100).toFixed(0)}%` : `Animal ${(det.confidence * 100).toFixed(0)}%`;
        const fontSize = Math.max(14, canvas.width * 0.018);
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textWidth = ctx.measureText(label).width;
        const padding = 8;

        ctx.fillStyle = 'rgba(26, 58, 26, 0.9)';
        ctx.fillRect(x1, y1 - fontSize - padding * 2, textWidth + padding * 2, fontSize + padding * 2);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + padding, y1 - padding);
      });
    };

    if (img.complete) {
      drawBoxes();
    } else {
      img.onload = drawBoxes;
    }
  }, [latest]);

  if (!latest) {
    return (
      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">Detection Results</h1>
          <p className="page-description">Detection and classification results will appear here after processing images.</p>
        </div>
        <div className="empty-state">
          <div className="empty-state-title">No results yet</div>
          <div className="empty-state-text">Upload and process images to see detection results with bounding boxes.</div>
        </div>
      </div>
    );
  }

  const detections = latest.detection?.detections?.filter(d => d.category === 'animal') || [];
  const classifications = latest.classification?.classifications || [];
  const imageUrl = latest.upload?.filepath || latest.detection?.image_url;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Detection Results</h1>
        <p className="page-description">
          {latest.upload?.filename} — Processed in {latest.detection?.processing_time_ms?.toFixed(0)}ms
        </p>
      </div>

      <div className="results-grid">
        <div className="image-viewer">
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Detection result"
            style={{ display: 'none' }}
            crossOrigin="anonymous"
          />
          <canvas ref={canvasRef} style={{ width: '100%', height: 'auto' }} />
          
          {!latest.detection?.has_animal && (
            <div style={{ padding: '1rem' }}>
              <div className="no-animal-panel">
                <p>No animals detected — background frame</p>
              </div>
            </div>
          )}
        </div>

        <div className="detection-sidebar">
          <div className="detection-card">
            <div className="card-header">
              <span className="card-title">Detection Summary</span>
              <span className={`card-badge ${latest.detection?.has_animal ? 'badge-success' : 'badge-warning'}`}>
                {latest.detection?.has_animal ? 'ANIMAL FOUND' : 'EMPTY FRAME'}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--green-800)' }}>{detections.length}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Animals Detected</div>
              </div>
              <div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--green-800)' }}>{classifications.length}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Species Classified</div>
              </div>
            </div>
          </div>

          {classifications.map((cls, i) => (
            <div className="detection-card" key={i}>
              <div className="card-header">
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Detection #{i + 1}
                </span>
              </div>
              <div className="species-name">{cls.species}</div>
              <div className="species-confidence">
                <div className="confidence-bar">
                  <div
                    className={`confidence-fill ${cls.confidence >= 0.7 ? 'high' : cls.confidence >= 0.4 ? 'medium' : 'low'}`}
                    style={{ width: `${cls.confidence * 100}%` }}
                  />
                </div>
                <span className="confidence-label">{(cls.confidence * 100).toFixed(1)}%</span>
              </div>

              {cls.top_5 && cls.top_5.length > 1 && (
                <div className="top-predictions">
                  <h4>Other Possibilities</h4>
                  {cls.top_5.slice(1).map((pred, j) => (
                    <div className="prediction-row" key={j}>
                      <span className="prediction-species">{pred.species}</span>
                      <span className="prediction-conf">{(pred.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
