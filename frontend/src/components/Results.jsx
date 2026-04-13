import React, { useEffect, useRef, useState } from 'react';

export default function Results({ results }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const latest = results.length > 0 ? results[results.length - 1] : null;

  const comparisonData = latest?.comparisons || {};
  const detectionsByModel = comparisonData.detectionsByModel || (
    latest?.detection
      ? { [latest.detection.primary_detector || latest.detection.detector_key || 'primary']: latest.detection }
      : {}
  );
  const classificationsByModel = comparisonData.classificationsByModel || (
    latest?.classification
      ? { [latest?.detection?.primary_detector || latest?.detection?.detector_key || 'primary']: latest.classification }
      : {}
  );
  const detectorOrder = comparisonData.detectorOrder || Object.keys(detectionsByModel);
  const availableDetectors = comparisonData.availableDetectors || latest?.detection?.available_detectors || [];

  const [selectedDetectorKey, setSelectedDetectorKey] = useState(
    comparisonData.primaryDetector || latest?.detection?.primary_detector || detectorOrder[0] || null
  );

  useEffect(() => {
    setSelectedDetectorKey(comparisonData.primaryDetector || latest?.detection?.primary_detector || detectorOrder[0] || null);
  }, [latest, comparisonData.primaryDetector, latest?.detection?.primary_detector, detectorOrder]);

  const selectedDetection = selectedDetectorKey ? detectionsByModel[selectedDetectorKey] : null;
  const selectedClassification = selectedDetectorKey ? classificationsByModel[selectedDetectorKey] : null;
  const animalDetections = selectedDetection?.detections?.filter(det => det.category === 'animal') || [];
  const classifications = selectedClassification?.classifications || [];
  const imageUrl = latest?.upload?.filepath || selectedDetection?.image_url || latest?.detection?.image_url;

  useEffect(() => {
    if (!selectedDetection || !canvasRef.current || !imgRef.current) return;

    const img = imgRef.current;
    const canvas = canvasRef.current;

    const drawBoxes = () => {
      const ctx = canvas.getContext('2d');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      animalDetections.forEach((det, index) => {
        const x1 = det.x1 * canvas.width;
        const y1 = det.y1 * canvas.height;
        const x2 = det.x2 * canvas.width;
        const y2 = det.y2 * canvas.height;
        const width = x2 - x1;
        const height = y2 - y1;

        ctx.strokeStyle = '#2d5f2d';
        ctx.lineWidth = Math.max(2, canvas.width * 0.003);
        ctx.strokeRect(x1, y1, width, height);

        const cornerLength = Math.min(width, height) * 0.15;
        ctx.lineWidth = Math.max(3, canvas.width * 0.005);

        ctx.beginPath();
        ctx.moveTo(x1, y1 + cornerLength);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + cornerLength, y1);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x2 - cornerLength, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + cornerLength);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x1, y2 - cornerLength);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1 + cornerLength, y2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x2 - cornerLength, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - cornerLength);
        ctx.stroke();

        const cls = classifications[index];
        const label = cls
          ? `${cls.species} ${(cls.confidence * 100).toFixed(0)}%`
          : `Animal ${(det.confidence * 100).toFixed(0)}%`;
        const fontSize = Math.max(14, canvas.width * 0.018);
        const padding = 8;

        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textWidth = ctx.measureText(label).width;
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
  }, [selectedDetection, animalDetections, classifications]);

  if (!latest) {
    return (
      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">Detection Results</h1>
          <p className="page-description">Detection and classification results will appear here after processing images.</p>
        </div>
        <div className="empty-state">
          <div className="empty-state-title">No results yet</div>
          <div className="empty-state-text">Upload and process images to see detector comparisons with bounding boxes.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Detection Results</h1>
        <p className="page-description">
          {latest.upload?.filename} — compared across {detectorOrder.length} detector{detectorOrder.length !== 1 ? 's' : ''}
        </p>
      </div>

      {detectorOrder.length > 0 && (
        <div className="detector-tabs">
          {detectorOrder.map(detectorKey => {
            const detection = detectionsByModel[detectorKey];
            return (
              <button
                key={detectorKey}
                className={`detector-tab ${detectorKey === selectedDetectorKey ? 'active' : ''}`}
                onClick={() => setSelectedDetectorKey(detectorKey)}
              >
                <span>{detection?.detector_label || detectorKey}</span>
                <span className="detector-tab-meta">
                  {detection?.has_animal ? `${detection.detections.filter(det => det.category === 'animal').length} animal(s)` : 'Empty frame'}
                </span>
              </button>
            );
          })}
        </div>
      )}

      <div className="detector-comparison-grid">
        {detectorOrder.map(detectorKey => {
          const detection = detectionsByModel[detectorKey];
          const animalCount = detection?.detections?.filter(det => det.category === 'animal').length || 0;
          return (
            <button
              type="button"
              key={detectorKey}
              className={`detector-compare-card ${detectorKey === selectedDetectorKey ? 'selected' : ''}`}
              onClick={() => setSelectedDetectorKey(detectorKey)}
            >
              <div className="detector-meta-row">
                <strong>{detection?.detector_label || detectorKey}</strong>
                <span className={`card-badge ${detection?.has_animal ? 'badge-success' : 'badge-warning'}`}>
                  {detection?.has_animal ? 'animal found' : 'empty'}
                </span>
              </div>
              <div className="detector-note">{detection?.detector_description}</div>
              <div className="detector-meta-row">
                <span>{animalCount} animal box{animalCount !== 1 ? 'es' : ''}</span>
                <span>{detection?.processing_time_ms?.toFixed(0) || 0}ms</span>
              </div>
            </button>
          );
        })}
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

          {!selectedDetection?.has_animal && (
            <div style={{ padding: '1rem' }}>
              <div className="no-animal-panel">
                <p>No animals detected by {selectedDetection?.detector_label || 'this detector'}.</p>
              </div>
            </div>
          )}
        </div>

        <div className="detection-sidebar">
          <div className="detection-card">
            <div className="card-header">
              <span className="card-title">Selected Detector</span>
              <span className={`card-badge ${selectedDetection?.has_animal ? 'badge-success' : 'badge-warning'}`}>
                {selectedDetection?.detector_mode || 'unknown'}
              </span>
            </div>
            <div className="species-name" style={{ fontSize: '1.1rem' }}>{selectedDetection?.detector_label || 'No detector selected'}</div>
            <div className="detector-note" style={{ marginTop: '0.5rem' }}>{selectedDetection?.detector_description}</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginTop: '1rem' }}>
              <div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--green-800)' }}>{animalDetections.length}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Animal Boxes</div>
              </div>
              <div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--green-800)' }}>{classifications.length}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Species Classified</div>
              </div>
            </div>
            <div className="detector-meta-row" style={{ marginTop: '1rem' }}>
              <span>Inference time</span>
              <span>{selectedDetection?.processing_time_ms?.toFixed(0) || 0}ms</span>
            </div>
          </div>

          {classifications.length === 0 && selectedDetection?.has_animal && (
            <div className="detection-card">
              <div className="card-title">Classification Pending</div>
              <div className="detector-note" style={{ marginTop: '0.5rem' }}>
                Detection ran, but no classification result was returned for this detector yet.
              </div>
            </div>
          )}

          {classifications.map((cls, index) => (
            <div className="detection-card" key={index}>
              <div className="card-header">
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Detection #{index + 1}
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
                  {cls.top_5.slice(1).map((prediction, predictionIndex) => (
                    <div className="prediction-row" key={predictionIndex}>
                      <span className="prediction-species">{prediction.species}</span>
                      <span className="prediction-conf">{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}

          {availableDetectors.some(detector => detector.mode === 'unavailable') && (
            <div className="detection-card">
              <div className="card-title">Waiting on More Models</div>
              <div className="top-predictions" style={{ marginTop: '0.75rem' }}>
                {availableDetectors
                  .filter(detector => detector.mode === 'unavailable')
                  .map(detector => (
                    <div className="prediction-row" key={detector.key}>
                      <span className="prediction-species">{detector.label}</span>
                      <span className="prediction-conf">{detector.error || 'Not loaded'}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
