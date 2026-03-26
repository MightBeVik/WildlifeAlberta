import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

export default function Dashboard({ results, apiStatus }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    if (apiStatus === 'online') {
      api.getStats().then(setStats).catch(() => {});
    }
  }, [apiStatus, results.length]);

  const totalImages = results.length;
  const imagesWithAnimals = results.filter(r => r.detection?.has_animal).length;
  const emptyFrames = totalImages - imagesWithAnimals;

  const speciesCounts = {};
  results.forEach(r => {
    if (r.classification?.classifications) {
      r.classification.classifications.forEach(cls => {
        speciesCounts[cls.species] = (speciesCounts[cls.species] || 0) + 1;
      });
    }
  });

  const sortedSpecies = Object.entries(speciesCounts)
    .sort((a, b) => b[1] - a[1]);

  const maxCount = sortedSpecies.length > 0 ? sortedSpecies[0][1] : 1;

  const totalDetections = results.reduce((sum, r) => {
    return sum + (r.detection?.detections?.filter(d => d.category === 'animal')?.length || 0);
  }, 0);

  const avgConfidence = results.reduce((sum, r) => {
    const confs = r.classification?.classifications?.map(c => c.confidence) || [];
    return sum + confs.reduce((a, b) => a + b, 0);
  }, 0) / Math.max(totalDetections, 1);

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-description">
          Real-time overview of wildlife detection and classification results.
        </p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Images Processed</div>
          <div className="stat-value">{totalImages}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Animals Detected</div>
          <div className="stat-value">{totalDetections}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Species Found</div>
          <div className="stat-value">{sortedSpecies.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Empty Frames</div>
          <div className="stat-value">{emptyFrames}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Confidence</div>
          <div className="stat-value">{(avgConfidence * 100).toFixed(0)}%</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">System Status</div>
          <div className="stat-value" style={{ fontSize: '1.25rem' }}>
            {apiStatus === 'online' ? 'Live' : 'Demo'}
          </div>
        </div>
      </div>

      <div className="chart-section">
        <div className="chart-card">
          <div className="chart-title">Species Distribution</div>
          {sortedSpecies.length > 0 ? (
            <div className="bar-chart">
              {sortedSpecies.map(([species, count]) => (
                <div className="bar-row" key={species}>
                  <span className="bar-label">{species}</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{ width: `${(count / maxCount) * 100}%` }}
                    />
                  </div>
                  <span className="bar-value">{count}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state" style={{ padding: '2rem' }}>
              <div className="empty-state-text">Process images to see species distribution</div>
            </div>
          )}
        </div>

        <div className="chart-card">
          <div className="chart-title">Detection Rate</div>
          {totalImages > 0 ? (
            <div>
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem', marginTop: '0.5rem' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--green-700)' }}>
                    {((imagesWithAnimals / totalImages) * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    Images with Animals
                  </div>
                </div>
              </div>

              <div className="bar-chart">
                <div className="bar-row">
                  <span className="bar-label">With Animals</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${(imagesWithAnimals / totalImages) * 100}%` }} />
                  </div>
                  <span className="bar-value">{imagesWithAnimals}</span>
                </div>
                <div className="bar-row">
                  <span className="bar-label">Empty Frames</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${(emptyFrames / totalImages) * 100}%`,
                        background: 'linear-gradient(90deg, #d4a843, #b8860b)',
                      }}
                    />
                  </div>
                  <span className="bar-value">{emptyFrames}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state" style={{ padding: '2rem' }}>
              <div className="empty-state-text">Process images to see detection rates</div>
            </div>
          )}
        </div>
      </div>

      {results.length > 0 && (
        <div className="chart-card" style={{ marginTop: '1.5rem' }}>
          <div className="chart-title">Recent Activity</div>
          <div className="file-list">
            {results.slice(-5).reverse().map((r, i) => (
              <div className="file-item" key={i}>
                <div className="file-info">
                  <div className="file-name">{r.upload?.filename || 'Unknown'}</div>
                  <div className="file-size">
                    {r.detection?.has_animal
                      ? `${r.classification?.classifications?.length || 0} animal(s) detected`
                      : 'Background frame — no animals'}
                  </div>
                </div>
                <div className={`file-status ${r.detection?.has_animal ? 'done' : 'processing'}`}>
                  {r.detection?.processing_time_ms?.toFixed(0)}ms
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
