import React from 'react';

export default function Gallery({ results, setActivePage }) {
  if (results.length === 0) {
    return (
      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">Gallery</h1>
          <p className="page-description">Browse all processed camera trap images.</p>
        </div>
        <div className="empty-state">
          <div className="empty-state-title">Gallery is empty</div>
          <div className="empty-state-text">Upload and process images to populate the gallery.</div>
          <button
            className="btn btn-primary"
            style={{ marginTop: '1rem' }}
            onClick={() => setActivePage('upload')}
          >
            Go to Upload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Gallery</h1>
        <p className="page-description">
          {results.length} processed image{results.length !== 1 ? 's' : ''} — Click to view details.
        </p>
      </div>

      <div className="gallery-grid">
        {results.map((r, i) => {
          const imageUrl = r.upload?.filepath || r.detection?.image_url;
          const hasAnimal = r.detection?.has_animal;
          const species = r.classification?.classifications?.map(c => c.species) || [];

          return (
            <div
              className="gallery-item"
              key={i}
              onClick={() => setActivePage('results')}
              id={`gallery-item-${i}`}
            >
              <img
                src={imageUrl}
                alt={r.upload?.filename || 'Camera trap image'}
                className="gallery-thumb"
                loading="lazy"
              />
              <div className="gallery-info">
                <div className="gallery-filename">
                  {r.upload?.filename || `Image ${i + 1}`}
                </div>
                <div className="gallery-tags">
                  {hasAnimal ? (
                    species.map((s, j) => (
                      <span className="gallery-tag" key={j}>{s}</span>
                    ))
                  ) : (
                    <span className="gallery-tag empty">Empty Frame</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
