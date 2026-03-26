import React from 'react';

export default function Navbar({ activePage, setActivePage, apiStatus }) {
  const pages = [
    { id: 'upload', label: 'Upload' },
    { id: 'results', label: 'Results' },
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'gallery', label: 'Gallery' },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <div>
            <div className="navbar-title">WildSight AI</div>
            <div className="navbar-subtitle">eDNA Research Lab &bull; SAIT</div>
          </div>
        </div>

        <ul className="navbar-nav">
          {pages.map(page => (
            <li key={page.id}>
              <button
                className={`nav-btn ${activePage === page.id ? 'active' : ''}`}
                onClick={() => setActivePage(page.id)}
                id={`nav-${page.id}`}
              >
                {page.label}
              </button>
            </li>
          ))}
        </ul>

        <div className="navbar-status">
          <span className={`status-dot ${apiStatus === 'online' ? '' : 'offline'}`} />
          <span>{apiStatus === 'online' ? 'Connected' : 'Demo Mode'}</span>
        </div>
      </div>
    </nav>
  );
}
