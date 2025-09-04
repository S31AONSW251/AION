import React from 'react';

const Header = ({ soulState, setShowSettings, showSettings, showSoulPanel, setShowSoulPanel }) => {
  return (
    <header className="app-header">
      <div className="header-left">
        <h1>AION</h1>
        <div className="soul-status">
          <span className={`mood ${soulState.currentMood}`}>{soulState.currentMood}</span>
          <div className="energy-bar">
            <div
              className="energy-fill"
              style={{ width: `${soulState.energyLevel}%` }}
              title={`Energy: ${soulState.energyLevel.toFixed(0)}%`}
            ></div>
          </div>
        </div>
      </div>

      <div className="header-right">
        <button
          className={`icon-button ${showSoulPanel ? 'active' : ''}`}
          onClick={() => setShowSoulPanel(!showSoulPanel)}
          title="Soul Panel"
        >
          <i className="icon-soul"></i> {/* Placeholder for a soul icon */}
        </button>
        <button
          className={`icon-button ${showSettings ? 'active' : ''}`}
          onClick={() => setShowSettings(!showSettings)}
          title="Settings"
        >
          <i className="icon-settings"></i> {/* Placeholder for a settings icon */}
        </button>
      </div>
    </header>
  );
};

export default Header;