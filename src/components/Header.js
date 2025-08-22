import React from "react";
import { FaCog, FaMicrophone, FaMoon } from "react-icons/fa";

const Header = ({ soulState, settings, setSettings, setShowSettings }) => {
  const toggleTheme = () => {
    const newTheme = settings.theme === "dark" ? "light" : "dark";
    setSettings(prev => ({ ...prev, theme: newTheme }));
    document.body.className = newTheme === "dark" ? "" : "light-theme";
  };

  return (
    <header className="app-header">
      <div className="header-left">
        <h1 className="app-title">AION</h1>
        <div className="soul-status">
          <span className={`mood ${soulState.currentMood}`}>{soulState.currentMood}</span>
          <div className="energy-bar">
            <div className="energy-fill" style={{ width: `${soulState.energyLevel}%` }}></div>
          </div>
        </div>
      </div>
      <div className="header-right">
        <button
          className="icon-button"
          onClick={toggleTheme}
          aria-label="Toggle light/dark theme"
        >
          <FaMoon />
        </button>
        <button
          className="icon-button"
          onClick={() => setShowSettings(true)}
          aria-label="Open settings"
        >
          <FaCog />
        </button>
      </div>
    </header>
  );
};

export default Header;