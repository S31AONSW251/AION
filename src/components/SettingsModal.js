import React from "react";
import { FaTimes, FaVolumeUp } from "react-icons/fa";

const SettingsModal = ({ settings, setSettings, setShowSettings, speak, isSpeechSupported }) => {
  const handleSettingChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="settings-overlay">
      <div className="settings-content">
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="close-button" onClick={() => setShowSettings(false)}>
            <FaTimes />
          </button>
        </div>
        <div className="settings-tabs">
          <button className="settings-tab active">General</button>
          <button className="settings-tab">Advanced</button>
        </div>
        <div className="settings-body">
          <div className="settings-group">
            <h3>General</h3>
            <div className="setting-item toggle">
              <label>Auto-speak replies</label>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={settings.autoSpeakReplies}
                  onChange={(e) => handleSettingChange("autoSpeakReplies", e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="setting-item toggle">
              <label>Show Math Steps</label>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={settings.showMathSteps}
                  onChange={(e) => handleSettingChange("showMathSteps", e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
          </div>
          <div className="settings-group">
            <h3>Advanced</h3>
            <div className="setting-item">
              <label>Personality Intensity</label>
              <input
                type="range"
                min="0"
                max="100"
                value={settings.personalityIntensity}
                onChange={(e) => handleSettingChange("personalityIntensity", e.target.value)}
              />
            </div>
          </div>
        </div>
        <div className="settings-footer">
          <button
            className="test-button"
            onClick={speak}
            disabled={!isSpeechSupported}
          >
            <FaVolumeUp /> Test Voice
          </button>
          <button className="save-button" onClick={() => setShowSettings(false)}>
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;