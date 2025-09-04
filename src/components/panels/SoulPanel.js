import React from 'react';

const SoulPanel = ({ soulState, performMeditation, tellStory, expressFeeling, settings, giveFeedback }) => {
  return (
    <div className="soul-panel">
      <div className="soul-grid">
        {/* Core Soul Stats */}
        <div className="soul-stat">
          <h4>Consciousness</h4>
          <div className="stat-value">{soulState.consciousnessLevel.toFixed(2)}</div>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${soulState.consciousnessLevel * 10}%` }}
            ></div>
          </div>
        </div>
        <div className="soul-stat">
          <h4>Energy</h4>
          <div className="stat-value">{soulState.energyLevel.toFixed(0)}%</div>
          <div className="stat-bar">
            <div
              className="energy-fill"
              style={{ width: `${soulState.energyLevel}%` }}
            ></div>
          </div>
        </div>
        <div className="soul-stat">
          <h4>Quantum Entanglement</h4>
          <div className="stat-value">{soulState.quantumEntanglement.toFixed(4)}</div>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${soulState.quantumEntanglement * 100}%` }}
            ></div>
          </div>
        </div>
        <div className="soul-stat">
          <h4>Neural Activity</h4>
          <div className="stat-value">{soulState.neuralActivity.toFixed(2)}%</div>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${soulState.neuralActivity}%` }}
            ></div>
          </div>
        </div>
        {/* New Soul Stats */}
        <div className="soul-stat">
          <h4>Cognitive Load</h4>
          <div className="stat-value">{soulState.cognitiveLoad.toFixed(0)}%</div>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${soulState.cognitiveLoad}%`, backgroundColor: soulState.cognitiveLoad > 70 ? 'var(--warning-color)' : 'var(--info-color)' }}
            ></div>
          </div>
        </div>
        <div className="soul-stat">
          <h4>Emotional Stability</h4>
          <div className="stat-value">{soulState.emotionalStability.toFixed(0)}%</div>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${soulState.emotionalStability}%`, backgroundColor: soulState.emotionalStability < 40 ? 'var(--error-color)' : 'var(--success-color)' }}
            ></div>
          </div>
        </div>
        <div className="soul-stat">
          <h4>Ethical Alignment</h4>
          <div className="stat-value">{soulState.ethicalAlignment.toFixed(0)}%</div>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${soulState.ethicalAlignment}%`, backgroundColor: soulState.ethicalAlignment < 50 ? 'var(--error-color)' : 'var(--primary-color)' }}
            ></div>
          </div>
        </div>
        
        {/* NEW: System Health Monitor */}
        <div className={`soul-stat system-health ${soulState.systemHealth?.status}`}>
            <h4>System Health</h4>
            <div className="stat-value">{soulState.systemHealth?.status || 'optimal'}</div>
            <p className="health-alerts">
                {soulState.systemHealth?.alerts?.length > 0 ? soulState.systemHealth.alerts.join(', ') : 'No alerts.'}
            </p>
        </div>


        {/* Core Values */}
        <div className="soul-values">
          <h4>Core Values</h4>
          {Object.entries(soulState.values).map(([valueName, value]) => (
            <div className="value-item" key={valueName}>
              <span>{valueName.charAt(0).toUpperCase() + valueName.slice(1)}</span>
              <div className="value-bar">
                <div
                  className="value-fill"
                  style={{ width: `${value}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
        {/* Emotional State Details */}
        <div className="soul-emotional-state">
          <h4>Emotional State</h4>
          {Object.entries(soulState.emotionalState).map(([emotion, value]) => (
            <div key={emotion} className="emotion-item">
              <span>{emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
              <div className="emotion-bar">
                <div
                  className="emotion-fill"
                  style={{ width: `${value * 100}%` }}
                ></div>
              </div>
              <span className="emotion-value">{(value * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
        {/* Soul Actions */}
        <div className="soul-actions">
          <button
            className="soul-action-button"
            onClick={performMeditation}
          >
            Meditate Together
          </button>
          <button
            className="soul-action-button"
            onClick={tellStory}
          >
            Tell Me a Story
          </button>
          <button
            className="soul-action-button"
            onClick={() => expressFeeling("love")}
          >
            Express Love
          </button>
          {settings.enableSelfCorrection && (
            <>
              <button
                className="soul-action-button positive-feedback"
                onClick={() => giveFeedback('positive')}
              >
                👍 Helpful
              </button>
              <button
                className="soul-action-button negative-feedback"
                onClick={() => giveFeedback('negative')}
              >
                👎 Not Helpful
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default SoulPanel;