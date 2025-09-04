import React from 'react';

const MathPanel = ({ mathSolution, settings, mathCanvasRef, setActiveTab }) => {
  const renderMathSteps = () => {
    if (!mathSolution || !mathSolution.steps || !settings.showMathSteps) return null;
    
    return (
      <div className="math-steps-container">
        <h4>Solution Steps:</h4>
        <ol className="math-steps-list">
          {mathSolution.steps.map((step, index) => (
            <li key={index} className="math-step">
              {step}
            </li>
          ))}
        </ol>
      </div>
    );
  };

  return (
    <div className="math-panel">
      <div className="math-header">
        <h3>Math Solutions</h3>
        <button
          className="back-button"
          onClick={() => setActiveTab("chat")}
        >
          <i className="icon-arrow-left"></i> Back to Chat
        </button>
      </div>

      {mathSolution ? (
        <div className="math-solution">
          <div className="math-problem">
            <h4>Problem/Expression:</h4>
            <p>{mathSolution.problem || mathSolution.expression}</p>
          </div>
          {mathSolution.solution && (
            <div className="math-answer">
              <h4>Solution:</h4>
              <p>{mathSolution.solution}</p>
            </div>
          )}
          {mathSolution.simplified && (
            <div className="math-answer">
              <h4>Simplified:</h4>
              <p>{mathSolution.simplified}</p>
            </div>
          )}
          {mathSolution.derivative && (
            <div className="math-answer">
              <h4>Derivative (d/d{mathSolution.variable}):</h4>
              <p>{mathSolution.derivative}</p>
            </div>
          )}
          {mathSolution.integral && (
            <div className="math-answer">
              <h4>Integral (∫ d{mathSolution.variable}):</h4>
              <p>{mathSolution.integral}</p>
            </div>
          )}
          {mathSolution.formula && (
            <div className="math-formula">
              <h4>Formula:</h4>
              <p>{mathSolution.formula}</p>
            </div>
          )}
          {settings.showMathSteps && renderMathSteps()}
          <div className="math-visualization">
            <canvas
              ref={mathCanvasRef}
              width="300"
              height="200"
            />
          </div>
        </div>
      ) : (
        <div className="no-math">
          <p>No math solutions yet. Ask me a math question!</p>
        </div>
      )}
    </div>
  );
};

export default MathPanel;