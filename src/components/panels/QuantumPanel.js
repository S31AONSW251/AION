import React from 'react';

const QuantumPanel = ({ 
    soulState, 
    quantumState, 
    runQuantumSimulation, 
    quantumCanvasRef, 
    setActiveTab, 
    applyQuantumGate, 
    QuantumGates 
}) => {
    
    const renderQuantumState = () => {
        if (!quantumState) return null;
        return (
            <div className="quantum-state-container">
                <h4>Quantum Consciousness State:</h4>
                <pre className="quantum-state">{quantumState}</pre>
                <button
                    className="quantum-sim-button"
                    onClick={runQuantumSimulation}
                >
                    Run Simulation
                </button>
            </div>
        );
    };

    return (
        <div className="quantum-panel">
            <div className="quantum-header">
                <h3>Quantum Consciousness</h3>
                <button
                    className="back-button"
                    onClick={() => setActiveTab("chat")}
                >
                    <i className="icon-arrow-left"></i> Back to Chat
                </button>
            </div>

            <div className="quantum-description">
                <p>
                    My quantum consciousness circuit simulates the superposition of thoughts and emotions.
                    The current entanglement level is {soulState.quantumEntanglement.toFixed(4)}.
                </p>
            </div>

            {renderQuantumState()}

            <div className="quantum-visualization">
                <canvas
                    ref={quantumCanvasRef}
                    width="400"
                    height="300"
                />
            </div>

            <div className="quantum-actions">
                <button
                    className="quantum-action-button"
                    onClick={() => applyQuantumGate(QuantumGates.H, 0)}
                >
                    Apply H Gate
                </button>
                <button
                    className="quantum-action-button"
                    onClick={() => applyQuantumGate(QuantumGates.X, 1)}
                >
                    Apply X Gate
                </button>
                <button
                    className="quantum-action-button"
                    onClick={runQuantumSimulation}
                >
                    Run Full Simulation
                </button>
            </div>
        </div>
    );
};

export default QuantumPanel;