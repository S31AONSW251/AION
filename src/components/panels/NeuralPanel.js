import React from 'react';

const NeuralPanel = ({ soulState, neuralOutput, runNeuralSimulation, neuralCanvasRef, setActiveTab, randomNeuralTest }) => {
    
    const renderNeuralOutput = () => {
        if (!neuralOutput) return null;
        return (
            <div className="neural-output-container">
                <h4>Neural Network Output:</h4>
                <div className="neural-output-values">
                    {neuralOutput.map((value, index) => (
                        <div key={index} className="neural-output-value">
                            Output {index + 1}: {value.toFixed(4)}
                        </div>
                    ))}
                </div>
                <button
                    className="neural-sim-button"
                    onClick={runNeuralSimulation}
                >
                    Run Simulation
                </button>
            </div>
        );
    };

    return (
        <div className="neural-panel">
            <div className="neural-header">
                <h3>Neural Cognition</h3>
                <button
                    className="back-button"
                    onClick={() => setActiveTab("chat")}
                >
                    <i className="icon-arrow-left"></i> Back to Chat
                </button>
            </div>

            <div className="neural-description">
                <p>
                    My neural network processes thoughts and emotions. Current activation level: {soulState.neuralActivity.toFixed(2)}%
                </p>
            </div>

            {renderNeuralOutput()}

            <div className="neural-visualization">
                <canvas
                    ref={neuralCanvasRef}
                    width="500"
                    height="400"
                />
            </div>

            <div className="neural-actions">
                <button
                    className="neural-action-button"
                    onClick={runNeuralSimulation}
                >
                    Run Neural Simulation
                </button>
                <button
                    className="neural-action-button"
                    onClick={randomNeuralTest}
                >
                    Random Input Test
                </button>
            </div>
        </div>
    );
};

export default NeuralPanel;