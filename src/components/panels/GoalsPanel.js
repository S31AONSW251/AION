import React from 'react';

const GoalsPanel = ({ soulState, setActiveTab }) => {
    return (
        <div className="goals-panel">
            <div className="goals-header">
                <h3>AION's Goals</h3>
                <button
                    className="back-button"
                    onClick={() => setActiveTab("chat")}
                >
                    <i className="icon-arrow-left"></i> Back to Chat
                </button>
            </div>
            <div className="goals-description">
                <p>
                    These are the goals I am currently tracking. You can ask me to "set a goal to..." or "update goal [description] to complete".
                </p>
            </div>
            <div className="goal-list">
                {soulState.goals.length > 0 ? (
                    soulState.goals.map((goal, index) => (
                        <div key={index} className={`goal-item ${goal.status}`}>
                            <div className="goal-description">{goal.description}</div>
                            <div className="goal-status">Status: {goal.status}</div>
                            <div className="goal-time">Set: {goal.timestamp}</div>
                        </div>
                    ))
                ) : (
                    <p>No goals set yet. Help me define my purpose!</p>
                )}
            </div>
        </div>
    );
};

export default GoalsPanel;