import React from 'react';

const KnowledgePanel = ({ soulState, setActiveTab }) => {
    return (
        <div className="knowledge-panel">
            <div className="knowledge-header">
                <h3>AION's Knowledge Base</h3>
                <button
                    className="back-button"
                    onClick={() => setActiveTab("chat")}
                >
                    <i className="icon-arrow-left"></i> Back to Chat
                </button>
            </div>
            <div className="knowledge-description">
                <p>
                    This is what I have learned and stored. You can ask me to "remember that [fact]" or "what do you know about [topic]".
                </p>
            </div>
            <div className="knowledge-list">
                {Object.keys(soulState.knowledgeBase).length > 0 ? (
                    Object.entries(soulState.knowledgeBase).map(([key, data], index) => (
                        <div key={index} className="knowledge-item">
                            <div className="knowledge-key"><strong>{key}:</strong></div>
                            <div className="knowledge-value">{data.value}</div>
                            <div className="knowledge-time">Learned: {data.timestamp}</div>
                        </div>
                    ))
                ) : (
                    <p>My knowledge base is empty. Teach me something new!</p>
                )}
            </div>
        </div>
    );
};

export default KnowledgePanel;