import React from 'react';

const Tabs = ({ activeTab, setActiveTab, settings, mathSolution, isMathQuery, userInput }) => {
  return (
    <div className="tab-container">
      <button
        className={`tab-button ${activeTab === "chat" ? 'active' : ''}`}
        onClick={() => setActiveTab("chat")}
      >
        Chat
      </button>
      <button
        className={`tab-button ${activeTab === "soul" ? 'active' : ''}`}
        onClick={() => setActiveTab("soul")}
      >
        Soul
      </button>
      <button
        className={`tab-button ${activeTab === "memories" ? 'active' : ''}`}
        onClick={() => setActiveTab("memories")}
      >
        Memories
      </button>
      {settings.enableWebSearch && (
        <button
          className={`tab-button ${activeTab === "search" ? 'active' : ''}`}
          onClick={() => setActiveTab("search")}
        >
          Search
        </button>
      )}
      {settings.enableMathSolving && (
        <button
          className={`tab-button ${activeTab === "math" ? 'active' : ''}`}
          onClick={() => {
            setActiveTab("math");
          }}
          disabled={!mathSolution && !isMathQuery(userInput)}
        >
          Math
        </button>
      )}
      {settings.enableQuantum && (
        <button
          className={`tab-button ${activeTab === "quantum" ? 'active' : ''}`}
          onClick={() => setActiveTab("quantum")}
        >
          Quantum
        </button>
      )}
      {settings.enableNeural && (
        <button
          className={`tab-button ${activeTab === "neural" ? 'active' : ''}`}
          onClick={() => setActiveTab("neural")}
        >
          Neural
        </button>
      )}
      {settings.enableCreativeGeneration && (
        <button
          className={`tab-button ${activeTab === "creative" ? 'active' : ''}`}
          onClick={() => setActiveTab("creative")}
        >
          Creative
        </button>
      )}
      {settings.goalTracking && (
        <button
          className={`tab-button ${activeTab === "goals" ? 'active' : ''}`}
          onClick={() => setActiveTab("goals")}
        >
          Goals
        </button>
      )}
      {settings.knowledgeBase && (
        <button
          className={`tab-button ${activeTab === "knowledge" ? 'active' : ''}`}
          onClick={() => setActiveTab("knowledge")}
        >
          Knowledge
        </button>
      )}
    </div>
  );
};

export default Tabs;