import React, { useState } from "react";

const SearchPanel = ({
  agentStatus,
  searchPlan,
  thoughtProcessLog,
  searchResults,
  isSearching,
  onNewSearch,
  suggestedQueries = [],
  searchSummary,
}) => {
  const [query, setQuery] = useState("");
  const [sortOption, setSortOption] = useState("relevance");

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      onNewSearch(query);
      setQuery('');
    }
  };

  const clusters = searchResults?.reduce((acc, result) => {
    const category = result.category || "General Findings";
    if (!acc[category]) acc[category] = [];
    acc[category].push(result);
    return acc;
  }, {}) || {};

  Object.keys(clusters).forEach((cat) => {
    clusters[cat].sort((a, b) => {
      if (sortOption === "relevance") return (b.score || 0) - (a.score || 0);
      if (sortOption === "date")
        return new Date(b.date || 0) - new Date(a.date || 0);
      return 0;
    });
  });

  const completedSteps = searchPlan.filter(step => step.status === 'completed').length;
  const totalSteps = searchPlan.length;
  const progress = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;

  return (
    <div className="search-panel">
      <div className="search-panel-header">
        <h3>Autonomous Web Agent</h3>
        <div className="agent-status-container">
          <span className={`status-indicator ${agentStatus}`}></span>
          <strong>Agent Status:</strong>
          <span className={`agent-status ${agentStatus}`}>{agentStatus}</span>
        </div>
      </div>

      <div className="search-input-container">
        <input
          type="text"
          value={query}
          placeholder="Ask AION to research a new topic..."
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSearching}
        />
        <button onClick={() => { onNewSearch(query); setQuery(''); }} disabled={!query || isSearching}>
          {isSearching ? 'Working...' : 'Initiate Research'}
        </button>
      </div>

      <div className="agent-dashboard">
        {/* Left Column: Plan & Thoughts */}
        <div className="agent-left-column">
          {searchPlan?.length > 0 && (
            <div className="agent-plan agent-card">
              <h4>Research Plan</h4>
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
              </div>
              <ol>
                {searchPlan.map((step, index) => (
                  <li key={index} className={`plan-step ${step.status}`}>
                    <span className="step-icon"></span>
                    <strong>{step.action}:</strong> {step.query}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {thoughtProcessLog?.length > 0 && (
            <div className="agent-thought-process agent-card">
              <h4>Thought Process Log</h4>
              <pre>{thoughtProcessLog.join("\n")}</pre>
            </div>
          )}
        </div>

        {/* Right Column: Results & Suggestions */}
        <div className="agent-right-column">
          {isSearching && !searchResults?.length && (
            <div className="search-loading agent-card">
              <div className="spinner"></div>
              <p>AION is executing its research plan...</p>
            </div>
          )}

          {searchSummary && (
            <div className="search-summary agent-card">
                <h4>Synthesized Summary</h4>
                <p>{searchSummary}</p>
            </div>
          )}

          <div className="search-results-container agent-card">
            <div className="results-header">
                <h4>
                    Knowledge Sources
                    <span className="result-count">({searchResults?.length || 0} found)</span>
                </h4>
                {searchResults?.length > 0 && (
                    <div className="sort-controls">
                        <label>Sort by:</label>
                        <select value={sortOption} onChange={(e) => setSortOption(e.target.value)}>
                            <option value="relevance">Relevance</option>
                            <option value="date">Date</option>
                        </select>
                    </div>
                )}
            </div>
            
            <div className="search-results">
              {Object.keys(clusters).length > 0 ? (
                Object.keys(clusters).map((category, i) => (
                  <div key={i} className="result-cluster">
                    <h5 className="cluster-title">{category}</h5>
                    {clusters[category].map((result, index) => (
                      <div key={index} className="search-result">
                        <h4>
                          <a href={result.url} target="_blank" rel="noopener noreferrer">
                            {result.title}
                          </a>
                        </h4>
                        <p className="result-snippet">{result.snippet}</p>
                        <div className="result-meta">
                          <span className="result-source">{result.source || "Unknown"}</span>
                           {result.date && (<span className="result-date">{new Date(result.date).toLocaleDateString()}</span>)}
                           {result.score !== undefined && (<span className="result-score">Relevance: {(result.score * 100).toFixed(0)}%</span>)}
                        </div>
                      </div>
                    ))}
                  </div>
                ))
              ) : (
                !isSearching && (
                    <div className="no-results">
                        <p>No research initiated. Ask AION something like "research the future of AI" to begin.</p>
                    </div>
                )
              )}
            </div>
          </div>
          
          {!isSearching && suggestedQueries?.length > 0 && (
            <div className="suggested-queries agent-card">
              <h4>Suggested Follow-up Research</h4>
              <ul>
                {suggestedQueries.map((s, i) => (
                  <li key={i}>
                    <button className="suggestion-button" onClick={() => onNewSearch(s)}>
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
export default SearchPanel;