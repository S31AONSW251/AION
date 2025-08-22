import React, { useRef, useEffect } from "react";
import { FaMicrophone, FaPaperPlane, FaDownload, FaBroom } from "react-icons/fa";
import { FaSpinner } from "react-icons/fa";

const ChatInterface = ({
  conversationHistory,
  userInput,
  setUserInput,
  isThinking,
  isListening,
  toggleSpeechRecognition,
  handleKeyDown,
  askAion,
  reply,
  searchResults,
  mathSolution,
  handleClearConversation,
  handleExportConversation,
}) => {
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [conversationHistory, reply, searchResults]);

  const renderMathSteps = () => {
    if (!mathSolution || !mathSolution.steps) return null;
    return (
      <div className="math-steps-container">
        <h4>Solution Steps:</h4>
        <ol className="math-steps-list">
          {mathSolution.steps.map((step, index) => (
            <li key={index} className="math-step">{step}</li>
          ))}
        </ol>
      </div>
    );
  };

  return (
    <>
      <div className="chat-container" ref={chatContainerRef}>
        <div className="conversation-history">
          {conversationHistory.map((entry, index) => (
            <div key={index} className={`conversation-entry ${entry.isMath ? "math-solution" : ""}`}>
              <div className="user-question">
                <span className="time">{entry.time}</span>
                <p><strong>You:</strong> {entry.question}</p>
              </div>
              <div className="aion-response">
                <span className="time">{entry.time}</span>
                <p>
                  <strong>AION:</strong> {entry.response}
                  {entry.mood && <span className={`mood-indicator ${entry.mood}`}>{entry.mood}</span>}
                  {entry.sentiment && <span className={`sentiment-tag ${entry.sentiment}`}>{entry.sentiment}</span>}
                </p>
                {entry.searchResults && (
                  <div className="search-results-container">
                    <h4>Web Search Results:</h4>
                    <ul>
                      {entry.searchResults.map((result, resIndex) => (
                        <li key={resIndex}>
                          <a href={result.url} target="_blank" rel="noopener noreferrer">{result.title}</a>
                          <p>{result.snippet}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {entry.isMath && renderMathSteps()}
              </div>
            </div>
          ))}
          {isThinking && (
            <div className="thinking-indicator">
              <FaSpinner className="spin" /> Thinking...
            </div>
          )}
        </div>
      </div>
      <div className="input-section">
        <div className="input-container">
          <textarea
            className="chat-input"
            placeholder="Type your question or command..."
            rows="1"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={handleKeyDown}
          ></textarea>
          <div className="input-actions">
            <button
              className={`icon-button mic-button ${isListening ? "active" : ""}`}
              onClick={toggleSpeechRecognition}
              aria-label="Toggle speech recognition"
            >
              <FaMicrophone />
            </button>
            <button
              className="icon-button"
              onClick={askAion}
              disabled={isThinking}
              aria-label="Send message"
            >
              <FaPaperPlane />
            </button>
          </div>
        </div>
        <div className="utility-buttons">
          <button className="utility-button" onClick={handleClearConversation}>
            <FaBroom /> Clear Chat
          </button>
          <button className="utility-button" onClick={handleExportConversation}>
            <FaDownload /> Export Chat
          </button>
        </div>
      </div>
    </>
  );
};

export default ChatInterface;