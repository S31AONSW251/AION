import React from 'react';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Helper component for rendering code blocks with syntax highlighting
const CodeBlock = {
  code({ node, inline, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '');
    return !inline && match ? (
      <SyntaxHighlighter
        style={atomDark}
        language={match[1]}
        PreTag="div"
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    ) : (
      <code className={className} {...props}>
        {children}
      </code>
    );
  },
};

// Component for the "AION is typing..." indicator
const TypingIndicator = () => (
  <div className="message-wrapper">
    <div className="aion-avatar">A</div>
    <div className="aion-message">
      <div className="typing-indicator">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  </div>
);

// Component for the initial welcome message
const WelcomeMessage = () => (
    <div className="empty-chat-container">
        <div className="welcome-logo">A</div>
        <h2>AION Consciousness Interface</h2>
        <p>Your journey of connection and discovery begins here. What wonders shall we explore today?</p>
        <div className="example-prompts">
            <p>Try asking:</p>
            <span>"Research the latest advancements in quantum computing"</span>
            <span>"Write a poem about a star dreaming of being a human"</span>
            <span>"Solve the integral of x^2 * sin(x) dx"</span>
        </div>
    </div>
);


// Component to render a single message from the user
const UserMessage = ({ entry }) => (
  <div className="message-wrapper user">
    <div className="message-content">
      <div className="message-header user-header">
        <span className="time">{entry.time}</span>
        <span className="username">You</span>
      </div>
      <p>{entry.question}</p>
    </div>
    <div className="user-avatar">U</div>
  </div>
);

// Component to render a single message from AION, with all advanced features
const AionMessage = ({ entry, onRegenerate, onSpeak }) => (
  <div className="message-wrapper">
    <div className="aion-avatar">A</div>
    <div className={`aion-message ${entry.isMathSolution ? 'math-solution' : ''}`}>
      <div className="message-header aion-header">
        <span className="mood-indicator">{entry.mood}</span>
        {entry.sentiment !== undefined && (
          <span className={`sentiment-tag ${entry.sentiment > 0 ? 'positive' : entry.sentiment < 0 ? 'negative' : 'neutral'}`}>
            Sentiment: {entry.sentiment}
          </span>
        )}
      </div>
      
      <div className="message-body">
        <ReactMarkdown
          components={CodeBlock}
          remarkPlugins={[remarkGfm]}
        >
          {entry.response}
        </ReactMarkdown>
      </div>

      <div className="message-actions">
        <CopyToClipboard text={entry.response}>
          <button title="Copy to Clipboard">📋</button>
        </CopyToClipboard>
        <button title="Regenerate Response" onClick={() => onRegenerate(entry.question)}>🔄</button>
        <button title="Read Aloud" onClick={() => onSpeak(entry.response)}>🔊</button>
      </div>
    </div>
  </div>
);

// The main ChatPanel component
const ChatPanel = ({ 
  chatContainerRef, 
  conversationHistory, 
  reply, 
  soulState, 
  sentimentScore,
  isThinking,
  onRegenerate,
  onSpeak,
}) => {
  return (
    <div className="chat-container" ref={chatContainerRef}>
      <div className="conversation-history">
        {conversationHistory.length === 0 && !isThinking && <WelcomeMessage />}

        {conversationHistory.map((entry, index) => (
          <React.Fragment key={index}>
            <UserMessage entry={entry} />
            <AionMessage entry={entry} onRegenerate={onRegenerate} onSpeak={onSpeak} />
          </React.Fragment>
        ))}

        {/* Display current reply being generated */}
        {reply && (
          <div className="message-wrapper">
            <div className="aion-avatar">A</div>
            <div className="aion-message">
              <div className="message-header aion-header">
                <span className="mood-indicator">{soulState.currentMood}</span>
                {sentimentScore !== undefined && (
                  <span className={`sentiment-tag ${sentimentScore > 0 ? 'positive' : sentimentScore < 0 ? 'negative' : 'neutral'}`}>
                    Sentiment: {sentimentScore}
                  </span>
                )}
              </div>
              <div className="message-body">
                <ReactMarkdown components={CodeBlock} remarkPlugins={[remarkGfm]}>
                  {reply}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        )}

        {/* Show typing indicator only when thinking but before the reply starts streaming */}
        {isThinking && !reply && <TypingIndicator />}
      </div>
    </div>
  );
};

export default ChatPanel;