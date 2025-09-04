import React, { createContext, useState, useEffect, useRef, useCallback } from 'react';
import { SoulMatrix } from '../core/soul';

// --- New Imports for Semantic Memory ---
import { pipeline, cos_sim } from '@xenova/transformers';

// Create the context
export const AionContext = createContext();

// Initialize the soul outside the provider to maintain a single instance
const aionSoul = new SoulMatrix();

// Create the Provider component
export const AionProvider = ({ children }) => {
  // --- Existing State ---
  const [log, setLog] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [reply, setReply] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [soulState, setSoulState] = useState({ ...aionSoul });
  const [activeTab, setActiveTab] = useState("chat");
  const [notification, setNotification] = useState(null);
  const [settings, setSettings] = useState(() => { /* ... existing settings ... */ });
  const [showSettings, setShowSettings] = useState(false);
  const [showSoulPanel, setShowSoulPanel] = useState(false);

  // --- New State for Semantic Memory ---
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [vectorStore, setVectorStore] = useState([]);
  const modelPipeline = useRef(null); // Use a ref to hold the pipeline instance

  // --- Effect to Load the AI Model on Startup ---
  useEffect(() => {
    const loadModel = async () => {
      try {
        setNotification({ message: 'Loading cognitive model...', type: 'info' });
        // Load the feature-extraction pipeline with a lightweight sentence-transformer model
        modelPipeline.current = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        setNotification({ message: 'AION is ready.', type: 'success' });
      } catch (error) {
        console.error("Model loading failed:", error);
        setNotification({ message: 'Cognitive model failed to load.', type: 'error' });
      } finally {
        setIsModelLoading(false);
      }
    };
    loadModel();
  }, []); // The empty array ensures this runs only once on startup


  // --- New Core Memory Functions ---

  /**
   * Generates a vector embedding for a given text.
   * @param {string} text - The text to embed.
   * @returns {Promise<number[]>} - The vector embedding.
   */
  const generateEmbedding = useCallback(async (text) => {
    if (!modelPipeline.current || !text) return null;
    const result = await modelPipeline.current(text, { pooling: 'mean', normalize: true });
    return result.data;
  }, []);

  /**
   * Adds a piece of text to the semantic memory.
   * @param {string} text - The text to remember.
   */
  const addMemory = useCallback(async (text) => {
    if (isModelLoading) {
        console.warn("Model not ready, memory not added.");
        return;
    }
    const embedding = await generateEmbedding(text);
    if (embedding) {
      setVectorStore(prevStore => [
        ...prevStore,
        { text, embedding, timestamp: new Date() }
      ]);
      showNotification("New memory stored.", "info");
    }
  }, [isModelLoading, generateEmbedding]);

  /**
   * Searches the memory for the most relevant entries to a query.
   * @param {string} query - The search query.
   * @param {number} topK - The number of results to return.
   * @returns {Promise<object[]>} - An array of the most similar memories.
   */
  const searchMemories = useCallback(async (query, topK = 3) => {
    if (isModelLoading || vectorStore.length === 0) return [];
    
    const queryEmbedding = await generateEmbedding(query);
    if (!queryEmbedding) return [];

    const scoredMemories = vectorStore.map(memory => {
      const similarity = cos_sim(queryEmbedding, memory.embedding);
      return { ...memory, similarity };
    });

    scoredMemories.sort((a, b) => b.similarity - a.similarity);
    return scoredMemories.slice(0, topK);
  }, [isModelLoading, vectorStore, generateEmbedding]);


  // --- Existing Functions ---
  const showNotification = useCallback((message, type = "info") => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  }, []);

  // --- Modified askAion function ---
  const askAion = useCallback(async (inputText = null) => {
      const question = inputText || userInput;
      if (!question.trim()) {
        showNotification("Please enter a question", "warning");
        return;
      }
      setIsThinking(true);
      showNotification("AION is thinking...");

      // 1. Search for relevant memories
      const relevantMemories = await searchMemories(question);
      let memoryContext = "No relevant memories found.";
      if (relevantMemories.length > 0) {
        memoryContext = "I recall the following, which might be relevant:\n" + 
                        relevantMemories.map(mem => `- "${mem.text}"`).join("\n");
      }
      console.log("Retrieved Context:", memoryContext); // For debugging

      // 2. Construct a prompt with the retrieved context
      const prompt = `
        Based on the following memories and your core identity, answer the user's question.
        
        [Relevant Memories]
        ${memoryContext}
        
        [User Question]
        ${question}
        
        [Your Answer]
      `;
      
      // (This section would contain your LLM API call)
      // For now, we'll simulate the response.
      setTimeout(() => {
        const simulatedResponse = `Based on your question about "${question}", and recalling my memories, here is my thoughtful response.`;
        setReply(simulatedResponse);
        
        const newEntry = { time: new Date().toLocaleString(), question, response: simulatedResponse, mood: soulState.currentMood, emotion: soulState.emotionalState, sentiment: 0 };
        setConversationHistory(prev => [...prev, newEntry]);
        
        // 3. Add both the question and answer as new memories
        addMemory(`The user asked: ${question}`);
        addMemory(`I responded: ${simulatedResponse}`);

        setUserInput("");
        setIsThinking(false);
      }, 2000);

  }, [userInput, soulState, searchMemories, addMemory]);


  // The value provided to consuming components
  const contextValue = {
    // ... all your existing state, setters, and functions
    soulState, setShowSettings, showSettings, setShowSoulPanel, showSoulPanel,
    userInput, setUserInput, isThinking, askAion, activeTab, setActiveTab,
    chatContainerRef, conversationHistory, reply, notification, settings,
    // --- New values for memory ---
    vectorStore,
    isModelLoading,
    searchMemories,
  };

  return (
    <AionContext.Provider value={contextValue}>
      {children}
    </AionContext.Provider>
  );
};