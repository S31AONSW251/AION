import { db } from '../services/AionDB';

class AionMemory {
  /**
   * Simulates creating a vector embedding for a piece of text.
   * @param {string} text - The text to embed.
   * @returns {Promise<Array<number>>} A simulated vector.
   */
  async getVectorEmbedding(text) {
    const vector = Array(32).fill(0); // Increased dimensions for better simulation
    for (let i = 0; i < text.length; i++) {
      vector[i % 32] += text.charCodeAt(i);
    }
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => v / (magnitude || 1));
  }

  /**
   * Stores a new memory entry in the persistent database.
   * @param {object} interaction - The interaction object from conversation history.
   */
  async storeMemory(interaction) {
    const memoryText = `User: "${interaction.question}"\nAION: "${interaction.response}"`;
    const vector = await this.getVectorEmbedding(memoryText);
    const memoryRecord = {
      text: memoryText,
      vector: vector,
      timestamp: new Date(interaction.time),
      mood: interaction.mood,
      sentiment: interaction.sentiment
    };
    await db.memories.add(memoryRecord);
  }

  /**
   * Retrieves the most relevant memories from the database.
   * @param {string} queryText - The current user query.
   * @param {number} topK - The number of memories to retrieve.
   * @returns {Promise<Array<object>>} A list of relevant memories.
   */
  async retrieveRelevantMemories(queryText, topK = 3) {
    const memories = await db.memories.toArray();
    if (memories.length === 0) return [];

    const queryVector = await this.getVectorEmbedding(queryText);

    // Simulate cosine similarity
    const scores = memories.map(mem => {
      const dotProduct = mem.vector.reduce((sum, val, i) => sum + val * queryVector[i], 0);
      // Magnitudes are 1 due to normalization, so dotProduct is the similarity
      return { ...mem, score: dotProduct };
    });

    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, topK);
  }

  /**
   * Fetches all memories for the visualization.
   */
  async getAllMemoriesForVisualization() {
      return await db.memories.toArray();
  }
}

export const aionMemory = new AionMemory();