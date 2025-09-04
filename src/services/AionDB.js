import Dexie from 'dexie';

// Define the database schema for persistent memory
class AionDatabase extends Dexie {
  memories; // Table for long-term memories
  soulState; // Table to save the soul's state

  constructor() {
    super('AionDB');
    this.version(1).stores({
      memories: '++id, text, vector, timestamp, mood, sentiment',
      soulState: 'id, state',
    });
  }
}

export const db = new AionDatabase();