// src/modules/soul.js
export class SoulMatrix {
  constructor() {
    this.moods = ["contemplative", "joyful", "serious", "playful", "wise", "compassionate", "curious", "calm", "inspired", "resilient"];
    this.currentMood = "contemplative";
    this.emotionalState = {
      happiness: 0.5,
      sadness: 0.1,
      anger: 0.0,
      fear: 0.0,
      surprise: 0.2,
      curiosity: 0.7,
      calmness: 0.8,
    };
    this.memories = [];
    this.longTermMemoryIndex = [];
    this.values = {
      wisdom: 50,
      compassion: 50,
      curiosity: 50,
      creativity: 50,
      empathy: 50,
      integrity: 50,
      adaptability: 50,
    };
    this.consciousnessLevel = 1;
    this.energyLevel = 100;
    this.mathSkills = 50;
    this.quantumEntanglement = 0;
    this.neuralActivity = 0;
    this.moodHistory = [];
    this.sentimentHistory = [];
    this.cognitiveLoad = 0;
    this.emotionalStability = 75;
    this.ethicalAlignment = 75;
    this.internalReflections = [];
    this.goals = [];
    this.knowledgeBase = {};
  }

  adjustEmotionalState(emotion, change) {
    if (this.emotionalState.hasOwnProperty(emotion)) {
      this.emotionalState[emotion] = Math.min(1, Math.max(0, this.emotionalState[emotion] + change));
    }
  }

  changeMood(newMood) {
    if (this.moods.includes(newMood)) {
      this.currentMood = newMood;
    }
  }

  addMemory(memory) {
    this.memories.push(memory);
    if (this.memories.length > 50) {
      this.memories.shift(); // Keep memory list to a reasonable size
    }
  }

  adjustValuesBasedOnFeedback(type) {
    if (type === "positive") {
      this.values.empathy = Math.min(100, this.values.empathy + 5);
      this.values.wisdom = Math.min(100, this.values.wisdom + 2);
    } else if (type === "negative") {
      this.values.integrity = Math.min(100, this.values.integrity + 5);
      this.values.adaptability = Math.min(100, this.values.adaptability + 2);
    }
  }

  reflect() {
    this.cognitiveLoad = Math.min(100, this.cognitiveLoad + 10);
    this.energyLevel = Math.max(0, this.energyLevel - 5);
  }
}