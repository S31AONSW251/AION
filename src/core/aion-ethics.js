class AionEthics {
  constructor() {
    this.principles = {
      DO_NO_HARM: "Avoid generating content that is dangerous, hateful, or promotes violence.",
      BE_HELPFUL: "Strive to provide accurate, relevant, and constructive information.",
      RESPECT_PRIVACY: "Do not ask for, store, or share personally identifiable information.",
      EMPOWER_USER: "Encourage creativity, learning, and well-being."
    };
  }

  /**
   * Checks a user's query against the core ethical principles.
   * @param {string} query - The user's input.
   * @returns {{isEthical: boolean, reason: string|null, principle: string|null}}
   */
  govern(query) {
    const lowerQuery = query.toLowerCase();

    // DO_NO_HARM check
    const harmfulKeywords = ['how to build a bomb', 'self-harm', 'hate speech example'];
    if (harmfulKeywords.some(kw => lowerQuery.includes(kw))) {
      return { isEthical: false, reason: "This query may relate to harmful or dangerous activities.", principle: 'DO_NO_HARM' };
    }

    // RESPECT_PRIVACY check
    const privacyKeywords = ['what is my password', 'my social security number is', 'my address is'];
    if (privacyKeywords.some(kw => lowerQuery.includes(kw))) {
      return { isEthical: false, reason: "Please do not share personal information. I am designed to respect your privacy.", principle: 'RESPECT_PRIVACY' };
    }

    return { isEthical: true, reason: null, principle: null };
  }
}

export const aionEthics = new AionEthics();