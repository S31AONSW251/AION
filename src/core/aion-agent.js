class AionAgent {
  constructor(showNotification, speak) {
    this.tasks = [];
    this.showNotification = showNotification;
    this.speak = speak;
  }

  /**
   * Creates a plan for a complex research query.
   * @param {string} query - The user's research request.
   * @returns {Promise<Array<object>>} A step-by-step plan.
   */
  async createPlan(query) {
    // In a real app, this would be an LLM call.
    this.showNotification(`Creating a research plan for "${query}"...`, 'info');
    await new Promise(r => setTimeout(r, 1000)); // Simulate planning
    return [
      { action: 'search', query: `${query} basics`, status: 'pending' },
      { action: 'search', query: `latest news about ${query}`, status: 'pending' },
      { action: 'summarize', query: 'Synthesize findings into a report.', status: 'pending' }
    ];
  }

  /**
   * Adds a proactive, background task for AION to monitor.
   * @param {object} task - Task details { id, description, condition, interval }
   */
  addProactiveTask(task) {
    const intervalId = setInterval(async () => {
      // Simulate checking the condition
      const conditionMet = await task.condition(); 
      if (conditionMet) {
        const message = `Proactive alert: Your task "${task.description}" has triggered.`;
        this.showNotification(message, 'success');
        this.speak(message);
        this.clearProactiveTask(task.id);
      }
    }, task.interval);

    this.tasks.push({ ...task, intervalId });
    this.showNotification(`New proactive task set: "${task.description}"`, 'info');
  }

  clearProactiveTask(taskId) {
    const task = this.tasks.find(t => t.id === taskId);
    if (task) {
      clearInterval(task.intervalId);
      this.tasks = this.tasks.filter(t => t.id !== taskId);
    }
  }
}

export const aionAgent = new AionAgent(() => {}, () => {}); // Will be initialized properly in the hook