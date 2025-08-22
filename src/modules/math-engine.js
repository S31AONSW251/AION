// src/modules/math-engine.js
import * as math from 'mathjs';

export class MathEngine {
  constructor() {
    this.math = math.create(math.all, { number: 'BigNumber', precision: 64 });
  }

  solve(expression) {
    try {
      const node = this.math.parse(expression);
      const simplified = this.math.simplify(node);
      const solution = node.evaluate();

      return {
        expression,
        simplified: simplified.toString(),
        solution: this.math.format(solution),
        steps: this.generateSteps(node, simplified),
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  generateSteps(before, after) {
    const steps = [];
    steps.push(`Original expression: ${before.toString()}`);
    steps.push(`Simplified form: ${after.toString()}`);
    steps.push(`Solution: ${this.math.format(after.evaluate())}`);
    return steps;
  }
}