import * as math from 'mathjs';

// ========== MATH ENGINE ==========
/**
 * Provides mathematical computation capabilities using math.js.
 * Includes solving expressions, geometry, symbolic simplification, differentiation, and basic integration.
 */
export class MathEngine {
  constructor() {
    // Initialize math.js with BigNumber for precision
    this.math = math.create(math.all, {
      number: 'BigNumber',
      precision: 64
    });
  }

  /**
   * Solves a mathematical expression.
   * @param {string} expression - The expression to solve.
   * @returns {object} An object containing the expression, simplified form, solution, and steps, or an error.
   */
  solve(expression) {
    try {
      const node = this.math.parse(expression);
      const simplified = this.math.simplify(node);
      const solution = node.evaluate();
      
      return {
        expression,
        simplified: simplified.toString(),
        solution: this.math.format(solution), // Format solution for consistent output
        steps: this.generateSteps(node, simplified)
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  /**
   * Generates step-by-step explanation for a solved expression.
   * @param {math.Node} before - The parsed node before simplification.
   * @param {math.Node} after - The parsed node after simplification.
   * @returns {string[]} An array of explanation steps.
   */
  generateSteps(before, after) {
    const steps = [];
    steps.push(`Original expression: ${before.toString()}`);
    steps.push(`Simplified form: ${after.toString()}`);
    steps.push(`Solution: ${this.math.format(after.evaluate())}`);
    return steps;
  }

  /**
   * Solves common geometry problems.
   * @param {string} problem - The geometry problem description.
   * @returns {object} An object containing the problem, solution, formula, and steps, or an error.
   */
  solveGeometry(problem) {
    const lowerProblem = problem.toLowerCase();
    if (lowerProblem.includes('area of circle')) {
      const radiusMatch = lowerProblem.match(/radius\s*(\d+(\.\d+)?)/i);
      const radius = radiusMatch ? parseFloat(radiusMatch[1]) : null;
      
      if (radius === null || isNaN(radius)) {
        return { error: "Could not extract radius for circle area calculation." };
      }

      const area = this.math.multiply(this.math.pi, this.math.pow(radius, 2));
      return {
        problem,
        solution: this.math.format(area),
        formula: 'π × r²',
        steps: [
          `Identify radius: r = ${radius}`,
          `Apply area formula: π × r²`,
          `Calculate: π × ${radius}² = ${this.math.format(area)}`
        ]
      };
    } else if (lowerProblem.includes('circumference of circle')) {
      const radiusMatch = lowerProblem.match(/radius\s*(\d+(\.\d+)?)/i);
      const diameterMatch = lowerProblem.match(/diameter\s*(\d+(\.\d+)?)/i);
      
      let radius = null;
      if (radiusMatch) {
          radius = parseFloat(radiusMatch[1]);
      } else if (diameterMatch) {
          radius = parseFloat(diameterMatch[1]) / 2;
      }

      if (radius === null || isNaN(radius)) {
          return { error: "Could not extract radius or diameter for circle circumference calculation." };
      }

      const circumference = this.math.multiply(2, this.math.pi, radius);
      return {
          problem,
          solution: this.math.format(circumference),
          formula: '2 × π × r',
          steps: [
              `Identify radius: r = ${radius}`,
              `Apply circumference formula: 2 × π × r`,
              `Calculate: 2 × π × ${radius} = ${this.math.format(circumference)}`
          ]
      };
    } else if (lowerProblem.includes('area of rectangle')) {
      const lengthMatch = lowerProblem.match(/length\s*(\d+(\.\d+)?)/i);
      const widthMatch = lowerProblem.match(/width\s*(\d+(\.\d+)?)/i);
      
      const length = lengthMatch ? parseFloat(lengthMatch[1]) : null;
      const width = widthMatch ? parseFloat(widthMatch[1]) : null;

      if (length === null || isNaN(length) || width === null || isNaN(width)) {
          return { error: "Could not extract length and width for rectangle area calculation." };
      }

      const area = this.math.multiply(length, width);
      return {
          problem,
          solution: this.math.format(area),
          formula: 'length × width',
          steps: [
              `Identify length: l = ${length}`,
              `Identify width: w = ${width}`,
              `Apply area formula: l × w`,
              `Calculate: ${length} × ${width} = ${this.math.format(area)}`
          ]
      };
    }
    return { error: "Geometry problem not recognized" };
  }

  /**
   * Symbolically simplifies a mathematical expression.
   * @param {string} expression - The expression to simplify.
   * @returns {object} An object with the original, simplified expression, and steps.
   */
  simplifyExpression(expression) {
    try {
      const node = this.math.parse(expression);
      const simplified = this.math.simplify(node);
      return {
        expression,
        simplified: simplified.toString(),
        steps: [
          `Original: ${expression}`,
          `Simplified: ${simplified.toString()}`
        ]
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  /**
   * Performs basic symbolic differentiation of an expression with respect to a variable.
   * @param {string} expression - The expression to differentiate.
   * @param {string} variable - The variable to differentiate with respect to.
   * @returns {object} An object with the expression, variable, derivative, and steps.
   */
  differentiate(expression, variable) {
    try {
      const derivative = this.math.derivative(expression, variable);
      return {
        expression,
        variable,
        derivative: derivative.toString(),
        steps: [
          `Function: ${expression}`,
          `Variable: ${variable}`,
          `Derivative: ${derivative.toString()}`
        ]
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  /**
   * Performs very basic symbolic integration. Math.js has limited symbolic integration capabilities.
   * Only handles simple power rule for x^n.
   * @param {string} expression - The expression to integrate.
   * @param {string} variable - The variable to integrate with respect to.
   * @returns {object} An object with the expression, variable, integral, and steps, or an error.
   */
  integrate(expression, variable) {
    // Math.js does not have a direct symbolic integral function for complex cases.
    // This is a very basic implementation for demonstration (e.g., integral of x is (1/2)x^2).
    const lowerExpr = expression.toLowerCase().trim();
    if (lowerExpr === `${variable}`) {
      return {
        expression,
        variable,
        integral: `(1/2) * ${variable}^2 + C`,
        steps: [
          `Function: ${expression}`,
          `Variable: ${variable}`,
          `Integral (basic power rule): (1/2) * ${variable}^2 + C`
        ]
      };
    } else if (lowerExpr === `${variable}^2`) {
      return {
        expression,
        variable,
        integral: `(1/3) * ${variable}^3 + C`,
        steps: [
          `Function: ${expression}`,
          `Variable: ${variable}`,
          `Integral (basic power rule): (1/3) * ${variable}^3 + C`
        ]
      };
    }
    return { error: "Symbolic integration beyond basic power rule is not directly supported by Math.js in this setup." };
  }
}