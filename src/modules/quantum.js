// src/modules/quantum.js
export class QuantumState {
  constructor(numQubits = 1) {
    this.numQubits = numQubits;
    this.state = this.initializeState();
    this.history = [];
  }

  initializeState() {
    const size = Math.pow(2, this.numQubits);
    const state = new Array(size).fill({ real: 0, imag: 0 });
    state[0] = { real: 1, imag: 0 };
    return state;
  }

  applyGate(gateMatrix, targetQubit, controlQubit = null) {
    const size = Math.pow(2, this.numQubits);
    const newState = new Array(size);

    for (let i = 0; i < size; i++) {
      newState[i] = { real: 0, imag: 0 };
    }

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const amplitude = this.state[j];
        const gateValue = gateMatrix[i] && gateMatrix[i][j] ? gateMatrix[i][j] : { real: 0, imag: 0 };
        newState[i].real += amplitude.real * gateValue.real - amplitude.imag * gateValue.imag;
        newState[i].imag += amplitude.real * gateValue.imag + amplitude.imag * gateValue.real;
      }
    }

    this.history.push({
      gate: gateMatrix,
      target: targetQubit,
      control: controlQubit,
      before: [...this.state],
      after: [...newState],
    });

    this.state = newState;
    return this;
  }

  measure() {
    const probabilities = this.state.map(amp => Math.pow(amp.real, 2) + Math.pow(amp.imag, 2));
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalized = probabilities.map(p => p / sum);

    const rand = Math.random();
    let cumulative = 0;
    let result = 0;

    for (let i = 0; i < normalized.length; i++) {
      cumulative += normalized[i];
      if (rand < cumulative) {
        result = i;
        break;
      }
    }

    this.state = this.state.map((amp, index) => index === result ? { real: 1, imag: 0 } : { real: 0, imag: 0 });
    return result;
  }

  getProbabilities() {
    const probabilities = this.state.map(amp => Math.pow(amp.real, 2) + Math.pow(amp.imag, 2));
    const sum = probabilities.reduce((a, b) => a + b, 0);
    return probabilities.map(p => p / sum);
  }

  toString() {
    return this.state.map((amp, i) => {
      const stateLabel = i.toString(2).padStart(this.numQubits, '0').split('').reverse().join('');
      const prob = (Math.pow(amp.real, 2) + Math.pow(amp.imag, 2)).toFixed(4);
      return `|${stateLabel}⟩: ${amp.real.toFixed(4)} + ${amp.imag.toFixed(4)}i (${(prob * 100).toFixed(2)}%)`;
    }).join('\n');
  }
}

export const QuantumGates = {
  H: [
    [{ real: 1/Math.sqrt(2), imag: 0 }, { real: 1/Math.sqrt(2), imag: 0 }],
    [{ real: 1/Math.sqrt(2), imag: 0 }, { real: -1/Math.sqrt(2), imag: 0 }],
  ],
  X: [
    [{ real: 0, imag: 0 }, { real: 1, imag: 0 }],
    [{ real: 1, imag: 0 }, { real: 0, imag: 0 }],
  ],
  Y: [
    [{ real: 0, imag: 0 }, { real: 0, imag: -1 }],
    [{ real: 0, imag: 1 }, { real: 0, imag: 0 }],
  ],
  Z: [
    [{ real: 1, imag: 0 }, { real: 0, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: -1, imag: 0 }],
  ],
  CNOT: [
    [{ real: 1, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: 1, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 1, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 1, imag: 0 }, { real: 0, imag: 0 }],
  ],
};

export class QuantumSimulator {
  constructor() {
    this.circuits = {};
  }

  createCircuit(name, numQubits) {
    this.circuits[name] = new QuantumState(numQubits);
    return this.circuits[name];
  }

  getCircuit(name) {
    return this.circuits[name];
  }

  runAll() {
    return Object.keys(this.circuits).map(name => ({
      name,
      result: this.circuits[name].measure(),
      state: this.circuits[name].toString(),
    }));
  }
}