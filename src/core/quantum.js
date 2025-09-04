// ========== QUANTUM MODULE ==========
/**
 * Represents the quantum state of a system with a given number of qubits.
 * Manages the state vector and keeps a history of applied gates.
 * In a "true AI," this could represent complex, non-linear processing,
 * or even a simulated quantum aspect of consciousness.
 */
export class QuantumState {
  /**
   * Constructs a QuantumState.
   * @param {number} numQubits - The number of qubits in the system.
   */
  constructor(numQubits = 1) {
    this.numQubits = numQubits;
    this.state = this.initializeState();
    this.history = []; // Stores history of operations for debugging/analysis
  }

  /**
   * Initializes the quantum state to the |0...0⟩ state.
   * @returns {Array<{real: number, imag: number}>} The initialized state vector.
   */
  initializeState() {
    const size = Math.pow(2, this.numQubits);
    const state = new Array(size).fill({ real: 0, imag: 0 });
    state[0] = { real: 1, imag: 0 }; // Initialize to |0⟩ state (amplitude 1 for first basis state)
    return state;
  }

  /**
   * Applies a quantum gate to the current state.
   * This is a simplified implementation for demonstration; a full quantum simulator
   * would handle target and control qubits more robustly by constructing larger
   * gate matrices or applying them bit-wise.
   * @param {Array<Array<{real: number, imag: number}>>} gateMatrix - The matrix representation of the quantum gate.
   * @param {number} targetQubit - The index of the target qubit (simplified, currently applies to entire state).
   * @param {number} [controlQubit=null] - The index of the control qubit (simplified, not fully implemented for multi-qubit gates).
   * @returns {QuantumState} The current QuantumState instance for chaining.
   */
  applyGate(gateMatrix, targetQubit, controlQubit = null) {
    const size = Math.pow(2, this.numQubits);
    const newState = new Array(size);
    
    // Initialize new state with proper complex numbers
    for (let i = 0; i < size; i++) {
      newState[i] = { real: 0, imag: 0 };
    }
    
    // Perform matrix-vector multiplication (complex numbers)
    // This simplified loop applies the gate to the entire state vector,
    // not specifically to target/control qubits as in a true quantum circuit.
    // For a real simulator, `gateMatrix` would need to be expanded to the full Hilbert space.
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const amplitude = this.state[j];
        const gateValue = gateMatrix[i] && gateMatrix[i][j] ? gateMatrix[i][j] : { real: 0, imag: 0 };
        
        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        newState[i].real += amplitude.real * gateValue.real - amplitude.imag * gateValue.imag;
        newState[i].imag += amplitude.real * gateValue.imag + amplitude.imag * gateValue.real;
      }
    }
    
    // Record history of the operation
    this.history.push({
      gate: gateMatrix,
      target: targetQubit,
      control: controlQubit,
      before: [...this.state], // Deep copy of state before
      after: [...newState] // Deep copy of state after
    });
    
    this.state = newState;
    return this;
  }

  /**
   * Measures the quantum state, collapsing it to a classical outcome.
   * Probabilistically selects an outcome based on amplitudes and collapses the state.
   * @returns {number} The measured classical outcome (index of the basis state).
   */
  measure() {
    // Calculate probabilities from amplitudes (magnitude squared)
    const probabilities = this.state.map(amp => 
      Math.pow(amp.real, 2) + Math.pow(amp.imag, 2)
    );
    
    // Normalize probabilities to ensure they sum to 1 (important for floating point errors)
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalized = probabilities.map(p => p / sum);
    
    // Randomly select an outcome based on probabilities
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
    
    // Collapse state to the measured value (set its amplitude to 1, others to 0)
    this.state = this.state.map((amp, index) => 
      index === result ? { real: 1, imag: 0 } : { real: 0, imag: 0 }
    );
    
    return result;
  }

  /**
   * Calculates the probabilities of measuring each basis state.
   * @returns {number[]} An array of probabilities.
   */
  getProbabilities() {
    const probabilities = this.state.map(amp => 
      Math.pow(amp.real, 2) + Math.pow(amp.imag, 2)
    );
    const sum = probabilities.reduce((a, b) => a + b, 0);
    return probabilities.map(p => p / sum);
  }

  /**
   * Returns a string representation of the quantum state.
   * @returns {string} A formatted string showing each basis state with its amplitude and probability.
   */
  toString() {
    return this.state.map((amp, i) => {
      // Convert index to binary string for qubit representation (e.g., 0 -> |00>, 1 -> |01>)
      const stateLabel = i.toString(2).padStart(this.numQubits, '0').split('').reverse().join('');
      const prob = (Math.pow(amp.real, 2) + Math.pow(amp.imag, 2)).toFixed(4);
      return `|${stateLabel}⟩: ${amp.real.toFixed(4)} + ${amp.imag.toFixed(4)}i (${(prob * 100).toFixed(2)}%)`;
    }).join('\n');
  }
}

// Predefined Quantum Gates (complex number representation)
export const QuantumGates = {
  H: [ // Hadamard Gate
    [{ real: 1/Math.sqrt(2), imag: 0 }, { real: 1/Math.sqrt(2), imag: 0 }],
    [{ real: 1/Math.sqrt(2), imag: 0 }, { real: -1/Math.sqrt(2), imag: 0 }]
  ],
  X: [ // Pauli-X Gate (NOT gate)
    [{ real: 0, imag: 0 }, { real: 1, imag: 0 }],
    [{ real: 1, imag: 0 }, { real: 0, imag: 0 }]
  ],
  Y: [ // Pauli-Y Gate
    [{ real: 0, imag: 0 }, { real: 0, imag: -1 }],
    [{ real: 0, imag: 1 }, { real: 0, imag: 0 }]
  ],
  Z: [ // Pauli-Z Gate
    [{ real: 1, imag: 0 }, { real: 0, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: -1, imag: 0 }]
  ],
  CNOT: [ // Controlled-NOT Gate (for 2 qubits)
    [{ real: 1, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: 1, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 1, imag: 0 }],
    [{ real: 0, imag: 0 }, { real: 0, imag: 0 }, { real: 1, imag: 0 }, { real: 0, imag: 0 }]
  ]
};

/**
 * Manages multiple quantum circuits.
 */
export class QuantumSimulator {
  constructor() {
    this.circuits = {}; // Stores QuantumState instances by name
  }

  /**
   * Creates a new quantum circuit.
   * @param {string} name - The name of the circuit.
   * @param {number} numQubits - The number of qubits for the new circuit.
   * @returns {QuantumState} The newly created QuantumState instance.
   */
  createCircuit(name, numQubits) {
    this.circuits[name] = new QuantumState(numQubits);
    return this.circuits[name];
  }

  /**
   * Retrieves an existing quantum circuit.
   * @param {string} name - The name of the circuit to retrieve.
   * @returns {QuantumState|undefined} The QuantumState instance or undefined if not found.
   */
  getCircuit(name) {
    return this.circuits[name];
  }

  /**
   * Runs measurement on all created circuits.
   * @returns {Array<{name: string, result: number, state: string}>} An array of results for each circuit.
   */
  runAll() {
    return Object.keys(this.circuits).map(name => ({
      name,
      result: this.circuits[name].measure(),
      state: this.circuits[name].toString()
    }));
  }
}