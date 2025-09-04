// ========== NEURAL NETWORK MODULE ==========

/**
 * A utility class for matrix operations, used by the Neural Network.
 */
export class Matrix {
  /**
   * Constructs a Matrix.
   * @param {number} rows - Number of rows.
   * @param {number} cols - Number of columns.
   */
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    // Initialize matrix with zeros
    this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
  }

  /**
   * Fills the matrix with random values between -1 and 1.
   */
  randomize() {
    this.data = this.data.map(row => 
      row.map(() => Math.random() * 2 - 1)
    );
  }

  /**
   * Creates a Matrix from a 1D array (as a column vector).
   * @param {number[]} arr - The input array.
   * @returns {Matrix} A new Matrix instance.
   */
  static fromArray(arr) {
    return new Matrix(arr.length, 1).map((_, i) => arr[i]);
  }

  /**
   * Converts the Matrix to a 1D array.
   * @returns {number[]} The 1D array representation.
   */
  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  /**
   * Performs matrix multiplication (static method).
   * @param {Matrix} a - The first matrix.
   * @param {Matrix} b - The second matrix.
   * @returns {Matrix|undefined} The result of the multiplication, or undefined if dimensions are incompatible.
   */
  static multiply(a, b) {
    if (a.cols !== b.rows) {
      console.error('Columns of A must match rows of B for matrix multiplication.');
      return;
    }
    
    let result = new Matrix(a.rows, b.cols);
    
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    
    return result;
  }

  /**
   * Multiplies the matrix by a scalar or performs Hadamard product with another matrix.
   * @param {number|Matrix} n - The scalar or matrix to multiply by.
   * @returns {Matrix} The current Matrix instance for chaining.
   */
  multiply(n) {
    if (n instanceof Matrix) {
      // Hadamard product (element-wise multiplication)
      this.data = this.data.map((row, i) => 
        row.map((val, j) => val * n.data[i][j])
      );
    } else {
      // Scalar product
      this.data = this.data.map(row => 
        row.map(val => val * n)
      );
    }
    return this;
  }

  /**
   * Adds a scalar or another matrix to the current matrix.
   * @param {number|Matrix} n - The scalar or matrix to add.
   * @returns {Matrix} The current Matrix instance for chaining.
   */
  add(n) {
    if (n instanceof Matrix) {
      this.data = this.data.map((row, i) => 
        row.map((val, j) => val + n.data[i][j])
      );
    } else {
      this.data = this.data.map(row => 
        row.map(val => val + n)
      );
    }
    return this;
  }

  /**
   * Subtracts one matrix from another (static method).
   * @param {Matrix} a - The first matrix.
   * @param {Matrix} b - The second matrix.
   * @returns {Matrix} The result of the subtraction.
   */
  static subtract(a, b) {
    let result = new Matrix(a.rows, a.cols);
    result.data = a.data.map((row, i) => 
      row.map((val, j) => val - b.data[i][j])
    );
    return result;
  }

  /**
   * Transposes a matrix (static method).
   * @param {Matrix} matrix - The matrix to transpose.
   * @returns {Matrix} The transposed matrix.
   */
  static transpose(matrix) {
    let result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  /**
   * Applies a function to each element of the matrix (instance method).
   * @param {function(number, number, number): number} func - The function to apply (value, row, col).
   * @returns {Matrix} The current Matrix instance for chaining.
   */
  map(func) {
    this.data = this.data.map((row, i) => 
      row.map((val, j) => func(val, i, j))
    );
    return this;
  }

  /**
   * Applies a function to each element of a matrix (static method).
   * @param {Matrix} matrix - The matrix to map over.
   * @param {function(number, number, number): number} func - The function to apply (value, row, col).
   * @returns {Matrix} A new Matrix with the mapped values.
   */
  static map(matrix, func) {
    return new Matrix(matrix.rows, matrix.cols).map((_, i, j) => 
      func(matrix.data[i][j], i, j)
    );
  }

  /**
   * Prints the matrix data to the console (for debugging).
   * @returns {Matrix} The current Matrix instance for chaining.
   */
  print() {
    console.table(this.data);
    return this;
  }

  /**
   * Serializes the matrix to a JSON string.
   * @returns {string} The JSON string representation.
   */
  serialize() {
    return JSON.stringify(this);
  }

  /**
   * Deserializes a JSON string or object back into a Matrix instance.
   * @param {string|object} data - The JSON string or parsed object.
   * @returns {Matrix} The deserialized Matrix.
   */
  static deserialize(data) {
    if (typeof data === 'string') data = JSON.parse(data);
    let matrix = new Matrix(data.rows, data.cols);
    matrix.data = data.data;
    return matrix;
  }
}

/**
 * Represents a simple feedforward neural network.
 * In a "true AI," this would be a much more complex, potentially recurrent or transformer-based
 * network, capable of learning and adapting from vast amounts of data.
 */
export class NeuralNetwork {
  /**
   * Constructs a NeuralNetwork.
   * @param {number} inputNodes - Number of input neurons.
   * @param {number} hiddenNodes - Number of hidden layer neurons.
   * @param {number} outputNodes - Number of output neurons.
   */
  constructor(inputNodes, hiddenNodes, outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    // Initialize weights with random values
    this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes); // Weights from input to hidden
    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes); // Weights from hidden to output
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    // Initialize biases with random values
    this.bias_h = new Matrix(this.hiddenNodes, 1); // Bias for hidden layer
    this.bias_o = new Matrix(this.outputNodes, 1); // Bias for output layer
    this.bias_h.randomize();
    this.bias_o.randomize();

    this.learningRate = 0.1;
  }

  /**
   * The sigmoid activation function.
   * @param {number} x - The input value.
   * @returns {number} The sigmoid output.
   */
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * The derivative of the sigmoid function.
   * @param {number} y - The output of the sigmoid function.
   * @returns {number} The derivative of sigmoid.
   */
  dsigmoid(y) {
    // Note: y is already sigmoid(x)
    return y * (1 - y);
  }

  /**
   * Makes a prediction based on the input array.
   * @param {number[]} inputArray - The input data.
   * @returns {number[]} The predicted output array.
   */
  predict(inputArray) {
    // Convert input array to a Matrix
    let inputs = Matrix.fromArray(inputArray);

    // Calculate hidden layer outputs
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(this.sigmoid); // Apply activation function

    // Calculate output layer outputs
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(this.sigmoid); // Apply activation function

    return output.toArray();
  }

  /**
   * Trains the neural network using backpropagation.
   * @param {number[]} inputArray - The input data.
   * @param {number[]} targetArray - The expected target output data.
   */
  train(inputArray, targetArray) {
    // Feedforward pass to get outputs
    let inputs = Matrix.fromArray(inputArray);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(this.sigmoid);

    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(this.sigmoid);

    let targets = Matrix.fromArray(targetArray);

    // Calculate output errors (Error = Target - Output)
    let outputErrors = Matrix.subtract(targets, outputs);

    // Calculate output layer gradients
    let gradients = Matrix.map(outputs, this.dsigmoid); // Derivative of sigmoid
    gradients.multiply(outputErrors); // Element-wise multiplication by error
    gradients.multiply(this.learningRate); // Scale by learning rate

    // Calculate hidden to output weight deltas
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    // Adjust hidden to output weights and biases
    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradients);

    // Calculate hidden layer errors (backpropagate errors)
    let who_t = Matrix.transpose(this.weights_ho);
    let hiddenErrors = Matrix.multiply(who_t, outputErrors);

    // Calculate hidden layer gradients
    let hiddenGradient = Matrix.map(hidden, this.dsigmoid);
    hiddenGradient.multiply(hiddenErrors);
    hiddenGradient.multiply(this.learningRate);

    // Calculate input to hidden weight deltas
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hiddenGradient, inputs_T);

    // Adjust input to hidden weights and biases
    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hiddenGradient);
  }

  /**
   * Serializes the neural network to a JSON string.
   * @returns {string} The JSON string representation.
   */
  serialize() {
    return JSON.stringify(this);
  }

  /**
   * Deserializes a JSON string back into a NeuralNetwork instance.
   * @param {string} data - The JSON string.
   * @returns {NeuralNetwork} The deserialized NeuralNetwork.
   */
  static deserialize(data) {
    const parsed = JSON.parse(data);
    const nn = new NeuralNetwork(
      parsed.inputNodes, 
      parsed.hiddenNodes, 
      parsed.outputNodes
    );
    // Reconstruct Matrix objects from their serialized data
    nn.weights_ih = Matrix.deserialize(parsed.weights_ih);
    nn.weights_ho = Matrix.deserialize(parsed.weights_ho);
    nn.bias_h = Matrix.deserialize(parsed.bias_h);
    nn.bias_o = Matrix.deserialize(parsed.bias_o);
    nn.learningRate = parsed.learningRate;
    return nn;
  }
}