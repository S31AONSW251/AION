// src/modules/neural.js
import * as math from 'mathjs';

export class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes);
    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    this.bias_h = new Matrix(this.hiddenNodes, 1);
    this.bias_o = new Matrix(this.outputNodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();

    this.learningRate = 0.1;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  dsigmoid(y) {
    return y * (1 - y);
  }

  predict(inputArray) {
    let inputs = Matrix.fromArray(inputArray);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(this.sigmoid);
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(this.sigmoid);
    return output.toArray();
  }

  train(inputArray, targetArray) {
    let inputs = Matrix.fromArray(inputArray);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(this.sigmoid);
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(this.sigmoid);
    let targets = Matrix.fromArray(targetArray);

    let outputErrors = Matrix.subtract(targets, outputs);
    let gradients = Matrix.map(outputs, this.dsigmoid);
    gradients.multiply(outputErrors);
    gradients.multiply(this.learningRate);

    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradients);

    let who_t = Matrix.transpose(this.weights_ho);
    let hiddenErrors = Matrix.multiply(who_t, outputErrors);

    let hiddenGradient = Matrix.map(hidden, this.dsigmoid);
    hiddenGradient.multiply(hiddenErrors);
    hiddenGradient.multiply(this.learningRate);

    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hiddenGradient, inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hiddenGradient);
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    const parsed = JSON.parse(data);
    const nn = new NeuralNetwork(parsed.inputNodes, parsed.hiddenNodes, parsed.outputNodes);
    nn.weights_ih = Matrix.deserialize(parsed.weights_ih);
    nn.weights_ho = Matrix.deserialize(parsed.weights_ho);
    nn.bias_h = Matrix.deserialize(parsed.bias_h);
    nn.bias_o = Matrix.deserialize(parsed.bias_o);
    nn.learningRate = parsed.learningRate;
    return nn;
  }
}

export class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
  }

  randomize() {
    this.data = this.data.map(row => row.map(() => Math.random() * 2 - 1));
  }

  static fromArray(arr) {
    return new Matrix(arr.length, 1).map((_, i) => arr[i]);
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

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

  multiply(n) {
    if (n instanceof Matrix) {
      this.data = this.data.map((row, i) => row.map((val, j) => val * n.data[i][j]));
    } else {
      this.data = this.data.map(row => row.map(val => val * n));
    }
    return this;
  }

  add(n) {
    if (n instanceof Matrix) {
      this.data = this.data.map((row, i) => row.map((val, j) => val + n.data[i][j]));
    } else {
      this.data = this.data.map(row => row.map(val => val + n));
    }
    return this;
  }

  static subtract(a, b) {
    let result = new Matrix(a.rows, a.cols);
    result.data = a.data.map((row, i) => row.map((val, j) => val - b.data[i][j]));
    return result;
  }

  static transpose(matrix) {
    let result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  map(func) {
    this.data = this.data.map((row, i) => row.map((val, j) => func(val, i, j)));
    return this;
  }

  static map(matrix, func) {
    return new Matrix(matrix.rows, matrix.cols).map((_, i, j) => func(matrix.data[i][j], i, j));
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data === 'string') data = JSON.parse(data);
    let matrix = new Matrix(data.rows, data.cols);
    matrix.data = data.data;
    return matrix;
  }
}