//This is just a basic neural network i try to build in C, to get just some understanding of neural network.


/*
Explanation of Key Sections:

    1. Neural Network Structure: We use a NeuralNetwork structure to hold weights and biases, making it easier to pass as a single parameter.

    2. Training and Testing Functions: The train function runs multiple epochs, and test outputs the neural network's predictions for each XOR input.

    3. Feedforward and Backpropagation Functions:
        a. Feedforward: Calculates outputs for the hidden and output layers.
        b. Backpropagation: Adjusts weights and biases based on errors to improve accuracy.
        */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES  2
#define HIDDEN_NODES 4
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.5
#define EPOCHS 100000 // Sufficiently high for convergence

// Sigmoid activation function and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Neural Network Structure
typedef struct {
    double weights_input_hidden[INPUT_NODES][HIDDEN_NODES];
    double weights_hidden_output[HIDDEN_NODES][OUTPUT_NODES];
    double hidden_bias[HIDDEN_NODES];
    double output_bias[OUTPUT_NODES];
} NeuralNetwork;

// Initialize weights and biases with random values
void initialize_weights(NeuralNetwork *nn) {
    srand(time(NULL));
    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES; j++)
            nn->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;

    for (int i = 0; i < HIDDEN_NODES; i++) {
        nn->hidden_bias[i] = ((double)rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < OUTPUT_NODES; j++)
            nn->weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < OUTPUT_NODES; i++)
        nn->output_bias[i] = ((double)rand() / RAND_MAX) - 0.5;
}

// Feedforward pass
void feedforward(NeuralNetwork *nn, double input[], double hidden[], double *output) {
    // Input to hidden layer
    for (int i = 0; i < HIDDEN_NODES; i++) {
        hidden[i] = 0.0;
        for (int j = 0; j < INPUT_NODES; j++)
            hidden[i] += input[j] * nn->weights_input_hidden[j][i];
        hidden[i] += nn->hidden_bias[i];
        hidden[i] = sigmoid(hidden[i]);
    }

    // Hidden to output layer
    *output = 0.0;
    for (int i = 0; i < HIDDEN_NODES; i++)
        *output += hidden[i] * nn->weights_hidden_output[i][0];
    *output += nn->output_bias[0];
    *output = sigmoid(*output);
}

// Backpropagation to adjust weights and biases
void backpropagate(NeuralNetwork *nn, double input[], double hidden[], double output, double target) {
    double output_error = target - output;
    double output_delta = output_error * sigmoid_derivative(output);

    // Calculate error for hidden layer
    double hidden_errors[HIDDEN_NODES];
    for (int i = 0; i < HIDDEN_NODES; i++)
        hidden_errors[i] = output_delta * nn->weights_hidden_output[i][0];

    // Update weights for hidden to output
    for (int i = 0; i < HIDDEN_NODES; i++)
        nn->weights_hidden_output[i][0] += LEARNING_RATE * output_delta * hidden[i];
    nn->output_bias[0] += LEARNING_RATE * output_delta;

    // Update weights for input to hidden
    for (int i = 0; i < HIDDEN_NODES; i++) {
        double hidden_delta = hidden_errors[i] * sigmoid_derivative(hidden[i]);
        for (int j = 0; j < INPUT_NODES; j++)
            nn->weights_input_hidden[j][i] += LEARNING_RATE * hidden_delta * input[j];
        nn->hidden_bias[i] += LEARNING_RATE * hidden_delta;
    }
}

// Training function
void train(NeuralNetwork *nn, double inputs[][INPUT_NODES], double targets[], int num_samples) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            double hidden[HIDDEN_NODES];
            double output;
            feedforward(nn, inputs[i], hidden, &output);
            backpropagate(nn, inputs[i], hidden, output, targets[i]);
        }
    }
}

// Testing function to evaluate the network on the XOR problem
void test(NeuralNetwork *nn, double inputs[][INPUT_NODES], int num_samples) {
    printf("Testing Results:\n");
    for (int i = 0; i < num_samples; i++) {
        double hidden[HIDDEN_NODES];
        double output;
        feedforward(nn, inputs[i], hidden, &output);
        printf("Input: (%.1f, %.1f) -> Output: %.5f\n", inputs[i][0], inputs[i][1], output);
    }
}

int main() {
    // XOR training data
    double inputs[4][INPUT_NODES] = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    double targets[4] = {0.0, 1.0, 1.0, 0.0};

    NeuralNetwork nn;
    initialize_weights(&nn);

    // Training
    train(&nn, inputs, targets, 4);

    // Testing
    test(&nn, inputs, 4);

    return 0;
}

/* This network includes:

    1. One hidden layer
    2. Sigmoid activation
    3. Initialization of weights and biases
    4. Feedforward and backpropagation steps 

    */