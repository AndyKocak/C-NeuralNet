#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// Sigmoid Activation Function
double sigmoid(double x) {
    double e = 2.71828;
    return 1.0 / (1.0 + pow(e, -x));
}

// Derivative of Sigmoid
double sigmoid_derivative(double x) {
    return sigmoid(x) * sigmoid(1.0 - x);
}

// ReLU activation function
double ReLU(double x){
    return x > 0 ? x : 0;
}

// Derivative of ReLU
double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// MSE loss function
double mean_square_error(double* x, double* y, int size){
    double loss = 0.0;
    for (int i = 0; i < size; i++){
        loss += pow(x[i] - y[i], 2);
    }
    return loss / size; // Average loss
}

double binary_cross_entropy(double x, double y, int size){
    double loss = 0.0;

    double epsilon = 1e-15;
    x = fmax(epsilon, fmin(1 - epsilon, x));
    
    loss = - (y * log(x) + (1 - y) * log(1 - x));
    return loss;
}

// Function to initialize weights with small random values
void init_weights(double* weights, int size, int start, double mult_tendancy, double add_tendancy) {

    time_t current_time;
    current_time = time(NULL);

    srand(((unsigned int)current_time) + ((unsigned int) start));
    for (int i = 0; i < size; i++) {
        weights[i+start] = ((double) rand() / (RAND_MAX)) * mult_tendancy + add_tendancy; // Small random values
    }
}

// Function to initialize biases to zero
void init_biases(double* biases, int size, int start) {
    for (int i = 0; i < size; i++) {
        biases[i+start] = 0.0;
    }
}
