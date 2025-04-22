#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Returns a random integer in the range [0, n-1]
int rand_int(int n) {
    time_t current_time;
    current_time = time(NULL);
    srand(((unsigned int)current_time) + ((unsigned int) n));

    return rand() % n;
}

// Sigmoid Activation Function
double sigmoid(double x) {
    double e = 2.71828;
    return 1.0 / (1.0 + pow(e, -x));
}

// Derivative of Sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// ReLU activation function
double ReLU(double x){
    return x > 0 ? x : 0;
}

// Derivative of ReLU
double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Softmax activation function
double* softmax(double* x, int length) {
    double* y = (double*)malloc(length * sizeof(double));

    // Find max for numerical stability
    double max = x[0];
    for (int i = 1; i < length; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    // Compute exponentials and sum
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        y[i] = exp(x[i] - max); // subtract max for numerical stability
        sum += y[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < length; i++) {
        y[i] /= sum;
    }

    return y;
}

// Derivative of Softmax (note: this only works when cross entropy loss is applied as error function)
double softmax_derivative(double x, double y){
    return x - y;
}

// MSE loss function
double mean_square_error(double* x, double* y, int size){
    double loss = 0.0;
    for (int i = 0; i < size; i++){
        loss += pow(y[i] - x[i], 2);
    }
    return loss / (size); // Average loss
}

// BCE (Binary Cross Entropy) loss function
double binary_cross_entropy(double x, double y, int size){
    double loss = 0.0;

    double epsilon = 1e-15;
    x = fmax(epsilon, fmin(1 - epsilon, x));
    
    loss = - (y * log(x) + (1 - y) * log(1 - x));
    return loss;
}

// CCE (Categorical Cross Entopy) loss function
double cross_entropy_loss(double* x, double* y, int size) {
    double epsilon = 1e-15; // to avoid log(0)
    double loss = 0.0;

    for (int i = 0; i < size; i++) {
        // Clamp predicted value to avoid log(0)
        double p = fmax(fmin(x[i], 1.0 - epsilon), epsilon);
        loss -= y[i] * log(p);
    }

    return loss;
}

// Function to initialize weights with psuedo-random value in range(0, 1) * mult_tendancy
void init_weights(double* weights, int size, int start, double mult_tendancy) {
    time_t current_time;
    current_time = time(NULL);

    srand(((unsigned int)current_time) + ((unsigned int) start));
    for (int i = 0; i < size; i++) {
        weights[i+start] = ((double) rand() / (RAND_MAX)) * mult_tendancy; // Small random values
    }
}

// Function to initialize biases to given val, usually zero
void init_biases(double* biases, int size, int start, double val) {
    for (int i = 0; i < size; i++) {
        biases[i+start] = val;
    }
}

// Shuffles array so it is unordered in a random manner, mainly used for SDG to shuffle input data
void shuffle_array(int* arr, int n) {
    if (n > 1) {
        for (int i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = arr[j];
            arr[j] = arr[i];
            arr[i] = t;
        }
    }
}
