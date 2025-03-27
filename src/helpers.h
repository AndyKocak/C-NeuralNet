#ifndef helpers_h 
#define helpers_h 

double sigmoid(double x);

double sigmoid_derivative(double x);

double ReLU(double x);

double relu_derivative(double x);

double mean_square_error(double* x, double* y, int size);

double binary_cross_entropy(double x, double y, int size);

void init_weights(double* weights, int size, int start, double mult_tendancy, double add_tendancy);

void init_biases(double* biases, int size, int start);

#endif