#ifndef helpers_h 
#define helpers_h 

int rand_int(int n);

double sigmoid(double x);

double sigmoid_derivative(double x);

double ReLU(double x);

double relu_derivative(double x);

double* softmax(double* input, int length);

double softmax_derivative(double x, double y);

double mean_square_error(double* x, double* y, int size);

double binary_cross_entropy(double x, double y, int size);

double cross_entropy_loss(double* x, double* y, int size);

void init_weights(double* weights, int size, int start, double mult_tendancy);

void init_biases(double* biases, int size, int start, double val);

void shuffle_array(int* arr, int n);

#endif