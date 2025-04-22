#ifndef mlpnet_h 
#define mlpnet_h 

// Struct definition of neural network a.k.a MLP (Multi Layer Perceptron)
typedef struct
{
    // Neuron values for calculating activation
    double* inputs;
    double* weights;
    double* biases;

    // Stored neuron values
    double* activations;
    double* raw_activations;
    double* outputs;

    // Layer information
    char **activation_funcs;
    int* num_of_neurons;
    int* num_of_weights;
    double loss;

    // Global network information
    int size;
    int total_neurons;
    int total_weights;
    int total_inputs;
    int max_layers;
    int max_neurons;
    int max_weights;
} MLP;

void init_mlp(MLP *net, int input_size);

void add_forward_layer(MLP *(net), int units, char *activator, double w_multiplier, double bias_val);

void feed_forward(MLP *net, double* inputs);

void backprop(MLP (*net), double learning_rate, double* target, char *error_func);

void train_from_source(MLP (*net), int input_size, int out_size, double lr, int epochs, int batch_size, int n_train, double input_data[batch_size][input_size], double target_data[batch_size][out_size], int display_thresh, char *error_func);

void free_mlp(MLP *net);

#endif

