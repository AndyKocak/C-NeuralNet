#ifndef neural_h 
#define neural_h 

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
    double* raw_outputs;

    // Layer information
    char **activation_funcs;
    int* num_of_neurons;
    int* num_of_weights;
    int size;
    int total_neurons;
    int total_weights;
    int total_inputs;
} MLP;

void init_net(MLP *net, double* inputs, int input_size);

void add_forward_layer(MLP *(net), int units, char *activator,double w_mult_bias,double w_add_bias);

void add_out_layer(MLP *net, int units, char *activator, double w_mult_bias, double w_add_bias);

void feed_forward(MLP *net, double* inputs, int input_len);

#endif

