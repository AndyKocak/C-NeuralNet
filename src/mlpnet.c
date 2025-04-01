#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlpnet.h"
#include "helpers.h"

// construct network struct with input layer
void init_mlp(MLP *net, int input_size){

    // Init array that holds input values and wrappers that hold total layer count (size) and total neuron count.
    (*net).inputs = (double*)malloc(input_size * sizeof(double));
    (*net).size = 0;
    (*net).total_neurons = 0;
    (*net).total_weights = 0;
    (*net).total_inputs = input_size;

    // Init array that stores how many neurons and weights in each layer
    (*net).num_of_neurons = (int*)malloc(1 * sizeof(int));
    (*net).num_of_weights = (int*)malloc(1 * sizeof(int));
    
    // Init all forward/backward pass variable arrays
    (*net).weights = (double*)malloc(1 * sizeof(double));
    (*net).biases = (double*)malloc(1 * sizeof(double));
    (*net).activations = (double*)malloc(1 * sizeof(double));
    (*net).raw_activations = (double*)malloc(1 * sizeof(double));
    (*net).outputs = (double*)malloc(1 * sizeof(double));

    // Init string array that stores activation functions for each layer
    (*net).activation_funcs = (char **)malloc(1 * sizeof(char *));

}

// add new forward pass layer to network 
void add_forward_layer(MLP *net, int units, char *activator, double w_multiplier, double bias_val){

    // Get current length of network
    int length = (*net).size;

    // Change input_size depending on whether this is initial hidden layer or not.
    int input_size = ((*net).size == 0) ? (*net).total_inputs : (*net).num_of_neurons[((*net).size - 1)];

    // Reallocate array that stores # of neurons in each layer
    (*net).num_of_neurons = realloc((*net).num_of_neurons,(length+1) * sizeof(int));
    (*net).num_of_neurons[(length)] = units;

    // Reallocate string array to add activator value
    (*net).activation_funcs = realloc((*net).activation_funcs, (length+1) * sizeof(char *));
    (*net).activation_funcs[(length)] = (char *)malloc(50 * sizeof(char));
    strcpy((*net).activation_funcs[(length)], activator);

    // Reallocate array that stores # weight for each layer
    (*net).num_of_weights = realloc((*net).num_of_weights, (length+1) * sizeof(int));
    (*net).num_of_weights[(length)] = units * input_size;

    // Reallocate weigth array and generate/assign random values using helper function
    length = (*net).total_weights + (units * input_size);
    (*net).weights = realloc((*net).weights,(length + 1) * sizeof(double));
    init_weights((*net).weights, (units * input_size), (*net).total_weights, w_multiplier);

    // Reallocate activations and biases arrays to store each neurons activation and bias values
    length = (*net).total_neurons + units;
    (*net).activations = realloc((*net).activations, length * sizeof(double));
    (*net).raw_activations = realloc((*net).activations, length * sizeof(double));
    (*net).biases = realloc((*net).biases, length * sizeof(double));

    // Call helper function to generate initial bias values
    init_biases((*net).biases, units, length, bias_val);

    // Increment counters for network sizes
    (*net).size = (*net).size + 1;
    (*net).total_neurons = (*net).total_neurons + units;
    (*net).total_weights += (units * input_size);

}

// feed forward input data to get output on constructed MLP
void feed_forward(MLP *net, double* inputs){
    
    // Change input array of MLP to new input arg
    (*net).inputs = inputs;

    // Reset trackers to properly traverse arrays
    int total_neurons = 0;
    int total_weights = 0;

    // Init output array
    int out_size = (*net).num_of_neurons[(*net).size - 1];
    (*net).outputs = realloc((*net).outputs, out_size * sizeof(double));

    // Iterate through each layer starting from input layer
    for (int n = 0; n < (*net).size; n++){

        // Get input size for each layer
        int input_size = (n == 0) ? (*net).total_inputs : (*net).num_of_neurons[n-1];
        
        // Init layer specific info
        int units = (*net).num_of_neurons[n];
        char *activator = (*net).activation_funcs[n];

        // If in layer right after input
        if (n == 0){
            // Apply recursive formula for perceptron
            for (int i = 0; i < units; i++){
                (*net).raw_activations[(i)] = 0;
                for(int j = 0; j < input_size; j++){
                    (*net).raw_activations[(i)] += (*net).inputs[(j)] * (*net).weights[i * input_size + j];
                }

                // Apply bias to node
                (*net).activations[(i)] = (*net).raw_activations[(i)];
                (*net).activations[(i)] += (*net).biases[(i)];

                // Conditionally apply activation based on what was given as arg
                if (strcmp(activator, "ReLU") == 0){
                    (*net).activations[i] = ReLU((*net).activations[(i)]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    (*net).activations[i] = sigmoid((*net).activations[(i)]);
                }   
                
                // If only 1 layer exists in network add activations to outputs array
                if (n == ((*net).size - 1)){
                    (*net).outputs[i] = (*net).activations[i];
                }
            }
        }

        // If in hidden layer or output layer
        else{
            for (int i = 0; i < units; i++){
                // Calculate index of currently accessed node and the row adress of its weight matrix
                int cur_activation_index = i + total_neurons;
                int cur_weight_row_index = (i * input_size) + total_weights;

                // The index at which last layers activations started
                int prev_activation_start = (total_neurons - input_size);

                // Reinit activation values
                (*net).raw_activations[cur_activation_index] = 0;
                for(int j = 0; j < input_size; j++){
                    (*net).raw_activations[cur_activation_index] += (*net).raw_activations[j + prev_activation_start] * (*net).weights[j + cur_weight_row_index];
                }

                (*net).activations[(cur_activation_index)] = (*net).raw_activations[cur_activation_index];
                // Add bias to activation
                (*net).activations[cur_activation_index] += (*net).biases[cur_activation_index];
                                
                
                // Conditionally apply activation based on what was given as arg
                if (strcmp(activator, "ReLU") == 0){
                    (*net).activations[cur_activation_index] = ReLU((*net).activations[cur_activation_index]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    (*net).activations[cur_activation_index] = sigmoid((*net).activations[cur_activation_index]);
                }

                // If on output layer add activations to outputs array
                if (n == ((*net).size - 1)){
                    (*net).outputs[i] = (*net).activations[cur_activation_index];
                }
            }
        }

        // Update total navigated neuron tracker
        total_neurons += (*net).num_of_neurons[n];

        // Update total weight tracker
        total_weights += (units * input_size);
    }
}

// THIS ONE CAUSES PROBLEMS!!!
void backprop(MLP (*net), double learning_rate, double* target, char *error_func){
    double* weight_delta = (double*)malloc(((*net).total_weights + 1) * sizeof(double));

    // size and weight counters
    int cur_size = (*net).size;
    int cur_neurons = (*net).total_neurons;
    int cur_weights = (*net).total_weights;

    double cur_delta = 0;
    double error = 0;

    while (cur_size > 0){

        // Get input size
        int input_size = (cur_size == 1) ? (*net).total_inputs : (*net).num_of_neurons[cur_size-2];

        // starting search index for activations in this layer
        int a_start = cur_neurons - (*net).num_of_neurons[cur_size - 1];

        // local end index for activations in this layer
        int a_end = (*net).num_of_neurons[cur_size - 1];

        // starting search index for weights
        int w_start = cur_weights - (*net).num_of_weights[cur_size - 1];

        // weight per neuron on this layer
        int w_per_n = (*net).num_of_weights[cur_size-1] / (*net).num_of_neurons[cur_size-1];

        // current layer activation function
        char *activator = (*net).activation_funcs[cur_size - 1];

        // if at output layer
        if (cur_size == (*net).size){
            for (int i = 0; i < a_end; i++){

                if (strcmp(error_func, "MSE") == 0){
                    error = mean_square_error((*net).outputs, target, (*net).num_of_neurons[cur_size - 1]);
                    (*net).loss = error;
                }
                else if (strcmp(error_func, "BCE") == 0){
                    error = binary_cross_entropy((*net).outputs[i], target[i], (*net).num_of_neurons[cur_size - 1]);
                    (*net).loss = error;
                }

                double dRaw = 0.0;
                if (strcmp(activator, "ReLU") == 0){
                    dRaw = relu_derivative((*net).activations[a_start + i]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    dRaw = sigmoid_derivative((*net).activations[a_start + i]);
                }

                // For the purposes of this framework I will be using (z - y) as loss term for output layer
                cur_delta = ((*net).outputs[i] - target[i]);

                // Update weights connecting to neuron using gradient descent. 
                for(int j = 0; j < w_per_n; j++){
                    // Thanks to chain rule, change in weight for output layer can be represented as 𝛿U𝑗 = (𝑧 − 𝑦) *  U𝑗, where U𝑗 is the activatio input coming from connected weight.
                    (*net).weights[j + w_start + (i * w_per_n)] = (*net).weights[j + w_start + (i * w_per_n)] - (cur_delta * (*net).activations[j + (a_start - input_size)] * learning_rate);
                    weight_delta[j + w_start + (i * w_per_n)] = (cur_delta * dRaw);
                }

                // Update bias based on loss
                (*net).biases[a_start + i] = (*net).biases[a_start + i] - (dRaw * cur_delta * learning_rate);
                
            }
        }

        // if at input layer: (Same methodology as hidden layer except use input data as previous layer info)
        //                    (check hidden layer for more notes on how this works)
        else if (cur_size == 1){

            // w_per_n and w_start from previous execution
            int prev_w_per_n = (*net).num_of_weights[cur_size] / (*net).num_of_neurons[cur_size];
            int prev_w_start = cur_weights;

            cur_delta = 0;

            // So now to find delta (𝛿) at some point j, we do: (∑ Wjl * 𝛿l) for all weights connected from node j to some arbitrary node l at the next layer.
            for(int k = 0; k < (*net).num_of_neurons[cur_size]; k++){
                for(int l = 0; l < prev_w_per_n; l++){
                    cur_delta += (weight_delta[l + prev_w_start + (k * prev_w_per_n)]);
                }
            }

            // Debug code for cur_delta here

            // Start traversing current layer nodes
            for (int i = 0; i < a_end; i++){

                // Next to find weight delta we apply the derivative of the activation function of current layer to current nodes raw values
                double dRaw = 0;
                if (strcmp(activator, "ReLU") == 0){
                    dRaw = relu_derivative((*net).activations[a_start + i]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    dRaw = sigmoid_derivative((*net).activations[a_start + i]);
                }

                for(int j = 0; j < w_per_n; j++){
                    // Update delta storage to save this runs delta
                    weight_delta[j + w_start + (i * w_per_n)] = cur_delta * dRaw * (*net).weights[j + w_start + (i * w_per_n)];

                    // Lastly to find weight delta we get the previous layers output and use it for compute
                    // So basically we get activation of neuron at Xi for layer k-1
                    double loss_for_weight = (*net).inputs[j];

                    // Now we can finally use delta rule to change connected weight by multiplying our last finding with the learning rate (step size) then subtracting that value from current weight.
                    (*net).weights[j + w_start + (i * w_per_n)] = (*net).weights[j + w_start + (i * w_per_n)] - (cur_delta * loss_for_weight * dRaw * learning_rate);
                }

                // Update bias using chain rule 
                (*net).biases[a_start + i] = (*net).biases[a_start + i] - (learning_rate * cur_delta * dRaw);
                
            }

        }

        // if at hidden layer:
        // To update weights at hidden layer we utilize chain rule to find 𝛿L/𝛿Wij, find 𝛿L/𝛿Wij, for some arbitrary weight going from node i to node j at layer k.
        // note: 𝛿L/𝛿Wij = 𝛿 * ℎ′(r𝑗) * Xij for some node i and j where Wj is the weight from j to next node, r𝑗 is the raw activation value at j (without activation function applied), h is the activation function and x is the activation value at node i.   
        else{
            
            // w_per_n and w_start from previous execution
            int prev_w_per_n = (*net).num_of_weights[cur_size] / (*net).num_of_neurons[cur_size];
            int prev_w_start = cur_weights;

            cur_delta = 0;

            // So now to find delta (𝛿) at some point j, we do: (∑ Wjl * 𝛿l) for all weights connected from node j to some arbitrary node l at the next layer.
            for(int k = 0; k < (*net).num_of_neurons[cur_size]; k++){
                for(int l = 0; l < prev_w_per_n; l++){
                    cur_delta += (weight_delta[l + prev_w_start + (k * prev_w_per_n)]);
                }
            }

            // Debug code for cur_delta here

            // Start traversing current layer nodes
            for (int i = 0; i < a_end; i++){

                // Next to find weight delta we apply the derivative of the activation function of current layer to current nodes raw values
                double dRaw = 0;
                if (strcmp(activator, "ReLU") == 0){
                    dRaw = relu_derivative((*net).activations[a_start + i]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    dRaw = sigmoid_derivative((*net).activations[a_start + i]);
                }

                for(int j = 0; j < w_per_n; j++){
                    // Update delta storage to save this runs delta
                    weight_delta[j + w_start + (i * w_per_n)] = cur_delta * dRaw * (*net).weights[j + w_start + (i * w_per_n)];

                    // Lastly to find weight delta we get the previous layers output and use it for compute
                    // So basically we get activation of neuron at Xi for layer k-1
                    double loss_for_weight = (*net).activations[j + (a_start - input_size)];

                    // Now we can finally use delta rule to change connected weight by multiplying our last finding with the learning rate (step size) then subtracting that value from current weight.
                    (*net).weights[j + w_start + (i * w_per_n)] = (*net).weights[j + w_start + (i * w_per_n)] - (cur_delta * loss_for_weight * dRaw * learning_rate);
                }

                // Update bias using chain rule 
                (*net).biases[a_start + i] = (*net).biases[a_start + i] - (learning_rate * cur_delta * dRaw);
                
            }
        }

        cur_neurons -= (*net).num_of_neurons[cur_size - 1];
        cur_weights -= (*net).num_of_weights[cur_size - 1];
        cur_size -= 1;
    }

    // free local arrays after completion
    free(weight_delta);
}
