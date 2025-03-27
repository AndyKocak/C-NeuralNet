#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neural.h"
#include "helpers.h"

// construct network struct with input layer
void init_net(MLP *net, double* inputs, int input_size){

    // Init array that holds input values and wrappers that hold total layer count (size) and total neuron count.
    (*net).inputs = inputs;
    (*net).size = 0;
    (*net).total_neurons = 0;
    (*net).total_weights = 0;
    (*net).total_inputs = input_size;

    // Init array that stores how many neurons and weights in each layer
    (*net).num_of_neurons = (int*)malloc(1 * sizeof(int));
    (*net).num_of_neurons[0] = 0;
    (*net).num_of_weights = (int*)malloc(1 * sizeof(int));
    (*net).num_of_weights[0] = 0;
    
    // Init all forward/backward pass variable arrays
    (*net).weights = (double*)malloc(1 * sizeof(double));
    (*net).weights[0] = 0;
    (*net).biases = (double*)malloc(1 * sizeof(double));
    (*net).biases[0] = 0;
    (*net).activations = (double*)malloc(1 * sizeof(double));
    (*net).activations[0] = 0;
    (*net).raw_activations = (double*)malloc(1 * sizeof(double));
    (*net).raw_activations[0] = 0;

    // Init string array that stores activation functions for each layer
    (*net).activation_funcs = (char **)malloc(1 * sizeof(char *));
    (*net).activation_funcs[0] = (char *)malloc(50 * sizeof(char));
    strcpy((*net).activation_funcs[0], "none");
}

// add new forward pass layer to network 
void add_forward_layer(MLP *(net), int units, char *activator, double w_mult_bias, double w_add_bias){

    // Increment counters for network size
    (*net).size = (*net).size + 1;
    (*net).total_neurons = (*net).total_neurons + units;

    // Reallocate pointer to dynamically add more values
    int length = (*net).size;
    (*net).num_of_neurons = realloc((*net).num_of_neurons,(length+1) * sizeof(int));
    (*net).num_of_neurons[(length)] = units;

    // Reallocate string array to add activator value
    (*net).activation_funcs = realloc((*net).activation_funcs, (length+1) * sizeof(char *));
    (*net).activation_funcs[(length)] = (char *)malloc(50 * sizeof(char));
    strcpy((*net).activation_funcs[(length)], activator);

    // change input_size depending on whether this is initial hidden layer or not.
    int input_size = ((*net).size == 1) ? (*net).total_inputs : (*net).num_of_neurons[((*net).size - 1)];

    // Reallocate array that stores # weight for each layer
    (*net).num_of_weights = realloc((*net).num_of_weights, (length+1) * sizeof(int));
    (*net).num_of_weights[(length)] = units * input_size;

    // Reallocate weigth/bias arays and generate/assign random values
    length = (*net).total_weights + 1;

    (*net).weights = realloc((*net).weights,(length+(units * input_size)) * sizeof(double));
    (*net).biases = realloc((*net).biases,(length+units) * sizeof(double));

    init_weights((*net).weights, (units * input_size), length, w_mult_bias, w_add_bias);
    init_biases((*net).biases, units, length);

    // Reallocate activations array to store each neurons activation values
    length = (*net).total_neurons;
    (*net).activations = realloc((*net).activations, (length+1) * sizeof(double));
    (*net).raw_activations = realloc((*net).activations, (length+1) * sizeof(double));

    // Forward pass through perceptrons in layer right after input layer
    if ((*net).size == 1){

        // Apply recursive formula for perceptron
        for (int i = 0; i < units; i++){
            (*net).activations[(i+1)] = 0;
            for(int j = 1; j <= input_size; j++){
                (*net).activations[(i+1)] += (*net).inputs[(j-1)] * (*net).weights[i * input_size + j];
                //printf("layer %d cur i: %d cur j: %d cur weight_index: %d cur weight: %f cur input: %f cur active_index: %d cur active: %f \n",(*net).size, i, j, i * input_size + j, (*net).weights[i * input_size + j], (*net).inputs[(j-1)], i+1, (*net).activations[(i+1)]);
            }
            (*net).activations[(i+1)] += (*net).biases[(i+1)];
            (*net).raw_activations[(i+1)] = (*net).activations[(i+1)];
            // Conditionally apply activation based on what was given as arg
            if (strcmp(activator, "ReLU") == 0){
                (*net).activations[i+1] = ReLU((*net).activations[i+1]);
            }
            if (strcmp(activator, "sigmoid") == 0){
                (*net).activations[i+1] = sigmoid((*net).activations[i+1]);
            }
        }

        // Change total weight count
        (*net).total_weights += (units * input_size);
    }

    // Pretty much the same except instead of taking input from input layer, use previous activators instead
    // (Make sure this one works)
    else{

        // Apply recursive formula for perceptron
        for (int i = 0; i < units; i++){
            (*net).activations[(i + 1 + ((*net).total_neurons - units))] = 0;
            for(int j = 1; j <= input_size; j++){
                (*net).activations[(i + 1 + ((*net).total_neurons - units))] += (*net).activations[j + ((*net).total_neurons - (units + input_size))] * (*net).weights[(i * input_size + j) + ((*net).total_weights)];
            }
            (*net).activations[(i + 1 + ((*net).total_neurons - units))] += (*net).biases[(i + 1 + ((*net).total_neurons - units))];
            (*net).raw_activations[(i+1)] = (*net).activations[(i+1)];
            // Conditionally apply activation based on what was given as arg
            if (strcmp(activator, "ReLU") == 0){
                (*net).activations[(i + 1 + ((*net).total_neurons - units))] = ReLU((*net).activations[(i + 1 + ((*net).total_neurons - units))]);
            }
            if (strcmp(activator, "sigmoid") == 0){
                (*net).activations[(i + 1 + ((*net).total_neurons - units))] = sigmoid((*net).activations[(i + 1 + ((*net).total_neurons - units))]);
            }
        }

        // Change total weight count, this is at the end because we want to use previous weight count when navigating 
        (*net).total_weights += (units * input_size);
    }
}

// add output layer to network
void add_out_layer(MLP *net, int units, char *activator, double w_mult_bias, double w_add_bias){

    // Increment counters for network size
    (*net).size = (*net).size + 1;
    (*net).total_neurons = (*net).total_neurons + units;

    // Reallocate pointer to dynamically add more values
    int length = (*net).size;
    (*net).num_of_neurons = realloc((*net).num_of_neurons,(length+1) * sizeof(int));
    (*net).num_of_neurons[(length)] = units;

    // Reallocate string array to add activator value
    (*net).activation_funcs = realloc((*net).activation_funcs, (length+1) * sizeof(char *));
    (*net).activation_funcs[length] = (char *)malloc(50 * sizeof(char));
    strcpy((*net).activation_funcs[length], activator);

    // change input_size depending on whether this is initial hidden layer or not.
    int input_size = ((*net).size == 1) ? (*net).total_inputs : (*net).num_of_neurons[((*net).size - 1)];

    // Reallocate array that stores # weight for each layer
    (*net).num_of_weights = realloc((*net).num_of_weights, (length+1) * sizeof(int));
    (*net).num_of_weights[(length)] = units * input_size;

    // Reallocate weigth/bias arays and generate/assign random values
    length = (*net).total_weights + 1;
    (*net).weights = realloc((*net).weights,(length+(units * input_size)) * sizeof(double));
    (*net).biases = realloc((*net).biases,(length+units) * sizeof(double));

    init_weights((*net).weights, (units * input_size), length, w_mult_bias, w_add_bias);
    init_biases((*net).biases, units, length);

    // Reallocate output array to store each neurons activation values
    (*net).outputs = (double*)malloc(units * sizeof(double));

    // Apply recursive formula for perceptron
    for (int i = 0; i < units; i++){
        (*net).outputs[i] = 0;
        for(int j = 1; j <= input_size; j++){
            (*net).outputs[i] += (*net).activations[j + ((*net).total_neurons - units - 2)] * (*net).weights[(i * input_size + j) + ((*net).total_weights)];
            //printf("layer %d cur i: %d cur j: %d cur weight_index: %d cur weight: %f cur input: %f cur active_index: %d cur active: %f total_weight: %d \n", (*net).size, i, j, (i * input_size + j) + ((*net).total_weights), (*net).weights[(i * input_size + j) + ((*net).total_weights)], (*net).activations[j + ((*net).total_neurons - units - 2)], (j + ((*net).total_neurons - units - 2)), (*net).outputs[i], (*net).total_weights);
        }
        (*net).outputs[i] += (*net).biases[i + 1 + ((*net).total_neurons - units)];
        //printf("Before activation function: %f Positioned at i: %d \n", (*net).outputs[(i)], i);
        // Conditionally apply activation based on what was given as arg
        if (strcmp(activator, "ReLU") == 0){
            (*net).outputs[i] = ReLU((*net).outputs[i]);
        }
        if (strcmp(activator, "sigmoid") == 0){
            (*net).outputs[i] = sigmoid((*net).outputs[i]);
        }
    }
    
    (*net).total_weights += (units * input_size);
}

// feed forward input data to get output on constructed neural-net
void feed_forward(MLP *net, double* inputs, int input_len){
    
    // Change input array of MLP to new input arg
    (*net).inputs = inputs;
    (*net).total_inputs = input_len;

    // Reset trackers to properly traverse arrays
    int total_neurons = 0;
    int total_weights = 0;

    // Iterate through each layer starting from input layer
    for (int n = 1; n <= (*net).size; n++){

        // Get input size for each layer
        int input_size = (n == 1) ? (*net).total_inputs : (*net).num_of_neurons[n-1];
        
        // Init layer specific info
        int units = (*net).num_of_neurons[n];
        char *activator = (*net).activation_funcs[n];

        // Update total navigated neuron tracker
        total_neurons += (*net).num_of_neurons[n];

        // If in layer right after input
        if (n == 1){
            // Apply recursive formula for perceptron
            for (int i = 0; i < units; i++){
                (*net).activations[(i+1)] = 0;
                for(int j = 1; j <= input_size; j++){
                    (*net).activations[(i+1)] += (*net).inputs[(j-1)] * (*net).weights[i * input_size + j];
                    //printf("layer %d cur i: %d cur j: %d cur weight_index: %d cur weight: %f cur input: %f cur active_index: %d cur active: %f \n", n, i, j, i * input_size + j, (*net).weights[i * input_size + j], (*net).inputs[(j-1)], i+1, (*net).activations[(i+1)]);
                }

                // Apply bias to node
                (*net).activations[(i+1)] += (*net).biases[(i+1)];
                (*net).raw_activations[(i+1)] = (*net).activations[(i+1)];

                // Conditionally apply activation based on what was given as arg
                if (strcmp(activator, "ReLU") == 0){
                    (*net).activations[i+1] = ReLU((*net).activations[i+1]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    (*net).activations[i+1] = sigmoid((*net).activations[i+1]);
                }
                //printf("Post function: %f \n", (*net).activations[(i+1)]);
            }
        }

        // If in output layer
        else if (n == ((*net).size)){
            //printf("Enter output one \n");
            for (int i = 0; i < units; i++){
                (*net).outputs[i] = 0;
                for(int j = 1; j <= input_size; j++){
                    (*net).outputs[i] += (*net).activations[j + (total_neurons - (units + input_size))] * (*net).weights[(i * input_size + j) + (total_weights)];
                    //printf("layer %d cur i: %d cur j: %d cur weight_index: %d cur weight: %f cur input: %f cur active_index: %d cur output: %f total_weight: %d \n", n, i, j, (i * input_size + j) + (total_weights), (*net).weights[(i * input_size + j) + total_weights], (*net).activations[j + (total_neurons - (units + input_size))], (j + (total_neurons - (units + input_size))), (*net).outputs[i], total_weights);
                }

                (*net).outputs[i] += (*net).biases[i + 1 + (total_neurons - units)];
                (*net).raw_activations[(i+1)] = (*net).activations[(i+1)];

                // Conditionally apply activation based on what was given as arg
                if (strcmp(activator, "ReLU") == 0){
                    (*net).outputs[i] = ReLU((*net).outputs[i]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    (*net).outputs[i] = sigmoid((*net).outputs[i]);
                }
                //printf("Post function: %f \n", (*net).outputs[i]);
            }
        }

        // If in hidden layer
        else{
            for (int i = 0; i < units; i++){
                (*net).activations[(i + 1 + (total_neurons - units))] = 0;
                for(int j = 1; j <= input_size; j++){
                    (*net).activations[(i + 1 + (total_neurons - units))] += (*net).activations[j + (total_neurons - (units + input_size))] * (*net).weights[(i * input_size + j) + (total_weights)];
                    //printf("layer %d cur i: %d cur j: %d cur weight_index: %d cur weight: %f cur input: %f cur active_index: %d cur active: %f total_weight: %d \n", n, i, j, (i * input_size + j) + (total_weights), (*net).weights[(i * input_size + j) + total_weights], (*net).activations[j + (total_neurons - (units + input_size))], (i + 1 + (total_neurons - units)), (*net).activations[(i + 1 + (total_neurons - units))], total_weights);
                }

                (*net).activations[(i + 1 + (total_neurons - units))] += (*net).biases[(i + 1 + (total_neurons - units))];
                (*net).raw_activations[(i+1)] = (*net).activations[(i+1)];
                
                // Conditionally apply activation based on what was given as arg
                if (strcmp(activator, "ReLU") == 0){
                    (*net).activations[(i + 1 + (total_neurons - units))] = ReLU((*net).activations[(i + 1 + (total_neurons - units))]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    (*net).activations[(i + 1 + (total_neurons - units))] = sigmoid((*net).activations[(i + 1 + (total_neurons - units))]);
                }
                //printf("Post function: %f \n", (*net).activations[(i+1 + (total_neurons - units))]);
            }
        }
        // update total weight
        total_weights += (units * input_size);
    }
}

// THIS ONE CAUSES PROBLEMS!!!
void backprop(MLP (*net), double learning_rate, double* target, char *error_func){
    double* output_delta = (double*)malloc((*net).num_of_neurons[(*net).size] * sizeof(double));
    double* hidden_delta = (double*)malloc(((*net).total_neurons + 1 - (*net).num_of_neurons[(*net).size]) * sizeof(double));

    hidden_delta[0] = 0;

    // size and weight counters
    int start_size = (*net).size + 1;
    int cur_neurons = (*net).total_neurons - (*net).num_of_neurons[(*net).size];
    int cur_weights = (*net).total_weights - (*net).num_of_weights[(*net).size];

    // Add more loss options in future
    if (strcmp(error_func, "MSE") == 0){
        printf("Loss: %f, output: %f, actual: %f \n", mean_square_error((*net).outputs, target, (*net).num_of_neurons[(*net).size]), (*net).outputs[0], target[0]);
    }
    if (strcmp(error_func, "BCE") == 0){
        printf("Loss: %f, output: %f, actual: %f \n", (target[0] - (*net).outputs[0]), (*net).outputs[0], target[0]);
    }

    double error = 0;

    // Find and update output delta
    for (int i = 0; i < (*net).num_of_neurons[start_size - 1]; i++){
        error += (target[i] - (*net).outputs[i]);
        char *activator = (*net).activation_funcs[start_size - 1];

        //printf("pre function delta: %f", error);

        if (strcmp(activator, "ReLU") == 0){
            output_delta[i] = error * relu_derivative((*net).outputs[i]);;
        }
        if (strcmp(activator, "sigmoid") == 0){
            output_delta[i] = error * sigmoid_derivative((*net).outputs[i]);
        } 
        //printf(" post function delta: %f ", output_delta[i]);
        // weight per neuron
        int w_per_n = (*net).num_of_weights[(*net).size] / (*net).num_of_neurons[(*net).size];

        // update weights
        for(int j = 0; j < w_per_n; j++){
            printf("Weight %d prev: %f , delta: %f\n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1], output_delta[i]);
            (*net).weights[(i * w_per_n) + cur_weights + j + 1] = (*net).weights[(i * w_per_n) + cur_weights + j + 1] - output_delta[i];
            printf("Weight %d after: %f \n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1]);
        }
    }

    start_size = start_size - 1;

    // Find and update hidden delta
    for(int cur_size = start_size; cur_size > 1; cur_size--){
        printf("Cur size: %d \n", cur_size);
        // update neuron and weight counters
        cur_neurons = cur_neurons - (*net).num_of_neurons[cur_size - 1];
        cur_weights = cur_weights - (*net).num_of_weights[cur_size - 1];

        // get activation_func for layer
        char *activator = (*net).activation_funcs[cur_size - 1];

        // weight per neuron
        int w_per_n = (*net).num_of_weights[cur_size - 1] / (*net).num_of_neurons[cur_size - 1];

        // edge case where only 1 hidden and 1 output layer exist
        if ((cur_size == (*net).size) && cur_size == 2){
            printf("Enter edge one \n");
            //printf("Num neurons: %d \n", (*net).num_of_neurons[cur_size - 1]);
            // calculate sum of all output neuron delta * corresponding neuron weight
            for (int i = 0; i < (*net).num_of_neurons[cur_size - 1]; i++){
                // Init current delta for ith node
                double cur_delta = 0;

                // Sum the dot product of each weight of i connecting to next layer and the corresponding node delta.
                for (int j = 0; j < (*net).total_inputs; j++) {
                    cur_delta += (*net).weights[(i * w_per_n) + j + 1] * output_delta[j];
                }

                //printf("pre function delta: %f", cur_delta);
                
                // Multiply by the derivative of activation of ith node
                if (strcmp(activator, "ReLU") == 0){
                    cur_delta = cur_delta * relu_derivative((*net).activations[cur_neurons + i + 1]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    cur_delta = cur_delta * sigmoid_derivative((*net).activations[cur_neurons + i + 1]);
                }

                hidden_delta[cur_neurons + i + 1] = cur_delta;
                //printf("CUR I: %d", i);
                // NOT SURE IF THIS IS CORRECT CHANGE LATER
                // Change weights connected to ith neuron by dot product of delta and learning rate.
                for (int j = 0; j < (*net).total_inputs; j++) {
                    printf("Weight %d prev: %f, activationn: %f, delta: %f\n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1], (*net).activations[cur_neurons + i + 1], cur_delta);
                    (*net).weights[(i * w_per_n) + cur_weights + j + 1] = (*net).weights[(i * w_per_n) + cur_weights + j + 1] - (learning_rate * cur_delta * (*net).activations[cur_neurons + i + 1]);
                    printf("Weight %d after: %f \n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1]);
                }

                // Update bias
                (*net).biases[cur_neurons + i + 1] = (*net).biases[cur_neurons + i + 1] - (learning_rate * cur_delta);
            }
        }

        // add dotproduct with delta output
        else if (cur_size == (*net).size){
            printf("Enter 1st one \n");
            // calculate sum of all output neuron delta * corresponding neuron weight
            for (int i = 0; i < (*net).num_of_neurons[cur_size - 1]; i++){
                // Init current delta for ith node
                double cur_delta = 0;

                // Sum the dot product of each weight of i connecting to next layer and the corresponding node delta.
                for (int j = 0; j < (*net).num_of_neurons[cur_size - 2]; j++) {
                    cur_delta += (*net).weights[(i * w_per_n) + cur_weights + j + 1] * output_delta[j];
                }
                
                //printf("pre function delta: %f", cur_delta);

                // Multiply by the derivative of activation of ith node
                if (strcmp(activator, "ReLU") == 0){
                    cur_delta = cur_delta * relu_derivative((*net).activations[cur_neurons + i + 1]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    cur_delta = cur_delta * sigmoid_derivative((*net).activations[cur_neurons + i + 1]);
                }

                hidden_delta[cur_neurons + i + 1] = cur_delta;

                // NOT SURE IF THIS IS CORRECT CHANGE LATER
                // Change weights connected to ith neuron by dot product of delta and learning rate.
                for (int j = 0; j < (*net).num_of_neurons[cur_size - 2]; j++) {
                    printf("Weight %d prev: %f, activationn: %f, delta: %f\n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1], (*net).activations[cur_neurons + i + 1], cur_delta);
                    (*net).weights[(i * w_per_n) + cur_weights + j + 1] = (*net).weights[(i * w_per_n) + cur_weights + j + 1] - (learning_rate * cur_delta * (*net).activations[cur_neurons + i + 1]);
                    printf("Weight %d after: %f \n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1]);
                }

                // Update bias
                (*net).biases[cur_neurons + i + 1] = (*net).biases[cur_neurons + i + 1] - (learning_rate * cur_delta);
            }
        }

        // add dotproduct with previous delta hidden but in activation derivative use input
        else if (cur_size == 2){
            printf("Enter 2nd one \n");
            //printf("Num neurons: %d \n", (*net).num_of_neurons[cur_size - 1]);
            // calculate sum of all output neuron delta * corresponding neuron weight
            for (int i = 0; i < (*net).num_of_neurons[cur_size - 1]; i++){
                // Init current delta for ith node
                double cur_delta = 0;

                // Sum the dot product of each weight of i connecting to next layer and the corresponding node delta.
                for (int j = 0; j < (*net).total_inputs; j++) {
                    cur_delta += (*net).weights[(i * w_per_n) + j + 1] * hidden_delta[j + 1 + (*net).num_of_neurons[cur_size - 1]];
                }

                //printf("pre function delta: %f", cur_delta);
                
                // Multiply by the derivative of activation of ith node
                if (strcmp(activator, "ReLU") == 0){
                    cur_delta = cur_delta * relu_derivative((*net).activations[cur_neurons + i + 1]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    cur_delta = cur_delta * sigmoid_derivative((*net).activations[cur_neurons + i + 1]);
                }

                hidden_delta[cur_neurons + i + 1] = cur_delta;
                //printf("CUR I: %d", i);
                // NOT SURE IF THIS IS CORRECT CHANGE LATER
                // Change weights connected to ith neuron by dot product of delta and learning rate.
                for (int j = 0; j < (*net).total_inputs; j++) {
                    printf("Weight %d prev: %f, activationn: %f, delta: %f\n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1], (*net).activations[cur_neurons + i + 1], cur_delta);
                    (*net).weights[(i * w_per_n) + cur_weights + j + 1] = (*net).weights[(i * w_per_n) + cur_weights + j + 1] - (learning_rate * cur_delta * (*net).activations[cur_neurons + i + 1]);
                    printf("Weight %d after: %f \n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1]);
                }

                // Update bias
                (*net).biases[cur_neurons + i + 1] = (*net).biases[cur_neurons + i + 1] - (learning_rate * cur_delta);
            }
        }

        // add dotproduct with previous delta hidden
        else{
            printf("Enter 3rd one \n");
            // calculate sum of all output neuron delta * corresponding neuron weight
            for (int i = 0; i < (*net).num_of_neurons[cur_size - 1]; i++){
                // Init current delta for ith node
                double cur_delta = 0;

                // Sum the dot product of each weight of i connecting to next layer and the corresponding node delta.
                for (int j = 0; j < (*net).num_of_neurons[cur_size - 2]; j++) {
                    cur_delta += (*net).weights[(i * w_per_n) + cur_weights + j + 1] * hidden_delta[j + 1 + cur_neurons];
                }

                //printf("pre function delta: %f", cur_delta);
                
                // Multiply by the derivative of activation of ith node
                if (strcmp(activator, "ReLU") == 0){
                    cur_delta = cur_delta * relu_derivative((*net).activations[cur_neurons + i + 1]);
                }
                if (strcmp(activator, "sigmoid") == 0){
                    cur_delta = cur_delta * sigmoid_derivative((*net).activations[cur_neurons + i + 1]);
                }

                hidden_delta[cur_neurons + i + 1] = cur_delta;

                // NOT SURE IF THIS IS CORRECT CHANGE LATER
                // Change weights connected to ith neuron by dot product of delta and learning rate.
                for (int j = 0; j < (*net).num_of_neurons[cur_size - 2]; j++) {
                    printf("Weight %d prev: %f, activationn: %f, delta: %f\n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1], (*net).activations[cur_neurons + i + 1], cur_delta);
                    (*net).weights[(i * w_per_n) + cur_weights + j + 1] = (*net).weights[(i * w_per_n) + cur_weights + j + 1] - (learning_rate * cur_delta * (*net).activations[cur_neurons + i + 1]);
                    printf("Weight %d after: %f \n", (i * w_per_n) + cur_weights + j + 1, (*net).weights[(i * w_per_n) + cur_weights + j + 1]);
                }

                // Update bias
                (*net).biases[cur_neurons + i + 1] = (*net).biases[cur_neurons + i + 1] - (learning_rate * cur_delta);
            }
        }
    }

    // free local arrays after completion
    free(output_delta);
    free(hidden_delta);
}

// THIS IS FOR TESTING LOCALLY BEFORE BACKPROP DELETE LATER

// Write example code to see if local prints work and go from there.
int main(){

    printf("Hello, World!\n");
    
    MLP ANN;

    // XOR TEST

    // Hyper Parameters
    int input_size = 2;
    double lr = 0.1;
    int epochs = 10;

    // Training Data
    double input_data[4][2] = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
    double target_data[4][1] = {{0}, {1}, {1}, {0}};

    // Initial Input slit
    double input_slit[2] = {1, 1};
    double output_slit[1] = {0};

    // Construct 2-2-1 Neural Net (not counting input)
    init_net(&ANN, input_slit, input_size);
    //add_forward_layer(&ANN, 2, "sigmoid", 0.8, 0);
    add_forward_layer(&ANN, 2, "ReLU", 0.8, 0);
    add_out_layer(&ANN, 1, "sigmoid", 1, 0);

    backprop(&ANN, lr, output_slit, "BCE");


    for(int epoch = 0; epoch < epochs; epoch++){
        for (int i = 0; i < 4; i++) {
            double input[2] = {input_data[i][0], input_data[i][1]};
            double target[1] = {target_data[i][0]};

            printf("input: [%f, %f], target: %f", input[0], input[1], target[0]);

            feed_forward(&ANN, input, input_size);

            backprop(&ANN, lr, target, "BCE");

        }
    }


    return 0;
}
