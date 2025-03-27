#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neural.h"
#include "helpers.h"

int main() {
  
    printf("Hello, World!\n");
    
    MLP ANN;

    double input_data[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    double input_slit[2] = {3, 2};

    double target_data[4][1] = {{0}, {1}, {1}, {0}};

    init_net(&ANN, input_slit);

    add_forward_layer(&ANN, 2, "ReLU");

    add_out_layer(&ANN, 2, "sigmoid");

    int size = sizeof(ANN.outputs) / sizeof(ANN.outputs[0]);

   /* printf("size: %d", size);

    for (int i = 0; i < size; i++) {
        printf("Output %d: %f", i, ANN.outputs[i]);
    }*/

    //size = sizeof(ANN.activations) / sizeof(ANN.activations[0]);

    /*printf("size: %d \n", size);
    for (int i = 0; i < size; i++) {
        printf("activations %d: %f \n", i, ANN.activations[i]);
    }*/

    printf("size weight: %d \n", ANN.total_weights);
    for (int i = 0; i < ANN.total_weights; i++) {
        printf("weight %d: %f \n", i, ANN.weights[i]);
    }

    printf("neurons: %d \n", ANN.total_neurons);
    for (int i = 0; i < ANN.total_neurons; i++) {
        printf("by layer %d: %d \n", i, ANN.num_of_neurons[i]);
    }

    return 0;

}