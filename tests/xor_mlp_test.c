#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/mlpnet.h"
#include "../src/helpers.h"

int main(){

    printf("Hello, World!\n");
    
    MLP ANN;

    // XOR TEST

    // Hyper Parameters
    int input_size = 2;
    double lr = 0.1;
    int epochs = 1000;

    // Training Data
    double input_data[4][2] = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
    double target_data[4][1] = {{0}, {1}, {1}, {0}};

    // Initial Input slit
    double input_slit[2] = {1, 1};
    double output_slit[1] = {0};

    // Construct 2-2-1 Neural Net (counting input as first layer)
    init_mlp(&ANN, input_size); 
    add_forward_layer(&ANN, 2, "ReLU", 0.8, 0);
    add_forward_layer(&ANN, 1, "sigmoid", 1, 0);

    feed_forward(&ANN, input_slit);
    backprop(&ANN, lr, output_slit, "MSE");


    for(int epoch = 0; epoch < epochs; epoch++){
        for (int i = 0; i < 4; i++) {
            double input[2] = {input_data[i][0], input_data[i][1]};
            double target[1] = {target_data[i][0]};

            //printf("input: [%f, %f], target: %f", input[0], input[1], target[0]);

            feed_forward(&ANN, input);

            backprop(&ANN, lr, target, "MSE");

            if (epoch % 200 == 0){
                printf("input: [%f, %f], target: %f", input[0], input[1], target[0]);
                printf(" loss: %f, output: %f \n", ANN.loss, ANN.outputs[0]);
            }

        }
    }

    return 0;
}
