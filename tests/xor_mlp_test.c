#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/mlpnet.h"
#include "../src/helpers.h"

int main(){
    
    MLP ANN;

    // XOR TEST

    // Hyper Parameters
    int input_size = 2;
    int out_size  = 1;
    double lr = 0.1;
    int epochs = 1000;
    int batch_size = 4;
    int threshold = 200;
    int n_train = 4;

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

    // Example call for inputing data through network and using backpropogation
    feed_forward(&ANN, input_slit);
    backprop(&ANN, lr, output_slit, "MSE");

    // Use a 2D Array of double to train neural network
    train_from_source(&ANN, input_size, out_size, lr, epochs, batch_size, n_train, input_data, target_data, threshold, "MSE");

    return 0;
}
