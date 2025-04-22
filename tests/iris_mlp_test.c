#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/mlpnet.h"
#include "../src/helpers.h"

#define MAX_FILE_SIZE 1000000  // Adjust this size depending on your needs

int read_csv(int r_size, int c_size, int n_out, double inputArr[r_size][c_size],double targetArr[r_size][n_out]){
    // Substitute the full file path
    // for the string file_path
    FILE* fp = fopen("data/Iris.csv", "r");
 
    if (!fp)
        perror("Can't open file\n");
 
    else {
        char *buffer = (char *)malloc(MAX_FILE_SIZE * sizeof(char));
        if (buffer == NULL) {
            perror("Failed to allocate memory\n");
            fclose(fp);
            return 1;
        }   
 
        int row = 0;
        int column = 0;
 
        while (fgets(buffer, MAX_FILE_SIZE, fp)) {
            column = 0;
            row++;
 
            // To avoid printing of column
            // names in file can be changed
            // according to need
            if (row == 1)
                continue;
 
            // Splitting the data
            char* value = strtok(buffer, ", ");
 
            while (value) {
                // Column 1 Id
                if (column == 0) {
                    printf("\tId: ");
                }
 
                // Column 2 Sepal Length
                if (column == 1) {
                    double sl = atof(value);
                    inputArr[row-2][column-1] = sl;
                    printf("\tSepalLength: ");
                }
 
                // Column 3 Sepal Width
                if (column == 2) {
                    double sw = atof(value);
                    inputArr[row-2][column-1] = sw;
                    printf("\tSepalWidth: ");
                }

                // Column 4 Petal Length
                if (column == 3) {
                    double pl = atof(value);
                    inputArr[row-2][column-1] = pl;
                    printf("\tPetalLength: ");
                }

                // Column 5 Petsal Width
                if (column == 4) {
                    double pw = atof(value);
                    inputArr[row-2][column-1] = pw;
                    printf("\tPetalWidth: ");
                }

                // Column 6 Species
                // This applies One Hot Encoding to label values
                // In this specfic example there are 3 labels but you can
                // Write your own encoding for some other multi-class data
                if (column == 5) {
                    printf("OHE val: [ ");

                    if (strcmp(value, "Iris-setosa") == 10){
                        double encoding[3] = {1, 0, 0};
                        for (int n = 0; n < n_out; n++){
                            targetArr[row-2][n] = encoding[n];
                            printf(" %f", encoding[n]);
                        }
                    }
                    else if (strcmp(value, "Iris-versicolor") == 10){
                        double encoding[3] = {0, 1, 0};
                        for (int n = 0; n < n_out; n++){
                            targetArr[row-2][n] = encoding[n];
                            printf(" %f", encoding[n]);
                        }
                    }
                    else if (strcmp(value, "Iris-virginica") == 10){
                        double encoding[3] = {0, 0, 1};
                        for (int n = 0; n < n_out; n++){
                            targetArr[row-2][n] = encoding[n];
                            printf(" %f", encoding[n]);
                        }
                    }
                    printf(" ] ");
                    printf("\tSpecies: ");
                }
 
                printf("%s", value);
                value = strtok(NULL, ", ");
                column++;
            }
 
            printf("\n");
        }
 
        // Close the file
        fclose(fp);
    }
    return 0;
}

int main(){    
    // Number of data
    int n_data = 150;
    // Number of input features
    int n_feature = 4;
    // Number of possible outputs
    int n_out = 3;

    // Init seperated arrays
    double input_arr[n_data][n_feature];
    double target_arr[n_data][n_out];

    // Read Iris data from csv
    read_csv(n_data, n_feature, n_out, input_arr, target_arr);

    // Check if data read correctly (you can comment these out if you want)
    printf("\n Extracted data: \n");
    printf(" Input Data: [");

    for (int i = 1; i < n_data; i++){
        printf("[ ");
        for (int j = 0; j < n_feature; j++){
            printf("%f ", input_arr[i][j]);
        }
        printf(" ]\n");
    }

    printf("] \n Target Values: [");
    for (int i = 1; i < n_data; i++){
        printf("[ ");
        for (int j = 0; j < n_out; j++){
            printf("%f ", target_arr[i][j]);
        }
        printf(" ]\n");
    }

    // Finally it's time to build MLP

    MLP ANN;

    // Hyper Parameters
    int input_size = 4;
    int out_size  = 3;
    double lr = 0.1;
    int epochs = 500;
    int batch_size = 20;
    int n_train = 150;
    int threshold = 100;

    // Construct Network
    init_mlp(&ANN, input_size); 
    add_forward_layer(&ANN, 8, "ReLU", 0.5, 0);
    add_forward_layer(&ANN, 3, "softmax", 0.5, 0);

    // Use extracted training data to train neural network
    train_from_source(&ANN, input_size, out_size, lr, (epochs+1), batch_size, n_train, input_arr, target_arr, threshold, "CCE");

    return 1;
}

