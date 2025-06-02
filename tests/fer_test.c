#include <stdio.h>
#include <stdlib.h>
#include "../src/mlpnet.h"
#include "../src/helpers.h"

#define IMG_WIDTH 48
#define IMG_HEIGHT 48
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)
#define NUM_EMOTIONS 7

// Read the .raw formatted data as converted by the python script.
unsigned char* read_raw_image(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    unsigned char* img_data = (unsigned char*)malloc(IMG_SIZE * sizeof(unsigned char));
    if (!img_data) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    size_t read = fread(img_data, sizeof(unsigned char), IMG_SIZE, file);
    if (read != IMG_SIZE) {
        perror("Incomplete read");
        free(img_data);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return img_data;
}

void preprocess_image(double* normalized, unsigned char* raw_pixels) {
    // Normalize pixel values to 0-1 range
    for (int i = 0; i < IMG_SIZE; i++) {
        normalized[i] = raw_pixels[i] / 255.0;
    }
    
    // Could add more preprocessing here:
    // - Face cropping
    // - Histogram equalization
    // - Data augmentation
}


void load_fer_data(double images[][IMG_SIZE], double labels[][NUM_EMOTIONS], 
    char* image_dir, char* label_file, int num_samples) {
    // In a real implementation, you would:
    // 1. Read CSV file containing image paths and labels
    // 2. Load each image file
    // 3. Preprocess each image
    // 4. One-hot encode the labels

    // This is a placeholder structure:
    for (int i = 0; i < num_samples; i++) {
        // Simulate loading image data
        unsigned char raw_image[IMG_SIZE];
        // ... (code to read actual image would go here)

        preprocess_image(images[i], raw_image);

        // Simulate one-hot encoded label
        int label = rand() % NUM_EMOTIONS;
        for (int j = 0; j < NUM_EMOTIONS; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
}

void build_emotion_network(MLP* net) {
    init_mlp(net, IMG_SIZE);
    
    // Recommended architecture for FER:
    add_forward_layer(net, 256, "ReLU", 0.1, 0.01);  // First hidden layer
    add_forward_layer(net, 128, "ReLU", 0.1, 0.01);  // Second hidden layer
    add_forward_layer(net, 64, "ReLU", 0.1, 0.01);   // Third hidden layer
    add_forward_layer(net, NUM_EMOTIONS, "softmax", 0.1, 0.01); // Output layer
}

void train_and_evaluate() {
    MLP net;
    build_emotion_network(&net);
    
    // Load data - replace with actual paths
    int num_train = 28709;  // Typical FER training set size
    double train_images[num_train][IMG_SIZE];
    double train_labels[num_train][NUM_EMOTIONS];
    load_fer_data(train_images, train_labels, "train/images/", "train/labels.csv", num_train);
    
    // Training parameters
    double lr = 0.001;
    int epochs = 15;
    int batch_size = 64;
    
    train_from_source(&net, IMG_SIZE, NUM_EMOTIONS, lr, epochs, 
                     batch_size, num_train, train_images, 
                     train_labels, 1, "CCE");
    
    // Evaluation
    int num_test = 3589;  // Typical FER test set size
    double test_images[num_test][IMG_SIZE];
    double test_labels[num_test][NUM_EMOTIONS];
    load_fer_data(test_images, test_labels, "test/images/", "test/labels.csv", num_test);
    
    double test_input[IMG_SIZE];
    int correct = 0;
    
    for (int i = 0; i < num_test; i++) {
        preprocess_image(test_input, test_images[i]);
        feed_forward(&net, test_input);
        
        // Get predicted emotion
        int pred = 0;
        for (int j = 1; j < NUM_EMOTIONS; j++) {
            if (net.outputs[j] > net.outputs[pred]) {
                pred = j;
            }
        }
        
        // Get true emotion
        int true_emotion = 0;
        for (int j = 0; j < NUM_EMOTIONS; j++) {
            if (test_labels[i][j] == 1.0) {
                true_emotion = j;
                break;
            }
        }
        
        if (pred == true_emotion) correct++;
    }
    
    printf("Emotion Detection Accuracy: %.2f%%\n", 
          (correct / (double)num_test) * 100);
    
    free_mlp(&net);
}