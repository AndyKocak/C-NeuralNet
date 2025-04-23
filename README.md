# C-NeuralNet 

## Description
A highly modular and beginner friendly C framework to build and train MLP Neural Networks for regression and classification tasks using only the C99 standard library. The framework itself serves as a simple abstraction for fundemental functionalities of
feedforward ANN's such as backpropogation, gradient descent, and data fitting to train the network. Basically
it's a C implementation of some popular high-level machine learning libraries most are accustomed to such as 
scikit-learn, Keras, and Pytorch. It's a great way to both understand how a Neural Network works under the hood
as well as build performant solutions for classic machine learning problems as demonstrated in some of the 
examples I've provided under tets.

## Features
* __Customizable Architecture__: You can define the number of layers, the number of neurons per layer, the activation function for the layer, an initial offset for bias in layer, and more.

* __Simple to Modify__: The MLP struct type holds all the necessary information that might be needed to implement any additional features that are not present in the current build like new activation functions or different optimizers such as ADAM.

* __Includes Examples and Test Suite__: For those new to deep learning the framework features an example implementation of binary classification for the XOR problem, and multi-class classification for the Iris dataset. All using the frameworks functionalities. 

* __This tool is released under the MIT license so you are free to use it on your own projects.__

## TODO
- [ ] Add more activation functions and optimizers for the framework. 
- [ ] Add functionality to save and load trained neural networks.
- [ ] Finish constructing documentation.
- [x] Finish makefile for simple linking/compiling
- [x] Build tests for XOR and Iris datasets.
- [x] Create functionality for data feedforward as well as backpropogation for any possible MLP struct architecture.
- [x] Change MLP to be able to handle dynamic growth for large neural networks as well as function to release from memory.
- [x] Implement basic struct to hold MLP features and parameters, as well as an initializer.
- [x] Add functionality for the derivatives of each activation function for use in backpropogation.
- [x] Add support for key mathematical functions like sigmoid, tanh, softmax, ReLU, etc.

## Building


