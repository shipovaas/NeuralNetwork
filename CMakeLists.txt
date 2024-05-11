cmake_minimum_required(VERSION 3.25)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_executable(neural_network main.cpp ActivationFunction.cpp LossFunction.cpp Layer.h Layer.cpp NeuralNetwork.h NeuralNetwork.cpp)