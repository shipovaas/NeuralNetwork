cmake_minimum_required(VERSION 3.25)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)

add_executable(neural_network main.cpp neural_network.h ActivationFunction.cpp ActivationFunction.h LossFunction.cpp LossFunction.h)