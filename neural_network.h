#pragma once

#include <iostream>
#include <vector>
#include <cmath>

class NeuralNetwork {
private:
    std::vector<int> layers_size;
public:
    NeuralNetwork(std::vector<int> &layers) : layers_size(layers) {}
};
