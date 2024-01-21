#include "ActivationFunction.h"
#include <cmath>

ActivationFunction::ActivationFunction(std::function<signature> s0, std::function<signature> s1) : f0_(std::move(s0)), f1_(std::move(s1)){};

std::vector<double> ActivationFunction::ApplyFunction(const std::vector<double> &vec) {
    std::vector<double> output;
    output.clear();
    for (double value: vec) {
        output.push_back(atan(value));
    };
    return output;
}

std::vector<double> ActivationFunction::ApplyDerivative(const std::vector<double> &vec) {
    std::vector<double> output;
    for (double value: vec) {
        output.push_back(1 / (1 + value * value));
    };
    return output;
}

double Sigmoid::ApplyFunction(double x) {
    return 1 / (1 + exp(-x));
}

double Sigmoid::ApplyDerivative(double x) {
    return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}