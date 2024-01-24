#include "ActivationFunction.h"
#include <cmath>

ActivationFunction::ActivationFunction(std::function<signature> s0, std::function<signature> s1) : f0_(std::move(s0)), f1_(std::move(s1)){};

ActivationFunction::Vector ActivationFunction::ApplyFunction(const Vector &vec) {
    return vec.unaryExpr([this](double x){return f0_(x);});
}

ActivationFunction::Vector ActivationFunction::ApplyDerivative(const Vector &vec) {
    return vec.unaryExpr([this](double x){return f1_(x);});
}

double Sigmoid::ApplyFunction(double x) {
    return 1 / (1 + exp(-x));
}

double Sigmoid::ApplyDerivative(double x) {
    return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}