#pragma once

#include <iostream>
#include <vector>

class ActivationFunction {
public:
    using signature = double(double);
    ActivationFunction(std::function<signature> s0, std::function<signature> s1);
    //using Vector = std::Vector

    std::vector<double> ApplyFunction(const std::vector<double> &vec);

    std::vector<double> ApplyDerivative(const std::vector<double> &vec);

private:
    std::function<signature> f1_;
    std::function<signature> f0_;
};

class Sigmoid {
public:
    double ApplyFunction(double x);

    double ApplyDerivative(double x);

};

