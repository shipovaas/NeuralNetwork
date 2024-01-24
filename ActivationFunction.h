#pragma once

#include "eigen/Eigen/Dense"

#include <iostream>
#include <vector>

class ActivationFunction {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
public:
    using signature = double(double);
    ActivationFunction(std::function<signature> s0, std::function<signature> s1);
    //using Vector = std::Vector

    Vector ApplyFunction(const Vector &vec);

    Vector ApplyDerivative(const Vector &vec);

private:
    std::function<signature> f1_;
    std::function<signature> f0_;
};

class Sigmoid {
public:
    double ApplyFunction(double x);

    double ApplyDerivative(double x);

};

