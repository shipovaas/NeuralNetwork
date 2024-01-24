#pragma once

#include "eigen/Eigen/Dense"

#include <iostream>
#include <vector>


class LossFunction {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
public:
    LossFunction(const Vector &pred, Vector &result);

    double Distance() const;

private:
    Vector prediction;
    Vector real_result;
};