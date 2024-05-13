#pragma once

#include "eigen/Eigen/Dense"
#include <cmath>
#include <functional>

namespace neuralnet {

    using data_type = double;
    using vector = Eigen::VectorXd;
    using matrix = Eigen::MatrixXd;

    enum class activation_type { Sigmoid, Tanh, ReLU, Linear, Softmax };

    class ActivationFunction {
    public:
        using function = std::function<data_type(data_type)>;

        ActivationFunction(function func, function deriv);

        data_type calc(data_type x) const;
        data_type derivative(data_type x) const;
        vector calc(const vector& x) const;
        matrix derivative(const vector& x) const;

        static ActivationFunction create(activation_type type);

    private:
        function func_;
        function deriv_;
    };

} // namespace neuralnet
