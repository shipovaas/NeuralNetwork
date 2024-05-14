#include "ActivationFunction.h"
#include <cmath>
#include "ActivationFunction.h"

namespace neuralnet {

    ActivationFunction::ActivationFunction(function func, function deriv)
            : func_(func), deriv_(deriv) {}

    data_type ActivationFunction::calc(data_type x) const {
        return func_(x);
    }

    data_type ActivationFunction::derivative(data_type x) const {
        return deriv_(x);
    }

    Eigen::VectorXd ActivationFunction::calc(const Eigen::VectorXd& x) const {
        return x.unaryExpr(func_);
    }

    Eigen::MatrixXd ActivationFunction::derivative(const Eigen::VectorXd& x) const {
        return x.unaryExpr(deriv_).asDiagonal();
    }

    ActivationFunction ActivationFunction::create(activation_type type) {
        switch (type) {
            case activation_type::Sigmoid:
                return ActivationFunction(
                        [](data_type x) { return 1.0 / (1.0 + std::exp(-x)); },
                        [](data_type x) { data_type s = 1.0 / (1.0 + std::exp(-x)); return s * (1 - s); });
            case activation_type::Tanh:
                return ActivationFunction(
                        [](data_type x) { return std::tanh(x); },
                        [](data_type x) { return 1.0 - std::pow(std::tanh(x), 2); });
            case activation_type::ReLU:
                return ActivationFunction(
                        [](data_type x) { return std::max(0.0, x); },
                        [](data_type x) { return x > 0 ? 1.0 : 0.0; });
            case activation_type::Linear:
                return ActivationFunction(
                        [](data_type x) { return x; },
                        [](data_type x) { return 1.0; });
            default:
                throw std::runtime_error("Unsupported activation type.");
        }
    }

} // namespace neuralnet
