#include "ActivationFunction.h"
#include <cmath>
#include "ActivationFunction.h"

namespace neuralnet {

    ActivationFunction::ActivationFunction(Function func, Function deriv)
            : func_(func), deriv_(deriv) {}

    DataType ActivationFunction::calc(DataType x) const {
        return func_(x);
    }

    DataType ActivationFunction::derivative(DataType x) const {
        return deriv_(x);
    }

    Eigen::VectorXd ActivationFunction::calc(const Eigen::VectorXd& x) const {
        return x.unaryExpr(func_);
    }

    Eigen::MatrixXd ActivationFunction::derivative(const Eigen::VectorXd& x) const {
        return x.unaryExpr(deriv_).asDiagonal();
    }

    ActivationFunction ActivationFunction::create(ActivationType type) {
        switch (type) {
            case ActivationType::Sigmoid:
                return ActivationFunction(
                        [](DataType x) { return 1.0 / (1.0 + std::exp(-x)); },
                        [](DataType x) { DataType s = 1.0 / (1.0 + std::exp(-x)); return s * (1 - s); });
            case ActivationType::Tanh:
                return ActivationFunction(
                        [](DataType x) { return std::tanh(x); },
                        [](DataType x) { return 1.0 - std::pow(std::tanh(x), 2); });
            case ActivationType::ReLU:
                return ActivationFunction(
                        [](DataType x) { return std::max(0.0, x); },
                        [](DataType x) { return x > 0 ? 1.0 : 0.0; });
            case ActivationType::Linear:
                return ActivationFunction(
                        [](DataType x) { return x; },
                        [](DataType x) { return 1.0; });
            default:
                throw std::runtime_error("Unsupported activation type.");
        }
    }

} // namespace neuralnet
