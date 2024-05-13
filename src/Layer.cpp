#include "Layer.h"

namespace neuralnet {

    Layer::Layer(int input_size, int output_size, ActivationFunction activation_func)
            : weights(Eigen::MatrixXd::Random(output_size, input_size)),
              biases(Eigen::VectorXd::Zero(output_size)),
              activation_function(std::move(activation_func)) {}

    Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {
        input_cache = input;
        Eigen::VectorXd z = weights * input + biases;
        output_cache = z.unaryExpr([this](double val) { return activation_function.calc(val); });
        return output_cache;
    }

    Eigen::VectorXd Layer::backward(const Eigen::VectorXd& grad_output, double learning_rate) {
        Eigen::VectorXd grad_input = weights.transpose() * grad_output.cwiseProduct(output_cache.unaryExpr([this](double val) { return activation_function.derivative(val); }));
        Eigen::MatrixXd grad_weights = grad_input * input_cache.transpose();
        Eigen::VectorXd grad_biases = grad_input;
        weights -= learning_rate * grad_weights;
        biases -= learning_rate * grad_biases;
        return weights.transpose() * grad_input;
    }

    void Layer::save(std::ofstream& file) const {
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));
    }

    void Layer::load(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(double));
        file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(double));
    }

    const Eigen::MatrixXd& Layer::get_weights() const {
        return weights;
    }

    const Eigen::VectorXd& Layer::get_biases() const {
        return biases;
    }

} // namespace neuralnet
