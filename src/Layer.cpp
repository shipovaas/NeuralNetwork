#include "Layer.h"
#include <iostream>

namespace neuralnet {

    Layer::Layer(int input_size, int output_size, ActivationFunction activation_func)
            : weights(Eigen::MatrixXd::Random(output_size, input_size)),
              biases(Eigen::VectorXd::Zero(output_size)),
              activation_function(std::move(activation_func)) {}

    Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {
        std::cout << "Input to layer size: " << input.size() << std::endl;
        input_cache = input;
        Eigen::VectorXd z = weights * input + biases;
        output_cache = z.unaryExpr([this](double val) { return activation_function.calc(val); });
        return output_cache;
    }

    Eigen::VectorXd Layer::backward(const Eigen::VectorXd& grad_output, double learning_rate) {
        std::cout << "Entering backward: grad_output size " << grad_output.size() << std::endl;
        std::cout << "weights.transpose() size: " << weights.transpose().rows() << "x" << weights.transpose().cols() << std::endl;
        std::cout << "output_cache size: " << output_cache.size() << std::endl;

        if (grad_output.size() != output_cache.size()) {
            throw std::runtime_error("Mismatch in grad_output size and output_cache size in backward propagation.");
        }
        Eigen::VectorXd derivative = output_cache.unaryExpr([this](double val) { return activation_function.derivative(val); });
        Eigen::VectorXd local_grad = grad_output.cwiseProduct(derivative);
        Eigen::MatrixXd transposed_grad_weights = input_cache * local_grad.transpose();
        Eigen::MatrixXd grad_weights = transposed_grad_weights.transpose();
        std::cout<<" grad_weights.cols() "<<grad_weights.cols()<<std::endl;
        std::cout<<" weights.cols() "<<weights.cols()<<std::endl;
        std::cout<<" grad_weights.rows() "<<grad_weights.rows()<<std::endl;
        std::cout<<"weights.rows() "<< weights.rows()<<std::endl;

        Eigen::VectorXd grad_input = weights.transpose() * local_grad;
        if (grad_input.size() != input_cache.size()) {
            throw std::runtime_error("Mismatch in input_cache size and grad_input size.");
        }

        if (grad_weights.rows() != weights.rows() || grad_weights.cols() != weights.cols()) {
            throw std::runtime_error("Mismatch in dimensions for weight gradients.");
        }

        Eigen::VectorXd grad_biases = local_grad;
        if (grad_biases.size() != biases.size()) {
            throw std::runtime_error("Mismatch in biases size and grad_biases size.");
        }

        weights -= learning_rate * grad_weights;
        biases -= learning_rate * grad_biases;

        return grad_input;
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
}
