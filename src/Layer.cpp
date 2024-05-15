#include "Layer.h"
#include <iostream>

namespace neuralnet {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    Layer::Layer(int input_size, int output_size, ActivationFunction activation_func)
            : weights(Matrix::Random(output_size, input_size)),
              biases(Vector::Random(output_size)),
              activation_function(std::move(activation_func)) {}

    Eigen::VectorXd Layer::forward(const Vector& input) {
        std::cout << "Input to layer size: " << input.size() << std::endl;
        input_cache = input;
        output_cache = weights * input + biases;
        return activation_function.calc(output_cache);
        //Vector z = weights * input + biases;
        //output_cache = activation_function.calc(z);
        //output_cache = z.unaryExpr([this](double val) { return activation_function.calc(val); });
        //eturn output_cache;
    }

    Vector Layer::backward(const Vector& grad_output, double learning_rate) {
        std::cout << "Entering backward: grad_output size " << grad_output.size() << std::endl;
        std::cout << "weights.transpose() size: " << weights.transpose().rows() << "x" << weights.transpose().cols() << std::endl;
        std::cout << "output_cache size: " << output_cache.size() << std::endl;

        if (grad_output.size() != output_cache.size()) {
            throw std::runtime_error("Mismatch in grad_output size and output_cache size in backward propagation.");
        }
        Vector derivative = output_cache.unaryExpr([this](double val) { return activation_function.derivative(val); });
        Vector local_grad = grad_output.cwiseProduct(derivative);
        Matrix transposed_grad_weights = input_cache * local_grad.transpose();
        Matrix grad_weights = transposed_grad_weights.transpose();
        std::cout<<" grad_weights.cols() "<<grad_weights.cols()<<std::endl;
        std::cout<<" weights.cols() "<<weights.cols()<<std::endl;
        std::cout<<" grad_weights.rows() "<<grad_weights.rows()<<std::endl;
        std::cout<<"weights.rows() "<< weights.rows()<<std::endl;

        Vector grad_input = weights.transpose() * local_grad;
        if (grad_input.size() != input_cache.size()) {
            throw std::runtime_error("Mismatch in input_cache size and grad_input size.");
        }

        if (grad_weights.rows() != weights.rows() || grad_weights.cols() != weights.cols()) {
            throw std::runtime_error("Mismatch in dimensions for weight gradients.");
        }

        Vector grad_biases = local_grad;
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

    const Matrix& Layer::get_weights() const {
        return weights;
    }

    const Vector& Layer::get_biases() const {
        return biases;
    }
}
