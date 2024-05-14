#pragma once

#include "Eigen/Dense"
#include <fstream>
#include "ActivationFunction.h"

namespace neuralnet {

    class Layer {
    public:
        Layer(int input_size, int output_size, ActivationFunction activation_func);

        Eigen::VectorXd forward(const Eigen::VectorXd& input);

        Eigen::VectorXd backward(const Eigen::VectorXd& grad_output, double learning_rate);

        const Eigen::MatrixXd& get_weights() const;
        const Eigen::VectorXd& get_biases() const;

        void save(std::ofstream& file) const;
        void load(std::ifstream& file);

    private:
        Eigen::MatrixXd weights;
        Eigen::VectorXd biases;
        Eigen::VectorXd input_cache;
        Eigen::VectorXd output_cache;

        ActivationFunction activation_function;
    };

} // namespace neuralnet
