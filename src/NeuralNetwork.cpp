#include "NeuralNetwork.h"
#include <fstream>
#include <memory>
#include <ranges>
#include <vector>

namespace neuralnet {

//    NeuralNetwork::NeuralNetwork() {}

    void NeuralNetwork::add_layer(Layer&& layer) {
        layers.push_back(std::move(layer));
    }

    Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd& input) {
        Eigen::VectorXd output = input;
        for (auto& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    void NeuralNetwork::fit(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < inputs.cols(); ++i) {
                Eigen::VectorXd input = inputs.col(i);
                Eigen::VectorXd target = targets.col(i);
                Eigen::VectorXd output = predict(input);

                Eigen::VectorXd error = target - output;
                for (auto & layer : layers) {
                    error = layer.backward(error, learning_rate);
                }
            }
        }
    }

    double NeuralNetwork::evaluate(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets) {
        double loss = 0.0;
        for (int i = 0; i < inputs.cols(); ++i) {
            Eigen::VectorXd output = predict(inputs.col(i));
            loss += (targets.col(i) - output).squaredNorm();
        }
        return loss / static_cast<double>(inputs.cols());
    }

    void NeuralNetwork::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file for writing.");
        }
        for (auto& layer : layers) {
            layer.save(file);
        }
    }

    void NeuralNetwork::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file for reading.");
        }
        for (auto& layer : layers) {
            layer.load(file);
        }
    }

} // namespace neuralnet
