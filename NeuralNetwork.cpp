#include "NeuralNetwork.h"
#include <fstream>

namespace neuralnet {

    NeuralNetwork::NeuralNetwork() {}

    void NeuralNetwork::addLayer(Layer* layer) {
        layers.push_back(layer);
    }

    Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd& input) {
        Eigen::VectorXd output = input;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void NeuralNetwork::fit(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < inputs.cols(); ++i) {
                Eigen::VectorXd input = inputs.col(i);
                Eigen::VectorXd target = targets.col(i);
                Eigen::VectorXd output = predict(input);

                Eigen::VectorXd error = target - output;
                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    error = (*it)->backward(error, learningRate);
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
        return loss / inputs.cols();
    }

    void NeuralNetwork::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        for (auto& layer : layers) {
            // Assuming Layer has a method to serialize its parameters
            layer->save(file);
        }
    }

    void NeuralNetwork::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        for (auto& layer : layers) {
            // Assuming Layer has a method to deserialize its parameters
            layer->load(file);
        }
    }

} // namespace neuralnet
