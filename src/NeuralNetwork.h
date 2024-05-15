#pragma once

#include <vector>
#include <memory>
#include "Layer.h"

namespace neuralnet {

    class NeuralNetwork {
    public:
        NeuralNetwork() = default;

        void add_layer(Layer&& layer);
        Eigen::VectorXd predict(const Eigen::VectorXd& input);

        void fit(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, int epochs, double learning_rate);
        double evaluate(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets);

        void save(const std::string& filename) const;
        void load(const std::string& filename);

    private:
        std::vector<Layer> layers;
    };

} // namespace neuralnet
