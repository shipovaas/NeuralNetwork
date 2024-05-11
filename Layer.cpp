#include "Layer.h"

namespace neuralnet {

    Layer::Layer(int inputSize, int outputSize, ActivationFunction activationFunc)
            : weights(Eigen::MatrixXd::Random(outputSize, inputSize)),
              biases(Eigen::VectorXd::Zero(outputSize)),
              activationFunction(std::move(activationFunc)) {}

    Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {
        inputCache = input; // Store input for use in the backward pass
        Eigen::VectorXd z = weights * input + biases;
        // Используйте calc для применения функции активации
        outputCache = z.unaryExpr([this](double val) { return activationFunction.calc(val); });
        return outputCache;
    }

    Eigen::VectorXd Layer::backward(const Eigen::VectorXd& gradOutput, double learningRate) {
        // Примените производную функции активации к выходному градиенту
        Eigen::VectorXd gradInput = weights.transpose() * gradOutput.cwiseProduct(outputCache.unaryExpr([this](double val) { return activationFunction.derivative(val); }));

        // Градиенты по весам и смещениям с использованием кэшированного входа
        Eigen::MatrixXd gradWeights = gradInput * inputCache.transpose();
        Eigen::VectorXd gradBiases = gradInput;

        // Обновление весов и смещений
        weights -= learningRate * gradWeights;
        biases -= learningRate * gradBiases;

        // Вычисление градиента по входу для предыдущего слоя
        return weights.transpose() * gradInput;
    }

    void Layer::save(std::ofstream& file) const {
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));
    }

    void Layer::load(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(double));
        file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(double));
    }

    const Eigen::MatrixXd& Layer::getWeights() const {
        return weights;
    }

    const Eigen::VectorXd& Layer::getBiases() const {
        return biases;
    }

} // namespace neuralnet
