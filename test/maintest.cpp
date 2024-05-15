#include "iostream"
#include "../src/Layer.h"
#include "NeuralNetwork.h"
#include "Eigen/Dense"
#include "../mnist/include/mnist/mnist_reader.hpp"

using namespace neuralnet;

int main() {

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");

    int input_size = 784;
    int num_classes = 10;
    int batch_size = 32;
    int epochs = 10;


    NeuralNetwork network;
    network.add_layer(Layer(input_size, 10, ActivationFunction::create(activation_type::ReLU)));
    //network.add_layer(Layer(128, num_classes, ActivationFunction::create(activation_type::Sigmoid)));

    Eigen::MatrixXd inputs(input_size, batch_size);
    Eigen::MatrixXd targets(num_classes, batch_size);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < dataset.training_images.size(); i += batch_size) {
            for (size_t j = 0; j < batch_size && (i + j) < dataset.training_images.size(); ++j) {
                size_t index = i + j;
                for (int k = 0; k < input_size; ++k) {
                    inputs(k, j) = dataset.training_images[index][k] / 255.0;
                }
                targets.setZero();
                targets(dataset.training_labels[index], j) = 1.0;
            }

            network.fit(inputs, targets, 1, 0.01);
        }
    }

    double correct_predictions = 0;
    for (size_t i = 0; i < dataset.training_images.size(); ++i) {
        Eigen::VectorXd input = Eigen::VectorXd::Zero(input_size);
        for (size_t j = 0; j < input_size; ++j) {
            input(j) = dataset.training_images[i][j] / 255.0;
        }

        Eigen::VectorXd predicted = network.predict(input);
        int predicted_label = std::distance(predicted.data(), std::max_element(predicted.data(), predicted.data() + predicted.size()));

        if (predicted_label == dataset.training_labels[i]) {
            correct_predictions += 1;
        }
    }

    double accuracy = correct_predictions / dataset.training_images.size();
    std::cout << "Training Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}