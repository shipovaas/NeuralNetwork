//#include <gtest/gtest.h>
#include "iostream"
#include "../src/Layer.h"
#include "NeuralNetwork.h"
#include "Eigen/Dense"
#include "../mnist/include/mnist/mnist_reader.hpp"

using namespace neuralnet;

int main() {
    // Загрузка данных MNIST
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");

    // Нормализация данных
    int input_size = 784;  // 28x28 пикселей
    int num_classes = 10;  // 10 классов цифр

    // Создание нейросети
    NeuralNetwork network;
    network.add_layer(Layer(input_size, 128, ActivationFunction::create(activation_type::ReLU)));
    network.add_layer(Layer(128, num_classes, ActivationFunction::create(activation_type::Sigmoid)));

    std::cout << "Network created with layers configured." << std::endl;

    // Обработка каждого образца индивидуально
    for (size_t i = 0; i < dataset.training_images.size(); ++i) {
        Eigen::VectorXd input = Eigen::VectorXd::Zero(input_size);
        for (size_t j = 0; j < input_size; ++j) {
            input(j) = dataset.training_images[i][j] / 255.0;
        }

        Eigen::VectorXd target = Eigen::VectorXd::Zero(num_classes);
        target(dataset.training_labels[i]) = 1.0;

        // Подготовка матрицы для одиночного ввода и цели
        Eigen::MatrixXd single_input = input;
        Eigen::MatrixXd single_target = target;

        // Обучение сети на одном образце
        network.fit(single_input, single_target, 1, 0.01);
    }

    // Оценка сети (простой пример, где мы используем тренировочный набор для оценки)
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
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}



// Тест линейного слоя
//TEST(LayerTest, ForwardPassLinear) {
//    Layer layer(1, 1, ActivationFunction(
//            [](double x) { return x; },  // Функция: f(x) = x
//            [](double x) { return 1; }   // Производная: f'(x) = 1
//    ));
//
//    Eigen::VectorXd input(1);
//    input << 1.0;
//
//    Eigen::VectorXd output = layer.forward(input);
//
//    ASSERT_NEAR(output[0], 1.0, 1e-9) << "The output should be equal to the input for a linear activation function.";
//}
//// Тест слоя с нулевым входом
//TEST(LayerTest, ForwardPassZeroInput) {
//    Layer layer(1, 1, ActivationFunction(
//            [](double x) { return x; },  // Функция
//            [](double x) { return 1; }   // Производная
//    ));
//
//    Eigen::VectorXd input(1);
//    input << 0.0;
//
//    Eigen::VectorXd output = layer.forward(input);
//
//    ASSERT_NEAR(output[0], 0.0, 1e-9) << "The output should be zero when the input is zero.";
//}


//отюда и вниз убрать комменты позже
//TEST(NN, nettest) {
//    neuralnet::NeuralNetwork net;
//    net.add_layer(Layer(1, 1, ActivationFunction(
//            [](double x) { return x; },  // Функция: f(x) = x
//            [](double x) { return 1; }   // Производная: f'(x) = 1
//    )));
//}
//
//int main(int argc, char **argv) {
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
//}