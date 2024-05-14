//#include <gtest/gtest.h>
#include "iostream"
#include "../src/Layer.h"
#include "NeuralNetwork.h"
#include "Eigen/Dense"
#include "../mnist/include/mnist/mnist_reader.hpp"


using namespace neuralnet;
int main() {
    // Загрузка данных MNIST
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>( "../../mnist");

    // Нормализация и подготовка данных
    int input_size = 784;  // 28x28 пикселей
    int num_classes = 10;  // 10 классов цифр
    Eigen::MatrixXd inputs(input_size, dataset.training_images.size());
    Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(num_classes, dataset.training_images.size());

    for (size_t i = 0; i < dataset.training_images.size(); ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            inputs(j, i) = dataset.training_images[i][j] / 255.0;
        }
        targets(dataset.training_labels[i], i) = 1.0;
    }

// Создание нейросети
// Создание нейросети
    neuralnet::NeuralNetwork network;
    network.add_layer(neuralnet::Layer(784, 128, neuralnet::ActivationFunction::create(neuralnet::activation_type::ReLU)));
    network.add_layer(neuralnet::Layer(128, 10, neuralnet::ActivationFunction::create(neuralnet::activation_type::Sigmoid)));

    std::cout << "Network created with layers configured." << std::endl;
// Убедитесь, что inputs и targets имеют правильные размеры перед передачей их в сеть
    std::cout << "Inputs size: " << inputs.rows() << "x" << inputs.cols() << std::endl;
    std::cout << "Targets size: " << targets.rows() << "x" << targets.cols() << std::endl;

// Здесь должны быть вызовы функций обучения или предсказания
    network.fit(inputs, targets, 10, 0.01);

// Обучение
    network.fit(inputs, targets, 10, 0.01);  // 10 эпох, скорость обучения 0.01

// Оценка
    double accuracy = network.evaluate(inputs, targets);
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