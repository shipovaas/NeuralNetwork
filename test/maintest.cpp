#include <gtest/gtest.h>
#include "../src/Layer.h"
#include "../src/ActivationFunction.h"
#include "NeuralNetwork.h"
#include "Eigen/Dense"

#include "iostream"

using namespace neuralnet;

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

TEST(NN, nettest) {
    neuralnet::NeuralNetwork net;
    net.add_layer(Layer(1, 1, ActivationFunction(
            [](double x) { return x; },  // Функция: f(x) = x
            [](double x) { return 1; }   // Производная: f'(x) = 1
    )));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}