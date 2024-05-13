#include <gtest/gtest.h>
#include "Layer.h"
#include "ActivationFunction.h"
#include <eigen/Eigen/Dense>

// Использование пространства имен для удобства
using namespace neuralnet;

// Тестовый случай для проверки прямого распространения
TEST(LayerTest, ForwardPass) {
// Создание слоя с одним входом и одним выходом, без функции активации
 Layer layer(1, 1, ActivationFunction([](double x) { return x; }, [](double x) { return 1; }));

// Создание входного вектора
Eigen::VectorXd input(1);
input << 1.0;

// Выполнение прямого распространения
Eigen::VectorXd output = layer.forward(input);

// Проверка, что выход равен входу (так как функция активации - тождественная)
ASSERT_NEAR(output[0], 1.0, 1e-9);  // ASSERT_NEAR проверяет, что два значения близки с заданной погрешностью
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
