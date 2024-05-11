#pragma once

#include "eigen/Eigen/Dense"
#include <cmath>
#include <functional>

namespace neuralnet {

    using DataType = double;  // Определение базового типа данных
    using Vector = Eigen::VectorXd;  // Использование типа Vector из Eigen
    using Matrix = Eigen::MatrixXd;  // Использование типа Matrix из Eigen

    enum class ActivationType { Sigmoid, Tanh, ReLU, Linear, Softmax }; // Перечисление типов функций активации

    class ActivationFunction {
    public:
        using Function = std::function<DataType(DataType)>; // Определение типа функции для активации

        ActivationFunction(Function func, Function deriv); // Конструктор с параметрами функций

        DataType calc(DataType x) const; // Вычисление значения функции активации
        DataType derivative(DataType x) const; // Вычисление производной функции активации
        Vector calc(const Vector& x) const; // Вычисление значения функции активации для вектора
        Matrix derivative(const Vector& x) const; // Вычисление матрицы производных для вектора

        static ActivationFunction create(ActivationType type); // Создание экземпляра функции активации

    private:
        Function func_; // Функция активации
        Function deriv_; // Производная функции активации
    };

} // namespace neuralnet
