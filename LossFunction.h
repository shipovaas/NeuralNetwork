#pragma once
#include "eigen/Eigen/Dense"
#include <iostream>
#include <vector>
#include <functional>
#include <cassert>
#include <stdexcept>

namespace neuralnet {

    enum class LossType { MSE, Manhattan, CrossEntropy };

    class LossFunction {
    public:
        using Func = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
        using GradFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

        LossFunction(Func lossFunc, GradFunc gradFunc); // Конструктор с параметрами функций потерь и градиента
        static LossFunction Create(LossType type); // Статический метод для создания функций потерь

        double Calculate(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) const; // Вычисление потерь
        Eigen::VectorXd CalculateGradient(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) const; // Вычисление градиента потерь

    private:
        Func loss_; // Функциональный объект для вычисления потерь
        GradFunc grad_; // Функциональный объект для вычисления градиента
    };

} // namespace neuralnet

