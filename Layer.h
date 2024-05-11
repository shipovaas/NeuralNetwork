#pragma once

#include "eigen/Eigen/Dense"
#include <fstream>
#include "ActivationFunction.h"

namespace neuralnet {

    class Layer {
    public:
        // Конструктор, принимающий размеры входа и выхода, а также функцию активации
        Layer(int inputSize, int outputSize, ActivationFunction activationFunc);

        // Метод для выполнения прямого распространения сигнала через слой
        Eigen::VectorXd forward(const Eigen::VectorXd& input);

        // Метод для выполнения обратного распространения ошибки и обновления весов
        Eigen::VectorXd backward(const Eigen::VectorXd& gradOutput, double learningRate);

        // Геттеры для доступа к весам и смещениям слоя
        const Eigen::MatrixXd& getWeights() const;
        const Eigen::VectorXd& getBiases() const;

        // Методы для сохранения и загрузки параметров слоя
        void save(std::ofstream& file) const;
        void load(std::ifstream& file);

    private:
        Eigen::MatrixXd weights;    // Матрица весов слоя
        Eigen::VectorXd biases;     // Вектор смещений слоя
        Eigen::VectorXd inputCache; // Кэш входных данных для использования в обратном распространении
        Eigen::VectorXd outputCache; // Кэш выходных данных для использования в обратном распространении

        ActivationFunction activationFunction; // Экземпляр функции активации
    };

} // namespace neuralnet
