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

        LossFunction(Func loss_func, GradFunc grad_func);
        static LossFunction Create(LossType type);

        double Calculate(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) const;
        Eigen::VectorXd CalculateGradient(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) const;

    private:
        Func loss_;
        GradFunc grad_;
    };

} // namespace neuralnet
