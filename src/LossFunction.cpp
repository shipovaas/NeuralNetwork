#include "LossFunction.h"
#include <cmath>

namespace neuralnet {

    LossFunction::LossFunction(Func loss_func, GradFunc grad_func)
            : loss_(std::move(loss_func)), grad_(std::move(grad_func)) {}

    LossFunction LossFunction::Create(LossType type) {
        switch (type) {
            case LossType::MSE:
                return LossFunction(
                        [](const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) {
                            return (predicted - target).squaredNorm();
                        },
                        [](const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) {
                            return 2.0 * (predicted - target);
                        });
            case LossType::Manhattan:
                return LossFunction(
                        [](const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) {
                            return (predicted - target).lpNorm<1>();
                        },
                        [](const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) {
                            return (predicted - target).unaryExpr([](double x) { return x > 0 ? 1.0 : -1.0; });
                        });
            case LossType::CrossEntropy:
                return LossFunction(
                        [](const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) {
                            return -(target.array() * (predicted.array() + 1e-7).log()).sum();
                        },
                        [](const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) {
                            return -(target.array() / (predicted.array() + 1e-7));
                        });
            default:
                throw std::invalid_argument("Unsupported loss type");
        }
    }

    double LossFunction::Calculate(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) const {
        return loss_(predicted, target);
    }

    Eigen::VectorXd LossFunction::CalculateGradient(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) const {
        return grad_(predicted, target);
    }

} // namespace neuralnet
