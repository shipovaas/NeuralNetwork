#include "LossFunction.h"
#include <cmath>

LossFunction::LossFunction(const Vector &pred, Vector &result) : prediction(pred),real_result(result) {}

double LossFunction::Distance() const {
    double euclideanDistance = (prediction - real_result).norm();
    return sqrt(distance);
}
