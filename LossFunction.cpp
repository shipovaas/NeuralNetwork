#include "LossFunction.h"
#include <cmath>

LossFunction::LossFunction(const Vector &pred, Vector &result) : prediction(pred),real_result(result) {}

double LossFunction::Distance() const {
    double distance = 0.0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        distance += pow(prediction[i] - real_result[i], 2);
    };
    return sqrt(distance);
}
