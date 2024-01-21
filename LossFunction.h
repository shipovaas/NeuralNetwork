#pragma once

#include <iostream>
#include <vector>


class LossFunction {
public:
    LossFunction(const std::vector<double> &pred, std::vector<double> &result);

    double Distance() const;

private:
    std::vector<double> prediction;
    std::vector<double> real_result;
};