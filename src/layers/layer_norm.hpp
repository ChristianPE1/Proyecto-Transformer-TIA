
#pragma once
#include <cstddef>
#include "utils/matrix.hpp"

class LayerNorm {
public:
    LayerNorm(std::size_t d_model, double epsilon = 1e-6);
    Matrix forward(const Matrix &input);
    Matrix backward(const Matrix &grad_output, const Matrix &input);
    void updateWeights(float learning_rate);
private:
    std::size_t d_model;
    double epsilon;
    Matrix gamma;
    Matrix beta;
    Matrix grad_gamma;
    Matrix grad_beta;
};
