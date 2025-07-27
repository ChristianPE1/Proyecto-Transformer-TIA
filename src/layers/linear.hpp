#pragma once
#include "utils/matrix.hpp"
#include <cstddef>

class Linear {
private:
    Matrix weights;
    Matrix bias;
    size_t input_dim;
    size_t output_dim;
    Matrix stored_grad_weights;
    Matrix stored_grad_bias;
public:
    Linear(size_t input_dim, size_t output_dim);
    Matrix forward(const Matrix &input);
    Matrix backward(const Matrix &grad_output, const Matrix &input);
    void updateWeights(float learning_rate);
    void initialize();
    void saveWeights(std::ofstream& file);
    void loadWeights(std::ifstream& file);
    ~Linear();
};
