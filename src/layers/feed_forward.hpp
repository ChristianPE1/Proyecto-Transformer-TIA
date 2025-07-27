#pragma once
#include "utils/matrix.hpp"
#include <cstddef>

class FeedForward {
public:
    FeedForward(std::size_t d_model, std::size_t d_ff);
    ~FeedForward() = default;
    Matrix forward(const Matrix &input);
    Matrix backward(const Matrix &grad_output, const Matrix &input);
    void updateWeights(float learning_rate);
    void initializeWeights();
    void saveWeights(std::ofstream& file);
    void loadWeights(std::ifstream& file);
private:
    std::size_t d_model;
    std::size_t d_ff;
    Matrix W1, W2;
    Matrix grad_W1, grad_W2;
    Matrix b1, b2;
    Matrix grad_b1, grad_b2;
};
