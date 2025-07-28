#pragma once
#include "utils/matrix.hpp"
#include <vector>
#include <cstddef>
#include <fstream>

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t n_heads);
    ~MultiHeadAttention();
    Matrix forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask = Matrix());
    Matrix backward(const Matrix &grad_output, const Matrix &query, const Matrix &key, const Matrix &value);
    void backward(const Matrix &grad_output, Matrix &grad_query, Matrix &grad_key, Matrix &grad_value);
    void updateWeights(float learning_rate);
    void updateGradients(const Matrix &grad_output, const Matrix &query, const Matrix &key, const Matrix &value);
    void saveWeights(std::ofstream& file);
    void loadWeights(std::ifstream& file);
private:
    void initializeWeights();
    size_t d_model;
    size_t n_heads;
    size_t d_k;
    size_t d_v;
    Matrix W_Q;
    Matrix W_K;
    Matrix W_V;
    Matrix W_O;
    Matrix grad_W_Q;
    Matrix grad_W_K;
    Matrix grad_W_V;
    Matrix grad_W_O;
    Matrix last_query, last_key, last_value;
    Matrix last_Q, last_K, last_V;  // Q, K, V proyectadas
    Matrix last_attention_weights;
    Matrix last_attention_output;   // Salida antes de W_O
};
