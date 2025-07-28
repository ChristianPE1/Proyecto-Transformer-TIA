#include "layer_norm.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>

LayerNorm::LayerNorm(std::size_t d_model, double epsilon) 
    : d_model(d_model), epsilon(epsilon), gamma(1, d_model, 1.0f), beta(1, d_model, 0.0f),
      grad_gamma(1, d_model, 0.0f), grad_beta(1, d_model, 0.0f) {
    // Use a much larger epsilon for numerical stability and to prevent activation collapse
    if (epsilon < 1e-3) {
        this->epsilon = 1e-3;  // Much larger epsilon to prevent over-normalization
    }
}

Matrix LayerNorm::forward(const Matrix &input) {
    int N = input.getRows();
    int D = input.getCols();

    const std::vector<float> &h_input = input.getDataVector();
    std::vector<float> h_output(N * D);

    // Get gamma and beta values
    const std::vector<float> &h_gamma = gamma.getDataVector();
    const std::vector<float> &h_beta = beta.getDataVector();

    // CPU LayerNorm implementation
    for (int i = 0; i < N; i++) {
        // Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < D; j++) {
            mean += h_input[i * D + j];
        }
        mean /= D;

        // Calculate variance
        float variance = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = h_input[i * D + j] - mean;
            variance += diff * diff;
        }
        variance /= D;

        // CRITICAL FIX: Protect against division by zero and ensure epsilon is meaningful
        float stddev = sqrt(variance + epsilon);
        
        // Additional protection: if stddev is still too small, use a minimum value
        if (stddev < 1e-3f) {
            stddev = 1e-3f;  // Larger minimum to prevent over-normalization
        }

        // Normalize and apply gamma and beta
        for (int j = 0; j < D; j++) {
            float normalized = (h_input[i * D + j] - mean) / stddev;
            h_output[i * D + j] = h_gamma[j] * normalized + h_beta[j];
        }
    }

    return Matrix(h_output, N, D);
}

Matrix LayerNorm::backward(const Matrix &grad_output, const Matrix &input) {
    int N = input.getRows();
    int D = input.getCols();
    
    // CPU implementation for simplicity
    const std::vector<float> &h_input = input.getDataVector();
    const std::vector<float> &h_grad_output = grad_output.getDataVector();
    std::vector<float> h_grad_input(N * D, 0.0f);
    const std::vector<float> &h_gamma = gamma.getDataVector();
    std::vector<float> h_grad_gamma(D, 0.0f);
    std::vector<float> h_grad_beta(D, 0.0f);
    
    for (int i = 0; i < N; i++) {
        // Calculate mean and variance
        float mean = 0.0f;
        for (int j = 0; j < D; j++) {
            mean += h_input[i * D + j];
        }
        mean /= D;
        
        float variance = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = h_input[i * D + j] - mean;
            variance += diff * diff;
        }
        variance /= D;
        
        float stddev = sqrt(variance + epsilon);
        
        // Compute gradients
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        
        for (int j = 0; j < D; j++) {
            float xhat = (h_input[i * D + j] - mean) / stddev;
            float dy = h_grad_output[i * D + j];
            
            sum_dy += dy * h_gamma[j];
            sum_dy_xhat += dy * h_gamma[j] * xhat;
            
            // Accumulate parameter gradients
            h_grad_gamma[j] += dy * xhat;
            h_grad_beta[j] += dy;
        }
        
        // Compute input gradients
        for (int j = 0; j < D; j++) {
            float xhat = (h_input[i * D + j] - mean) / stddev;
            float dy = h_grad_output[i * D + j];
            
            h_grad_input[i * D + j] = (h_gamma[j] / stddev) * 
                (dy - sum_dy / D - xhat * sum_dy_xhat / D);
        }
    }
    
    grad_gamma = Matrix(h_grad_gamma, 1, D);
    grad_beta = Matrix(h_grad_beta, 1, D);
    
    return Matrix(h_grad_input, N, D);
}

void LayerNorm::updateWeights(float learning_rate) {
    // Update gamma and beta using accumulated gradients
    std::vector<float> &h_gamma = gamma.getDataVector();
    std::vector<float> &h_beta = beta.getDataVector();
    const std::vector<float> &h_grad_gamma = grad_gamma.getDataVector();
    const std::vector<float> &h_grad_beta = grad_beta.getDataVector();
    
    for (size_t i = 0; i < d_model; i++) {
        h_gamma[i] -= learning_rate * h_grad_gamma[i];
        h_beta[i] -= learning_rate * h_grad_beta[i];
    }
    
    // Reset gradients
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
}

void LayerNorm::saveWeights(std::ofstream& file) {
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(double));
    Matrix::saveMatrix(file, gamma);
    Matrix::saveMatrix(file, beta);
}

void LayerNorm::loadWeights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&d_model), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(double));
    gamma = Matrix::loadMatrix(file);
    beta = Matrix::loadMatrix(file);
    grad_gamma = Matrix(gamma.getRows(), gamma.getCols(), 0.0);
    grad_beta = Matrix(beta.getRows(), beta.getCols(), 0.0);
}