#include "linear.hpp"
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>
#include <iostream>

Matrix Linear::forward(const Matrix &input) {
    int batch_size = input.getRows();
    int input_dim = input.getCols();
    int output_dim = weights.getCols();

    // Pure CPU implementation
    const std::vector<float> &h_input = input.getDataVector();
    const std::vector<float> &h_weights = weights.getDataVector();
    const std::vector<float> &h_bias = bias.getDataVector();

    std::vector<float> h_output(batch_size * output_dim, 0.0f);

    // CPU matrix multiplication: output = input * weights + bias
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                sum += h_input[b * input_dim + i] * h_weights[i * output_dim + o];
            }
            h_output[b * output_dim + o] = sum + h_bias[o];
        }
    }

    Matrix output(h_output, batch_size, output_dim);
    return output;
}

// Backward pass for Linear layer
Matrix Linear::backward(const Matrix& grad_output, const Matrix& input) {
    int batch_size = grad_output.getRows();
    int output_dim = grad_output.getCols();
    
    // Compute gradients for weights and bias
    Matrix grad_weights(input_dim, output_dim, 0.0f);
    Matrix grad_bias(1, output_dim, 0.0f);
    Matrix grad_input(batch_size, input_dim, 0.0f);
    
    // Get references to data vectors for CPU computation
    const std::vector<float> &h_grad_output = grad_output.getDataVector();
    const std::vector<float> &h_input = input.getDataVector();
    
    std::vector<float> h_grad_weights(input_dim * output_dim, 0.0f);
    std::vector<float> h_grad_bias(output_dim, 0.0f);
    std::vector<float> h_grad_input(batch_size * input_dim, 0.0f);
    
    // Compute gradients
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_dim; ++i) {
            for (int o = 0; o < output_dim; ++o) {
                // Gradient w.r.t weights: grad_w = input^T * grad_output
                h_grad_weights[i * output_dim + o] += h_input[b * input_dim + i] * h_grad_output[b * output_dim + o];
            }
        }
        
        // Gradient w.r.t bias: grad_b = sum(grad_output)
        for (int o = 0; o < output_dim; ++o) {
            h_grad_bias[o] += h_grad_output[b * output_dim + o];
        }
    }
    
    // Compute gradient w.r.t input: grad_input = grad_output * W^T
    const std::vector<float> &h_weights = weights.getDataVector();
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_dim; ++i) {
            for (int o = 0; o < output_dim; ++o) {
                h_grad_input[b * input_dim + i] += h_grad_output[b * output_dim + o] * h_weights[i * output_dim + o];
            }
        }
    }
    
    // Store gradients for weight update
    stored_grad_weights = Matrix(h_grad_weights, input_dim, output_dim);
    stored_grad_bias = Matrix(h_grad_bias, 1, output_dim);
    
    return Matrix(h_grad_input, batch_size, input_dim);
}

// Update weights using stored gradients
void Linear::updateWeights(float learning_rate) {
    // Check if we have valid gradients stored
    if (stored_grad_weights.getRows() == 0 || stored_grad_weights.getCols() == 0 || 
        stored_grad_bias.getRows() == 0 || stored_grad_bias.getCols() == 0) {
        // No gradients stored, skip update
        return;
    }
    
    // Verify dimensions match
    if (stored_grad_weights.getRows() != input_dim || stored_grad_weights.getCols() != output_dim) {
        return;
    }
    
    // Get references to data vectors
    std::vector<float> &h_weights = weights.getDataVector();
    std::vector<float> &h_bias = bias.getDataVector();
    
    const std::vector<float> &h_grad_weights = stored_grad_weights.getDataVector();
    const std::vector<float> &h_grad_bias = stored_grad_bias.getDataVector();
    
    // Verify we have valid gradient data
    if (h_grad_weights.empty() || h_grad_bias.empty()) {
        return;
    }
    
    // Check for NaN/Inf gradients and clip them
    bool has_nan_weights = false, has_nan_bias = false;
    float max_grad_weight = 0.0f, max_grad_bias = 0.0f;
    
    for (size_t i = 0; i < h_grad_weights.size(); ++i) {
        if (std::isnan(h_grad_weights[i]) || std::isinf(h_grad_weights[i])) {
            has_nan_weights = true;
        } else {
            max_grad_weight = std::max(max_grad_weight, std::abs(h_grad_weights[i]));
        }
    }
    
    for (size_t i = 0; i < h_grad_bias.size(); ++i) {
        if (std::isnan(h_grad_bias[i]) || std::isinf(h_grad_bias[i])) {
            has_nan_bias = true;
        } else {
            max_grad_bias = std::max(max_grad_bias, std::abs(h_grad_bias[i]));
        }
    }
    
    if (has_nan_weights || has_nan_bias) {
        std::cout << "[LINEAR] WARNING: Detected NaN/Inf gradients, skipping update" << std::endl;
        return;
    }
    
    // Apply gradient descent with adaptive learning rate
    float effective_lr = learning_rate;
    
    // For output projection layers, use higher learning rate to break symmetry
    if (output_dim > 500) {
        effective_lr *= 8.0f;
    }
    
    if (max_grad_weight > 0.5f || max_grad_bias > 0.5f) {
        effective_lr *= 0.5f;
    }
    
    // Apply gradient descent: w = w - lr * grad_w
    for (size_t i = 0; i < h_weights.size(); ++i) {
        h_weights[i] -= effective_lr * h_grad_weights[i];
        h_weights[i] = std::max(-5.0f, std::min(5.0f, h_weights[i]));
    }
    
    for (size_t i = 0; i < h_bias.size(); ++i) {
        h_bias[i] -= effective_lr * h_grad_bias[i];
        h_bias[i] = std::max(-3.0f, std::min(3.0f, h_bias[i]));
    }
    
    // Reset gradients to zero
    stored_grad_weights.fill(0.0f);
    stored_grad_bias.fill(0.0f);
}

// Constructor
Linear::Linear(size_t input_dim, size_t output_dim) 
    : input_dim(input_dim), output_dim(output_dim), 
      weights(input_dim, output_dim), bias(1, output_dim) {
    // Initialize stored gradients with proper dimensions but zero values
    stored_grad_weights = Matrix(input_dim, output_dim, 0.0f);
    stored_grad_bias = Matrix(1, output_dim, 0.0f);
    initialize();
}


Linear::~Linear() {}

// Initialize weights and bias
void Linear::initialize() {
    // Initialize weights with Xavier/Glorot initialization
    std::vector<float> weight_data(input_dim * output_dim);
    std::vector<float> bias_data(output_dim, 0.0f);
    
    // Use Xavier initialization
    float scale = sqrt(2.0f / (input_dim + output_dim));
    
    // For output projection layers, use larger scale
    if (output_dim > 500) {
        scale *= 3.0f;
        std::cout << "[LINEAR] Output projection layer detected, using LARGER scale for better variance" << std::endl;
    }
    
    // Reasonable scale range
    scale = std::max(0.01f, std::min(scale, 0.2f));
    
    // Random initialization for weights
    srand(time(nullptr));
    for (size_t i = 0; i < weight_data.size(); ++i) {
        float rand_val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        weight_data[i] = rand_val;
    }
    
    std::cout << "[LINEAR] Initialized " << input_dim << "x" << output_dim 
              << " with scale=" << scale << std::endl;
              
    weights = Matrix(weight_data, input_dim, output_dim);
    bias = Matrix(bias_data, 1, output_dim);
}

void Linear::saveWeights(std::ofstream& file) {
    file.write(reinterpret_cast<const char*>(&input_dim), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&output_dim), sizeof(size_t));
    Matrix::saveMatrix(file, weights);
    Matrix::saveMatrix(file, bias);
}

void Linear::loadWeights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&input_dim), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&output_dim), sizeof(size_t));
    weights = Matrix::loadMatrix(file);
    bias = Matrix::loadMatrix(file);
    stored_grad_weights = Matrix(weights.getRows(), weights.getCols(), 0.0);
    stored_grad_bias = Matrix(bias.getRows(), bias.getCols(), 0.0);
}