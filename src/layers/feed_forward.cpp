#include "feed_forward.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

Matrix FeedForward::forward(const Matrix &input) {
    int rows = input.getRows();
    int input_dim = input.getCols();
    int d_ff = this->d_ff;
    int output_dim = this->d_model;

    // Pure CPU implementation
    const std::vector<float> &input_h = input.getDataVector();
    const std::vector<float> &W1_h = W1.getDataVector();
    const std::vector<float> &W2_h = W2.getDataVector();
    const std::vector<float> &b1_h = b1.getDataVector();
    const std::vector<float> &b2_h = b2.getDataVector();

    std::vector<float> output_h(rows * output_dim);
    
    // Create intermediate activation matrix for the hidden layer
    std::vector<float> hidden(rows * d_ff);
    
    // First layer: input -> hidden (with ReLU)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < input_dim; ++k) {
                sum += input_h[i * input_dim + k] * W1_h[k * d_ff + j];
            }
            sum += b1_h[j];
            hidden[i * d_ff + j] = fmaxf(0.0f, sum); // ReLU activation
        }
    }
    
    // Second layer: hidden -> output (no activation)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_ff; ++k) {
                sum += hidden[i * d_ff + k] * W2_h[k * output_dim + j];
            }
            sum += b2_h[j];
            output_h[i * output_dim + j] = sum;
        }
    }
    
    return Matrix(output_h, rows, output_dim);
}

FeedForward::FeedForward(std::size_t d_model, std::size_t d_ff)
    : d_model(d_model), d_ff(d_ff), W1(d_model, d_ff), W2(d_ff, d_model),
      grad_W1(d_model, d_ff, 0.0f), grad_W2(d_ff, d_model, 0.0f),
      b1(1, d_ff, 0.0f), b2(1, d_model, 0.0f),
      grad_b1(1, d_ff, 0.0f), grad_b2(1, d_model, 0.0f) {
    initializeWeights();
}

void FeedForward::initializeWeights() {
    // Use proper seed
    srand(static_cast<unsigned>(time(nullptr)));
    
    // Initialize W1 weights with Xavier initialization
    int W1_size = d_model * d_ff;
    std::vector<float> W1_data(W1_size);
    float xavier_w1 = sqrt(2.0f / (d_model + d_ff));
    for (int i = 0; i < W1_size; ++i) {
        W1_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_w1;
    }
    W1 = Matrix(W1_data, d_model, d_ff);

    // Initialize W2 weights with Xavier initialization
    int W2_size = d_ff * d_model;
    std::vector<float> W2_data(W2_size);
    float xavier_w2 = sqrt(2.0f / (d_ff + d_model));
    for (int i = 0; i < W2_size; ++i) {
        W2_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_w2;
    }
    W2 = Matrix(W2_data, d_ff, d_model);

    // Biases are already initialized to zero in constructor
}

Matrix FeedForward::backward(const Matrix &grad_output, const Matrix &input) {
    int batch_size = grad_output.getRows();
    int output_dim = grad_output.getCols();
    int input_dim = input.getCols();
    
    // Initialize gradient for input
    Matrix grad_input(batch_size, input_dim, 0.0f);
    
    // Get data vectors
    const std::vector<float> &h_grad_output = grad_output.getDataVector();
    const std::vector<float> &h_input = input.getDataVector();
    
    // Get current weights for gradient computation
    const std::vector<float> &W1_data = W1.getDataVector();
    const std::vector<float> &W2_data = W2.getDataVector();
    
    // Initialize gradient accumulators
    std::vector<float> grad_W1_data(d_model * d_ff, 0.0f);
    std::vector<float> grad_W2_data(d_ff * d_model, 0.0f);
    std::vector<float> grad_b1_h(d_ff, 0.0f);
    std::vector<float> grad_b2_h(d_model, 0.0f);
    std::vector<float> grad_input_h(batch_size * input_dim, 0.0f);
    
    // Backward computation
    for (int b = 0; b < batch_size; ++b) {
        // Step 1: Compute intermediate values (W1 * input + b1)
        std::vector<float> z1(d_ff, 0.0f);  // Before ReLU
        std::vector<float> a1(d_ff, 0.0f);  // After ReLU
        
        for (int j = 0; j < d_ff; ++j) {
            for (int i = 0; i < input_dim; ++i) {
                z1[j] += W1_data[i * d_ff + j] * h_input[b * input_dim + i];
            }
            a1[j] = fmaxf(0.0f, z1[j]); // ReLU activation
        }
        
        // Step 2: Gradient of loss w.r.t b2 = grad_output
        for (int i = 0; i < d_model; ++i) {
            grad_b2_h[i] += h_grad_output[b * d_model + i];
        }
        
        // Step 3: Gradient of loss w.r.t W2
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                grad_W2_data[j * d_model + i] += a1[j] * h_grad_output[b * d_model + i];
            }
        }
        
        // Step 4: Gradient of loss w.r.t a1 (intermediate activation)
        std::vector<float> grad_a1(d_ff, 0.0f);
        for (int j = 0; j < d_ff; ++j) {
            for (int i = 0; i < d_model; ++i) {
                grad_a1[j] += W2_data[j * d_model + i] * h_grad_output[b * d_model + i];
            }
        }
        
        // Step 5: Gradient through ReLU (derivative is 1 if z1 > 0, else 0)
        std::vector<float> grad_z1(d_ff, 0.0f);
        for (int j = 0; j < d_ff; ++j) {
            grad_z1[j] = (z1[j] > 0.0f) ? grad_a1[j] : 0.0f;
        }
        
        // Step 6: Gradient w.r.t b1
        for (int j = 0; j < d_ff; ++j) {
            grad_b1_h[j] += grad_z1[j];
        }
        
        // Step 7: Gradient w.r.t W1
        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                grad_W1_data[i * d_ff + j] += h_input[b * input_dim + i] * grad_z1[j];
            }
        }
        
        // Step 8: Gradient w.r.t input
        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                grad_input_h[b * input_dim + i] += W1_data[i * d_ff + j] * grad_z1[j];
            }
        }
    }
    
    // Store gradients for weight updates
    grad_W1 = Matrix(grad_W1_data, d_model, d_ff);
    grad_W2 = Matrix(grad_W2_data, d_ff, d_model);
    grad_b1 = Matrix(grad_b1_h, 1, d_ff);
    grad_b2 = Matrix(grad_b2_h, 1, d_model);
    
    return Matrix(grad_input_h, batch_size, input_dim);
}

void FeedForward::updateWeights(float learning_rate) {
    if (learning_rate == 0.0f) {
        std::cout << "[FEEDFORWARD] WARNING: Learning rate is 0!" << std::endl;
        return;
    }
    
    // Update W1
    std::vector<float> &W1_data = W1.getDataVector();
    const std::vector<float> &grad_W1_data = grad_W1.getDataVector();
    
    // Check for NaN or inf in weights and gradients
    bool has_nan_w1 = false, has_nan_grad_w1 = false;
    for (size_t i = 0; i < W1_data.size(); ++i) {
        if (std::isnan(W1_data[i]) || std::isinf(W1_data[i])) has_nan_w1 = true;
        if (std::isnan(grad_W1_data[i]) || std::isinf(grad_W1_data[i])) has_nan_grad_w1 = true;
    }
    
    if (has_nan_w1 || has_nan_grad_w1) {
        std::cout << "[FEEDFORWARD] ERROR: NaN/Inf detected in W1 weights or gradients! Reinitializing..." << std::endl;
        // Reinitialize W1
        for (size_t i = 0; i < W1_data.size(); ++i) {
            W1_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        return;
    }
    
    // Apply gradient descent to W1
    for (size_t i = 0; i < W1_data.size(); ++i) {
        float clipped_grad = std::max(-2.0f, std::min(2.0f, grad_W1_data[i]));
        W1_data[i] -= learning_rate * clipped_grad;
        W1_data[i] = std::max(-1.0f, std::min(1.0f, W1_data[i]));
    }
    
    // Update W2
    std::vector<float> &W2_data = W2.getDataVector();
    const std::vector<float> &grad_W2_data = grad_W2.getDataVector();
    
    // Check for NaN or inf in W2
    bool has_nan_w2 = false, has_nan_grad_w2 = false;
    for (size_t i = 0; i < W2_data.size(); ++i) {
        if (std::isnan(W2_data[i]) || std::isinf(W2_data[i])) has_nan_w2 = true;
        if (std::isnan(grad_W2_data[i]) || std::isinf(grad_W2_data[i])) has_nan_grad_w2 = true;
    }
    
    if (has_nan_w2 || has_nan_grad_w2) {
        std::cout << "[FEEDFORWARD] ERROR: NaN/Inf detected in W2 weights or gradients! Reinitializing..." << std::endl;
        // Reinitialize W2
        for (size_t i = 0; i < W2_data.size(); ++i) {
            W2_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        return;
    }
    
    // Apply gradient descent to W2
    for (size_t i = 0; i < W2_data.size(); ++i) {
        float clipped_grad = std::max(-2.0f, std::min(2.0f, grad_W2_data[i]));
        W2_data[i] -= learning_rate * clipped_grad;
        W2_data[i] = std::max(-1.0f, std::min(1.0f, W2_data[i]));
    }
    
    // Update biases
    std::vector<float> &b1_h = b1.getDataVector();
    std::vector<float> &b2_h = b2.getDataVector();
    const std::vector<float> &grad_b1_h = grad_b1.getDataVector();
    const std::vector<float> &grad_b2_h = grad_b2.getDataVector();
    
    // Update b1
    for (size_t i = 0; i < b1_h.size(); ++i) {
        if (!std::isnan(grad_b1_h[i]) && !std::isinf(grad_b1_h[i])) {
            float clipped_grad = std::max(-2.0f, std::min(2.0f, grad_b1_h[i]));
            b1_h[i] -= learning_rate * clipped_grad;
            b1_h[i] = std::max(-1.0f, std::min(1.0f, b1_h[i]));
        }
    }
    
    // Update b2
    for (size_t i = 0; i < b2_h.size(); ++i) {
        if (!std::isnan(grad_b2_h[i]) && !std::isinf(grad_b2_h[i])) {
            float clipped_grad = std::max(-2.0f, std::min(2.0f, grad_b2_h[i]));
            b2_h[i] -= learning_rate * clipped_grad;
            b2_h[i] = std::max(-1.0f, std::min(1.0f, b2_h[i]));
        }
    }
    
    grad_W1.fill(0.0f);
    grad_W2.fill(0.0f);
    grad_b1.fill(0.0f);
    grad_b2.fill(0.0f);
}


void FeedForward::saveWeights(std::ofstream& file) {
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&d_ff), sizeof(size_t));
    Matrix::saveMatrix(file, W1);
    Matrix::saveMatrix(file, W2);
    Matrix::saveMatrix(file, b1);
    Matrix::saveMatrix(file, b2);
}

void FeedForward::loadWeights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&d_model), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&d_ff), sizeof(size_t));

    W1 = Matrix::loadMatrix(file);
    W2 = Matrix::loadMatrix(file);
    b1 = Matrix::loadMatrix(file);
    b2 = Matrix::loadMatrix(file);

    grad_W1 = Matrix(W1.getRows(), W1.getCols(), 0.0);
    grad_W2 = Matrix(W2.getRows(), W2.getCols(), 0.0);
    grad_b1 = Matrix(b1.getRows(), b1.getCols(), 0.0);
    grad_b2 = Matrix(b2.getRows(), b2.getCols(), 0.0);
}