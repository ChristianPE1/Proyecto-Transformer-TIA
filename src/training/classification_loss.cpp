#include "classification_loss.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

float CrossEntropyLoss::compute_loss(const Matrix& predictions, const std::vector<int>& labels) {
    int num_samples = predictions.getRows();
    int num_classes = predictions.getCols();
    
    std::vector<float> pred_data = predictions.getDataVector();
    
    // Check for NaN/Inf in predictions
    bool has_invalid = false;
    for (float val : pred_data) {
        if (std::isnan(val) || std::isinf(val)) {
            has_invalid = true;
            break;
        }
    }
    
    if (has_invalid) {
        // Reset invalid values
        for (float& val : pred_data) {
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.0f;
            }
        }
    }
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < num_samples; i++) {
        // Apply softmax to predictions for this sample
        std::vector<float> logits(num_classes);
        for (int j = 0; j < num_classes; j++) {
            logits[j] = pred_data[i * num_classes + j];
        }
        
        // Find max for numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            logits[j] = std::exp(std::min(logits[j] - max_logit, 20.0f)); // Clip to prevent overflow
            sum_exp += logits[j];
        }
        
        // Prevent division by zero
        if (sum_exp < 1e-8f) {
            sum_exp = 1e-8f;
        }
        
        for (int j = 0; j < num_classes; j++) {
            logits[j] /= sum_exp;
        }
        
        // Compute cross-entropy loss
        int true_label = labels[i];
        if (true_label >= 0 && true_label < num_classes) {
            float prob = std::max(logits[true_label], 1e-8f);
            float loss = -std::log(prob);
            
            // Check for NaN in loss
            if (std::isnan(loss) || std::isinf(loss)) {
                loss = 10.0f; // Large but finite loss
            }
            
            total_loss += loss;
        }
    }
    
    float avg_loss = total_loss / num_samples;
    
    // Final NaN check
    if (std::isnan(avg_loss) || std::isinf(avg_loss)) {
        return 10.0f;
    }
    
    return avg_loss;
}

Matrix CrossEntropyLoss::compute_gradients(const Matrix& predictions, const std::vector<int>& labels) {
    int num_samples = predictions.getRows();
    int num_classes = predictions.getCols();
    
    std::vector<float> pred_data = predictions.getDataVector();
    
    // Check for NaN/Inf in predictions
    bool has_invalid = false;
    for (float val : pred_data) {
        if (std::isnan(val) || std::isinf(val)) {
            has_invalid = true;
            break;
        }
    }
    
    if (has_invalid) {
        std::cout << "[GRADIENTS] WARNING: NaN/Inf detected in predictions" << std::endl;
        // Reset invalid values
        for (float& val : pred_data) {
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.0f;
            }
        }
    }
    
    std::vector<float> grad_data(num_samples * num_classes, 0.0f);
    
    for (int i = 0; i < num_samples; i++) {
        // Apply softmax to predictions for this sample
        std::vector<float> logits(num_classes);
        for (int j = 0; j < num_classes; j++) {
            logits[j] = pred_data[i * num_classes + j];
        }
        
        // Find max for numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            logits[j] = std::exp(std::min(logits[j] - max_logit, 20.0f)); // Clip to prevent overflow
            sum_exp += logits[j];
        }
        
        // Prevent division by zero
        if (sum_exp < 1e-8f) {
            sum_exp = 1e-8f;
        }
        
        for (int j = 0; j < num_classes; j++) {
            logits[j] /= sum_exp;
        }
        
        // Compute gradients
        int true_label = labels[i];
        for (int j = 0; j < num_classes; j++) {
            if (j == true_label) {
                grad_data[i * num_classes + j] = logits[j] - 1.0f;
            } else {
                grad_data[i * num_classes + j] = logits[j];
            }
            
            // Clip gradients to prevent explosion
            grad_data[i * num_classes + j] = std::max(-10.0f, std::min(10.0f, grad_data[i * num_classes + j]));
            
            // Check for NaN/Inf in gradients
            if (std::isnan(grad_data[i * num_classes + j]) || std::isinf(grad_data[i * num_classes + j])) {
                grad_data[i * num_classes + j] = 0.0f;
            }
        }
    }
    
    Matrix gradients(grad_data, num_samples, num_classes);
    return gradients;
}
