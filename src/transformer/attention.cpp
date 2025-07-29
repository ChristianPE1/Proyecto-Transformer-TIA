#include "attention.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads)
    : d_model(d_model), n_heads(n_heads), d_k(d_model / n_heads), d_v(d_model / n_heads),
      W_Q(d_model, d_model), W_K(d_model, d_model), W_V(d_model, d_model), W_O(d_model, d_model),
      grad_W_Q(d_model, d_model, 0.0f), grad_W_K(d_model, d_model, 0.0f), grad_W_V(d_model, d_model, 0.0f), grad_W_O(d_model, d_model, 0.0f) {
    
    // Inicializar pesos con Xavier/Glorot
    initializeWeights();
}

void MultiHeadAttention::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Inicialización mucho más conservadora para evitar explosión de gradientes
    float std_dev = std::sqrt(1.0f / (float)d_model) * 0.1f;  // Factor de 0.1 para mayor estabilidad
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    // Inicializar W_Q
    std::vector<float> wq_data(d_model * d_model);
    for (auto& w : wq_data) w = dist(gen);
    W_Q = Matrix(wq_data, d_model, d_model);
    
    // Inicializar W_K
    std::vector<float> wk_data(d_model * d_model);
    for (auto& w : wk_data) w = dist(gen);
    W_K = Matrix(wk_data, d_model, d_model);
    
    // Inicializar W_V
    std::vector<float> wv_data(d_model * d_model);
    for (auto& w : wv_data) w = dist(gen);
    W_V = Matrix(wv_data, d_model, d_model);
    
    // Inicializar W_O con valores aún más pequeños
    std::vector<float> wo_data(d_model * d_model, 0.0f);
    // Solo inicializar la diagonal para empezar con transformación casi identidad
    for (size_t i = 0; i < d_model && i < d_model; i++) {
        wo_data[i * d_model + i] = 0.1f;
    }
    W_O = Matrix(wo_data, d_model, d_model);
}

MultiHeadAttention::~MultiHeadAttention() {}

Matrix MultiHeadAttention::forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask) {
    // Implementacion de self-attention CON matrices de pesos entrenables
    int seq_len = query.getRows();
    int d_model_int = query.getCols();
    
    // Paso 1: Proyectar entrada a Q, K, V usando matrices de pesos
    Matrix Q = query.multiply(W_Q);
    Matrix K = key.multiply(W_K);
    Matrix V = value.multiply(W_V);
    
    // Paso 2: Calcular scores = Q * K**T / sqrt(d_k)
    Matrix scores(seq_len, seq_len, 0.0f);
    const std::vector<float> &q_data = Q.getDataVector();
    const std::vector<float> &k_data = K.getDataVector();
    
    float scale = 1.0f / std::sqrt((float)d_model_int);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float score = 0.0f;
            for (int k = 0; k < d_model_int; k++) {
                score += q_data[i * d_model_int + k] * k_data[j * d_model_int + k];
            }
            score *= scale;
            scores.setElement(i, j, score);
        }
    }
    
    // Paso 3: Aplicar softmax a scores
    std::vector<float> scores_data = scores.getDataVector();
    for (int i = 0; i < seq_len; i++) {
        // Encontrar max para estabilidad numérica
        float max_score = scores_data[i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            max_score = std::max(max_score, scores_data[i * seq_len + j]);
        }
        
        // Aplicar exp y sumar
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float exp_val = std::exp(scores_data[i * seq_len + j] - max_score);
            scores_data[i * seq_len + j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalizar
        if (sum_exp > 1e-8f) {
            for (int j = 0; j < seq_len; j++) {
                scores_data[i * seq_len + j] /= sum_exp;
            }
        } else {
            // Si sum_exp es muy pequeño, usar atención uniforme
            float uniform_attention = 1.0f / seq_len;
            for (int j = 0; j < seq_len; j++) {
                scores_data[i * seq_len + j] = uniform_attention;
            }
        }
    }
    
    // Paso 4: Aplicar atención a V
    Matrix attention_output(seq_len, d_model_int, 0.0f);
    const std::vector<float> &v_data = V.getDataVector();
    
    for (int i = 0; i < seq_len; i++) {
        for (int k = 0; k < d_model_int; k++) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                weighted_sum += scores_data[i * seq_len + j] * v_data[j * d_model_int + k];
            }
            attention_output.setElement(i, k, weighted_sum);
        }
    }
    
    // Paso 5: Proyección final con W_O
    Matrix output = attention_output.multiply(W_O);
    
    // Guardar estados para backward
    last_query = query;  // Entrada original
    last_key = key;
    last_value = value;
    last_Q = Q;         // Q proyectada
    last_K = K;         // K proyectada
    last_V = V;         // V proyectada
    last_attention_weights = Matrix(scores_data, seq_len, seq_len);
    last_attention_output = attention_output;
    
    return output;
}

Matrix MultiHeadAttention::backward(const Matrix &grad_output, const Matrix &query, const Matrix &key, const Matrix &value) {
    // Implementación mínima para evitar error de enlazado
    return Matrix(query.getRows(), query.getCols(), 0.0f);
}

void MultiHeadAttention::backward(const Matrix &grad_output, Matrix &grad_query, Matrix &grad_key, Matrix &grad_value) {
    // Backward completo de self-attention con gradientes correctos
    int seq_len = grad_output.getRows();
    int d_model_int = grad_output.getCols();
    
    // Paso 1: Gradiente de W_O
    // grad_W_O += last_attention_output^T * grad_output
    const std::vector<float> &attn_out_data = last_attention_output.getDataVector();
    const std::vector<float> &grad_out_data = grad_output.getDataVector();
    std::vector<float> grad_wo_data = grad_W_O.getDataVector();
    
    for (int i = 0; i < d_model_int; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                grad_sum += attn_out_data[k * d_model_int + i] * grad_out_data[k * d_model_int + j];
            }
            grad_wo_data[i * d_model_int + j] += grad_sum;
        }
    }
    grad_W_O = Matrix(grad_wo_data, d_model_int, d_model_int);
    
    // Paso 2: Gradiente hacia attention_output
    // grad_attention_output = grad_output * W_O**T
    Matrix grad_attention_output(seq_len, d_model_int, 0.0f);
    const std::vector<float> &wo_data = W_O.getDataVector();
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < d_model_int; k++) {
                grad_sum += grad_out_data[i * d_model_int + k] * wo_data[j * d_model_int + k];
            }
            grad_attention_output.setElement(i, j, grad_sum);
        }
    }
    
    // Paso 3: Gradiente hacia V
    // grad_V = attention_weights^T * grad_attention_output
    Matrix grad_V(seq_len, d_model_int, 0.0f);
    const std::vector<float> &attn_weights_data = last_attention_weights.getDataVector();
    const std::vector<float> &grad_attn_out_data = grad_attention_output.getDataVector();
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                grad_sum += attn_weights_data[k * seq_len + i] * grad_attn_out_data[k * d_model_int + j];
            }
            grad_V.setElement(i, j, grad_sum);
        }
    }
    
    // Paso 4: Gradiente hacia attention_weights (más complejo - gradiente de softmax)
    Matrix grad_attention_weights(seq_len, seq_len, 0.0f);
    const std::vector<float> &v_data = last_V.getDataVector();
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < d_model_int; k++) {
                grad_sum += grad_attn_out_data[i * d_model_int + k] * v_data[j * d_model_int + k];
            }
            grad_attention_weights.setElement(i, j, grad_sum);
        }
    }
    
    // Paso 5: Gradiente de softmax hacia scores
    Matrix grad_scores(seq_len, seq_len, 0.0f);
    const std::vector<float> &grad_attn_weights_data = grad_attention_weights.getDataVector();
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float grad_softmax = 0.0f;
            
            // Gradiente de softmax: s_i * (grad_i - sum(grad_k * s_k))
            float sum_term = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                sum_term += grad_attn_weights_data[i * seq_len + k] * attn_weights_data[i * seq_len + k];
            }
            
            grad_softmax = attn_weights_data[i * seq_len + j] * 
                          (grad_attn_weights_data[i * seq_len + j] - sum_term);
            
            grad_scores.setElement(i, j, grad_softmax);
        }
    }
    
    // Paso 6: Gradiente hacia Q y K desde scores
    // grad_Q = grad_scores * K / sqrt(d_k)
    // grad_K = grad_scores^T * Q / sqrt(d_k)
    Matrix grad_Q(seq_len, d_model_int, 0.0f);
    Matrix grad_K(seq_len, d_model_int, 0.0f);
    
    const std::vector<float> &grad_scores_data = grad_scores.getDataVector();
    const std::vector<float> &q_data = last_Q.getDataVector();
    const std::vector<float> &k_data = last_K.getDataVector();
    float scale = 1.0f / std::sqrt((float)d_model_int);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_q_sum = 0.0f;
            float grad_k_sum = 0.0f;
            
            for (int k = 0; k < seq_len; k++) {
                grad_q_sum += grad_scores_data[i * seq_len + k] * k_data[k * d_model_int + j];
                grad_k_sum += grad_scores_data[k * seq_len + i] * q_data[k * d_model_int + j];
            }
            
            grad_Q.setElement(i, j, grad_q_sum * scale);
            grad_K.setElement(i, j, grad_k_sum * scale);
        }
    }
    
    // Paso 7: Gradientes de matrices de peso W_Q, W_K, W_V
    // grad_W_Q += last_query^T * grad_Q
    // grad_W_K += last_key^T * grad_K  
    // grad_W_V += last_value^T * grad_V
    
    const std::vector<float> &query_data = last_query.getDataVector();
    const std::vector<float> &key_data = last_key.getDataVector();
    const std::vector<float> &value_data = last_value.getDataVector();
    const std::vector<float> &grad_q_data = grad_Q.getDataVector();
    const std::vector<float> &grad_k_data = grad_K.getDataVector();
    const std::vector<float> &grad_v_data = grad_V.getDataVector();
    
    // Actualizar grad_W_Q
    std::vector<float> grad_wq_data = grad_W_Q.getDataVector();
    for (int i = 0; i < d_model_int; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                grad_sum += query_data[k * d_model_int + i] * grad_q_data[k * d_model_int + j];
            }
            grad_wq_data[i * d_model_int + j] += grad_sum;
        }
    }
    grad_W_Q = Matrix(grad_wq_data, d_model_int, d_model_int);
    
    // Actualizar grad_W_K
    std::vector<float> grad_wk_data = grad_W_K.getDataVector();
    for (int i = 0; i < d_model_int; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                grad_sum += key_data[k * d_model_int + i] * grad_k_data[k * d_model_int + j];
            }
            grad_wk_data[i * d_model_int + j] += grad_sum;
        }
    }
    grad_W_K = Matrix(grad_wk_data, d_model_int, d_model_int);
    
    // Actualizar grad_W_V
    std::vector<float> grad_wv_data = grad_W_V.getDataVector();
    for (int i = 0; i < d_model_int; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                grad_sum += value_data[k * d_model_int + i] * grad_v_data[k * d_model_int + j];
            }
            grad_wv_data[i * d_model_int + j] += grad_sum;
        }
    }
    grad_W_V = Matrix(grad_wv_data, d_model_int, d_model_int);
    
    // Paso 8: Gradientes hacia las entradas originales
    // grad_query = grad_Q * W_Q^T
    // grad_key = grad_K * W_K^T
    // grad_value = grad_V * W_V^T
    
    grad_query = Matrix(seq_len, d_model_int, 0.0f);
    grad_key = Matrix(seq_len, d_model_int, 0.0f);
    grad_value = Matrix(seq_len, d_model_int, 0.0f);
    
    const std::vector<float> &wq_data = W_Q.getDataVector();
    const std::vector<float> &wk_data = W_K.getDataVector();
    const std::vector<float> &wv_data = W_V.getDataVector();
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model_int; j++) {
            float grad_query_sum = 0.0f;
            float grad_key_sum = 0.0f;
            float grad_value_sum = 0.0f;
            
            for (int k = 0; k < d_model_int; k++) {
                grad_query_sum += grad_q_data[i * d_model_int + k] * wq_data[j * d_model_int + k];
                grad_key_sum += grad_k_data[i * d_model_int + k] * wk_data[j * d_model_int + k];
                grad_value_sum += grad_v_data[i * d_model_int + k] * wv_data[j * d_model_int + k];
            }
            
            grad_query.setElement(i, j, grad_query_sum);
            grad_key.setElement(i, j, grad_key_sum);
            grad_value.setElement(i, j, grad_value_sum);
        }
    }
}

void MultiHeadAttention::updateWeights(float learning_rate) {
    // Actualizar todos los pesos usando gradientes acumulados con gradient clipping
    
    // Aplicar gradient clipping para evitar explosión de gradientes
    float clip_threshold = 1.0f;
    
    // Actualizar W_Q
    std::vector<float> wq_data = W_Q.getDataVector();
    const std::vector<float> &grad_wq_data = grad_W_Q.getDataVector();
    
    for (size_t i = 0; i < wq_data.size() && i < grad_wq_data.size(); i++) {
        float clipped_grad = std::max(-clip_threshold, std::min(clip_threshold, grad_wq_data[i]));
        wq_data[i] -= learning_rate * clipped_grad;
    }
    W_Q = Matrix(wq_data, d_model, d_model);
    
    // Actualizar W_K
    std::vector<float> wk_data = W_K.getDataVector();
    const std::vector<float> &grad_wk_data = grad_W_K.getDataVector();
    
    for (size_t i = 0; i < wk_data.size() && i < grad_wk_data.size(); i++) {
        float clipped_grad = std::max(-clip_threshold, std::min(clip_threshold, grad_wk_data[i]));
        wk_data[i] -= learning_rate * clipped_grad;
    }
    W_K = Matrix(wk_data, d_model, d_model);
    
    // Actualizar W_V
    std::vector<float> wv_data = W_V.getDataVector();
    const std::vector<float> &grad_wv_data = grad_W_V.getDataVector();
    
    for (size_t i = 0; i < wv_data.size() && i < grad_wv_data.size(); i++) {
        float clipped_grad = std::max(-clip_threshold, std::min(clip_threshold, grad_wv_data[i]));
        wv_data[i] -= learning_rate * clipped_grad;
    }
    W_V = Matrix(wv_data, d_model, d_model);
    
    // Actualizar W_O
    std::vector<float> wo_data = W_O.getDataVector();
    const std::vector<float> &grad_wo_data = grad_W_O.getDataVector();
    
    for (size_t i = 0; i < wo_data.size() && i < grad_wo_data.size(); i++) {
        float clipped_grad = std::max(-clip_threshold, std::min(clip_threshold, grad_wo_data[i]));
        wo_data[i] -= learning_rate * clipped_grad;
    }
    W_O = Matrix(wo_data, d_model, d_model);
    
    // Resetear gradientes después de actualizar
    grad_W_Q = Matrix(d_model, d_model, 0.0f);
    grad_W_K = Matrix(d_model, d_model, 0.0f);
    grad_W_V = Matrix(d_model, d_model, 0.0f);
    grad_W_O = Matrix(d_model, d_model, 0.0f);
    
    // Debug: Verificar que los pesos no son NaN o Inf
    bool has_nan = false;
    const std::vector<float> &check_wq = W_Q.getDataVector();
    for (float val : check_wq) {
        if (std::isnan(val) || std::isinf(val)) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "[WARNING] NaN/Inf detected in attention weights, reinitializing..." << std::endl;
        initializeWeights();
    }
}

void MultiHeadAttention::updateGradients(const Matrix &grad_output, const Matrix &query, const Matrix &key, const Matrix &value) {
    // Este método se mantiene por compatibilidad, no hace nada
}

// Implementación en MultiHeadAttention
void MultiHeadAttention::saveWeights(std::ofstream& file) {
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&n_heads), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&d_k), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&d_v), sizeof(size_t));
    Matrix::saveMatrix(file, W_Q);
    Matrix::saveMatrix(file, W_K);
    Matrix::saveMatrix(file, W_V);
    Matrix::saveMatrix(file, W_O);
}

void MultiHeadAttention::loadWeights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&d_model), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_heads), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&d_k), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&d_v), sizeof(size_t));
    W_Q = Matrix::loadMatrix(file);
    W_K = Matrix::loadMatrix(file);
    W_V = Matrix::loadMatrix(file);
    W_O = Matrix::loadMatrix(file);
    grad_W_Q = Matrix(W_Q.getRows(), W_Q.getCols(), 0.0);
    grad_W_K = Matrix(W_K.getRows(), W_K.getCols(), 0.0);
    grad_W_V = Matrix(W_V.getRows(), W_V.getCols(), 0.0);
    grad_W_O = Matrix(W_O.getRows(), W_O.getCols(), 0.0);
}

