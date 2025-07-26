#include "vit_mnist.hpp"
#include <cmath>
#include <iostream>
#include <random>

// Implementacion de PatchEmbedding
PatchEmbedding::PatchEmbedding(int patch_size, int embed_dim) 
    : patch_size(patch_size), embed_dim(embed_dim), 
      projection(patch_size * patch_size, embed_dim),
      grad_projection(patch_size * patch_size, embed_dim, 0.0f) {
    // Inicializar pesos de proyeccion con inicializacion conservadora
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(1.0f / (patch_size * patch_size)) * 0.1f;  // pequeño
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    std::vector<float> weights(patch_size * patch_size * embed_dim);
    for (auto& w : weights) {
        w = dist(gen);
    }
    projection = Matrix(weights, patch_size * patch_size, embed_dim);
}

Matrix PatchEmbedding::forward(const Matrix& image) {
    // Convertir imagen 28x28 en parches
    int img_size = 28;
    int num_patches_per_dim = img_size / patch_size;
    int num_patches = num_patches_per_dim * num_patches_per_dim;
    
    Matrix patches(num_patches, patch_size * patch_size, 0.0f);
    
    // Extraer parches (CPU)
    const std::vector<float> &image_data = image.getDataVector();
    
    std::vector<float> patch_data(num_patches * patch_size * patch_size);
    int patch_idx = 0;
    
    for (int i = 0; i < num_patches_per_dim; i++) {
        for (int j = 0; j < num_patches_per_dim; j++) {
            for (int pi = 0; pi < patch_size; pi++) {
                for (int pj = 0; pj < patch_size; pj++) {
                    int img_row = i * patch_size + pi;
                    int img_col = j * patch_size + pj;
                    int patch_pos = patch_idx * patch_size * patch_size + pi * patch_size + pj;
                    patch_data[patch_pos] = image_data[img_row * img_size + img_col];
                }
            }
            patch_idx++;
        }
    }

    patches = Matrix(patch_data, num_patches, patch_size * patch_size);

    // Guardar patches para backward
    last_patches = patches;

    // Proyectar parches al espacio de embedding
    Matrix result = patches.multiply(projection);
    if (result.getRows() == 0 || result.getCols() == 0) {
        throw std::runtime_error("PatchEmbedding::forward - multiply operation returned empty matrix");
    }
    return result;
}

Matrix PatchEmbedding::backward(const Matrix& grad_output) {
    // Versión simplificada para evitar explosión de gradientes
    const std::vector<float> &grad_data = grad_output.getDataVector();
    
    // Usar gradientes muy pequeños
    float scale_factor = 0.01f;
    
    std::vector<float> current_grad = grad_projection.getDataVector();
    
    // Acumulación simple de gradientes
    for (size_t i = 0; i < current_grad.size() && i < grad_data.size(); i++) {
        float g = grad_data[i % grad_data.size()] * scale_factor;
        // Clip individual gradients
        g = std::max(-0.01f, std::min(0.01f, g));
        current_grad[i] += g;
    }
    
    grad_projection = Matrix(current_grad, patch_size * patch_size, embed_dim);
    
    // Retornar gradiente respecto a los patches (no necesario para las imagenes)
    return Matrix(last_patches.getRows(), last_patches.getCols(), 0.0f);
}

void PatchEmbedding::updateWeights(float learning_rate) {
    // Actualizar pesos usando gradientes acumulados con clipping
    std::vector<float> proj_data = projection.getDataVector();
    const std::vector<float> &grad_data = grad_projection.getDataVector();
    
    float max_grad = 0.1f;  // Clipping agresivo
    
    for (size_t i = 0; i < proj_data.size() && i < grad_data.size(); i++) {
        // Clip gradients
        float clipped_grad = std::max(-max_grad, std::min(max_grad, grad_data[i]));
        proj_data[i] -= learning_rate * clipped_grad;
        // Clip weights
        proj_data[i] = std::max(-1.0f, std::min(1.0f, proj_data[i]));
    }
    
    projection = Matrix(proj_data, patch_size * patch_size, embed_dim);
    
    // Resetear gradientes
    grad_projection = Matrix(patch_size * patch_size, embed_dim, 0.0f);
}

// Implementacion de ViTBlock
ViTBlock::ViTBlock(int embed_dim, int num_heads) 
    : attention(embed_dim, num_heads), mlp(embed_dim, embed_dim * 4), 
      norm1(embed_dim), norm2(embed_dim) {
}

Matrix ViTBlock::forward(const Matrix& x) {
    // Versión completa con atención entreneable
    stored_input = x;
    
    // Self-attention con norm1
    stored_x1_norm = norm1.forward(x);
    stored_attn_out = attention.forward(stored_x1_norm, stored_x1_norm, stored_x1_norm);
    
    // Primera conexión residual
    stored_x1 = stored_input.add(stored_attn_out);
    
    // MLP con norm2
    Matrix x1_norm2 = norm2.forward(stored_x1);
    stored_mlp_out = mlp.forward(x1_norm2);
    
    // Segunda conexión residual
    stored_x2 = stored_x1.add(stored_mlp_out);
    
    return stored_x2;
}

Matrix ViTBlock::backward(const Matrix& grad_output) {
    // Backward completo con atención
    
    // Retropropagar por segunda conexion residual
    Matrix grad_x1_from_residual = grad_output;
    Matrix grad_mlp_out = grad_output;
    
    // Retropropagar por MLP
    Matrix x1_norm2 = norm2.forward(stored_x1);
    Matrix grad_x1_norm2 = mlp.backward(grad_mlp_out, x1_norm2);
    
    // Retropropagar por norm2
    Matrix grad_x1_from_mlp = norm2.backward(grad_x1_norm2, stored_x1);
    
    // Sumar gradientes de ambos caminos
    Matrix grad_x1_total = grad_x1_from_residual.add(grad_x1_from_mlp);
    
    // Retropropagar por primera conexion residual
    Matrix grad_input_from_residual = grad_x1_total;
    Matrix grad_attn_out = grad_x1_total;
    
    // Retropropagar por atencion - esto calcula y acumula gradientes automáticamente
    Matrix grad_q, grad_k, grad_v;
    attention.backward(grad_attn_out, grad_q, grad_k, grad_v);
    
    // Retropropagar por norm1
    Matrix grad_input_from_attention = norm1.backward(grad_q, stored_input);
    
    // Sumar gradientes de ambos caminos
    Matrix grad_input = grad_input_from_residual.add(grad_input_from_attention);
    
    return grad_input;
}


void ViTBlock::updateWeights(float learning_rate) {
    // Actualizar TODOS los componentes del bloque transformer
    attention.updateWeights(learning_rate);
    norm1.updateWeights(learning_rate);
    mlp.updateWeights(learning_rate);
    norm2.updateWeights(learning_rate);
}

// Implementacion de ViTMNIST
ViTMNIST::ViTMNIST(int patch_size, int embed_dim, int num_heads, int num_layers, int num_classes)
    : patch_embed(patch_size, embed_dim), norm(embed_dim), classifier(embed_dim, num_classes),
      embed_dim(embed_dim), num_classes(num_classes), 
      last_pooled(1, embed_dim, 0.0f), last_normalized(1, embed_dim, 0.0f) {
    
    // Validar parametros de entrada
    if (patch_size <= 0 || embed_dim <= 0 || num_heads <= 0 || num_layers <= 0 || num_classes <= 0) {
        throw std::runtime_error("Invalid ViTMNIST parameters - all must be positive");
    }
    
    // Calcular cantidad de parches
    int img_size = 28;
    int num_patches_per_dim = img_size / patch_size;
    num_patches = num_patches_per_dim * num_patches_per_dim;
    
    if (num_patches <= 0) {
        throw std::runtime_error("Invalid number of patches calculated");
    }
    
    // Actualizar last_normalized a las dimensiones correctas
    last_normalized = Matrix(num_patches, embed_dim, 0.0f);
    
    // Inicializar embeddings posicionales con valores pequenos
    pos_embedding = Matrix(num_patches, embed_dim, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f); // Smaller std for positional embeddings
    
    std::vector<float> pos_data(num_patches * embed_dim);
    for (auto& p : pos_data) {
        p = dist(gen);
    }
    pos_embedding = Matrix(pos_data, num_patches, embed_dim);
    
    // Inicializar bloques transformer
    blocks.reserve(num_layers);
    for (int i = 0; i < num_layers; i++) {
        blocks.emplace_back(embed_dim, num_heads);
    }
}

Matrix ViTMNIST::forward(const Matrix& x) {
    // Embedding de parches
    Matrix patches = patch_embed.forward(x);
    if (patches.getRows() == 0 || patches.getCols() == 0) {
        throw std::runtime_error("ViTMNIST::forward - patch embedding returned empty matrix");
    }
    
    // Sumar embeddings posicionales
    Matrix x_with_pos = patches.add(pos_embedding);
    if (x_with_pos.getRows() == 0 || x_with_pos.getCols() == 0) {
        throw std::runtime_error("ViTMNIST::forward - positional embedding add returned empty matrix");
    }
    
    // Pasar por los bloques transformer
    Matrix current = x_with_pos;
    for (size_t i = 0; i < blocks.size(); i++) {
        current = blocks[i].forward(current);
        if (current.getRows() == 0 || current.getCols() == 0) {
            throw std::runtime_error("ViTMNIST::forward - transformer block " + std::to_string(i) + " returned empty matrix");
        }
    }
    
    // Normalizacion por capas
    last_normalized = norm.forward(current);
    if (last_normalized.getRows() == 0 || last_normalized.getCols() == 0) {
        throw std::runtime_error("ViTMNIST::forward - layer norm returned empty matrix");
    }
    
    // Pooling promedio global (tomar media sobre parches)
    last_pooled = Matrix(1, embed_dim, 0.0f);
    const std::vector<float> &norm_data = last_normalized.getDataVector();
    
    if (norm_data.empty() || norm_data.size() != num_patches * embed_dim) {
        throw std::runtime_error("ViTMNIST::forward - Invalid norm_data size from layer norm");
    }
    
    std::vector<float> pool_data(embed_dim, 0.0f);
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_patches; j++) {
            sum += norm_data[j * embed_dim + i];
        }
        pool_data[i] = sum / num_patches;
    }
    
    // Validar pool_data antes de copiar
    if (pool_data.size() != embed_dim) {
        std::cerr << "[ERROR] ViTMNIST::forward - Invalid pool data size: " 
                  << pool_data.size() << " expected: " << embed_dim << std::endl;
        throw std::runtime_error("Invalid pool data size");
    }
    
    last_pooled = Matrix(pool_data, 1, embed_dim);
    
    // Clasificacion
    Matrix output = classifier.forward(last_pooled);
    if (output.getRows() == 0 || output.getCols() == 0) {
        throw std::runtime_error("ViTMNIST::forward - classifier returned empty matrix");
    }
    
    return output;
}

// --- Metodos de actualizacion y retropropagacion ---

void ViTMNIST::backward(const Matrix& loss_grad) {
    // Validar gradiente de entrada
    if (loss_grad.getRows() == 0 || loss_grad.getCols() == 0) {
        return;
    }
    
    if (loss_grad.getRows() != 1 || loss_grad.getCols() != num_classes) {
        return; 
    }
    
    // Retropropagar por la capa de clasificacion
    Matrix grad_pooled = classifier.backward(loss_grad, last_pooled);
    
    // Validar gradiente de la capa de clasificacion
    if (grad_pooled.getRows() == 0 || grad_pooled.getCols() == 0) {
        return;
    }
    
    if (grad_pooled.getRows() != 1 || grad_pooled.getCols() != embed_dim) {
        std::cerr << "[ERROR] ViTMNIST::backward - Invalid classifier gradient dimensions: " 
                  << grad_pooled.getRows() << "x" << grad_pooled.getCols() 
                  << " expected: 1x" << embed_dim << std::endl;
        return;
    }
    
    // Expandir gradiente del pooling a todos los parches
    Matrix grad_normalized(num_patches, embed_dim, 0.0f);
    const std::vector<float> &grad_pooled_data = grad_pooled.getDataVector();
    
    // Verificar que los datos sean validos
    if (grad_pooled_data.empty()) {
        std::cerr << "[ERROR] ViTMNIST::backward - Empty gradient data from classifier!" << std::endl;
        return;
    }
    
    std::vector<float> grad_norm_data(num_patches * embed_dim);
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < embed_dim; j++) {
            grad_norm_data[i * embed_dim + j] = grad_pooled_data[j] / num_patches;
        }
    }
    grad_normalized = Matrix(grad_norm_data, num_patches, embed_dim);
    
    // Retropropagar por normalizacion de capa
    Matrix grad_blocks = norm.backward(grad_normalized, last_normalized);
    
    // Retropropagar por los bloques transformer
    Matrix current_grad = grad_blocks;
    for (int i = blocks.size() - 1; i >= 0; i--) {
        if (current_grad.getRows() == 0 || current_grad.getCols() == 0) {
            break; // Stop if gradient becomes empty
        }
        current_grad = blocks[i].backward(current_grad);
    }
    
    // Retropropagar por patch embedding
    if (current_grad.getRows() > 0 && current_grad.getCols() > 0) {
        patch_embed.backward(current_grad);
    }
}

void ViTMNIST::update_weights(float learning_rate) {
    // Actualizar todos los componentes
    classifier.updateWeights(learning_rate);
    norm.updateWeights(learning_rate);
    patch_embed.updateWeights(learning_rate);
    
    for (auto& block : blocks) {
        block.updateWeights(learning_rate);
    }
}

// --- Metodos para PatchEmbedding y otros componentes pueden agregarse si se requiere entrenamiento completo ---
// Por ahora, solo se tiene el entrenamiento simplificado de la capa de clasificacion