#pragma once
#include "../utils/matrix.hpp"
#include "../layers/linear.hpp"
#include "attention.hpp"
#include "../layers/layer_norm.hpp"
#include "../layers/feed_forward.hpp"
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>
#include "data/mnist_loader.hpp"

class PatchEmbedding {
private:
    int patch_size;
    int embed_dim;
    Matrix projection;
    Matrix grad_projection;
    Matrix last_patches;
public:
    PatchEmbedding(int patch_size, int embed_dim);
    Matrix forward(const Matrix& image);
    Matrix backward(const Matrix& grad_output);
    void updateWeights(float learning_rate);
    void saveWeights(std::ofstream& file);
    void loadWeights(std::ifstream& file);
};

class ViTBlock {
private:
    MultiHeadAttention attention;
    FeedForward mlp;
    LayerNorm norm1, norm2;
    Matrix stored_input;
    Matrix stored_attn_out;
    Matrix stored_x1;
    Matrix stored_x1_norm;
    Matrix stored_mlp_out;
    Matrix stored_x2;
public:
    ViTBlock(int embed_dim, int num_heads, int mlp_hidden_layers_size);
    Matrix forward(const Matrix& x);
    Matrix backward(const Matrix& grad_output);
    void updateWeights(float learning_rate);
    void saveWeights(std::ofstream& file);
    void loadWeights(std::ifstream& file);
};

class ViTMNIST {
private:
    PatchEmbedding patch_embed;
    std::vector<ViTBlock> blocks;
    LayerNorm norm;
    Linear classifier;
    Matrix pos_embedding;
    int patch_size;
    int num_patches;
    int embed_dim;
    int num_classes;
    Matrix last_pooled;
    Matrix last_normalized;
public:
    ViTMNIST(int patch_size = 4,
        int embed_dim = 128,
        int num_heads = 8,
        int num_layers = 6,
        int mlp_hidden_layers_size = 128,
        int num_classes = 10);
    Matrix forward(const Matrix& x);
    void backward(const Matrix& loss_grad);
    void update_weights(float learning_rate = 0.001f);
    int predict(const Matrix& image);
    void save_weights(const std::string& file_path);
    void load_weights(const std::string& file_path);


    // Static method to load a pretrained model
    static ViTMNIST load_pretrained_model(const std::string& path = "./weights/mnist.bin") {
        int patch_size = 7;  
        int embed_dim = 64;   
        int num_heads = 2;    
        int num_layers = 3;  
        int mlp_hidden_layers_size = 96;
        int num_classes = 10;

        ViTMNIST vit_model(
            patch_size,
            embed_dim,
            num_heads,
            num_layers,
            mlp_hidden_layers_size,
            num_classes);

        vit_model.load_weights(path);

        return vit_model;
    }
};