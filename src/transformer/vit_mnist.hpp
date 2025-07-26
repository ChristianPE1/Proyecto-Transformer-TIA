#pragma once
#include "../utils/matrix.hpp"
#include "../layers/linear.hpp"
#include "attention.hpp"
#include "../layers/layer_norm.hpp"
#include "../layers/feed_forward.hpp"
#include <vector>

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
    ViTBlock(int embed_dim, int num_heads);
    Matrix forward(const Matrix& x);
    Matrix backward(const Matrix& grad_output);
    void updateWeights(float learning_rate);
};

class ViTMNIST {
private:
    PatchEmbedding patch_embed;
    std::vector<ViTBlock> blocks;
    LayerNorm norm;
    Linear classifier;
    Matrix pos_embedding;
    int num_patches;
    int embed_dim;
    int num_classes;
    Matrix last_pooled;
    Matrix last_normalized;
public:
    ViTMNIST(int patch_size = 4, int embed_dim = 128, int num_heads = 8, int num_layers = 6, int num_classes = 10);
    Matrix forward(const Matrix& x);
    void backward(const Matrix& loss_grad);
    void update_weights(float learning_rate = 0.001f);
};