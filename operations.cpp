#pragma once
#include "matrix.cpp"
#include "utils.cpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

// Codificacion Posicional
class PositionalEncoding
{
private:
   size_t d_model;
   size_t max_seq_len;
   Matrix encoding;

public:
   PositionalEncoding(size_t d_model, size_t max_seq_len = 5000)
       : d_model(d_model), max_seq_len(max_seq_len), encoding(max_seq_len, d_model)
   {

      for (size_t pos = 0; pos < max_seq_len; ++pos)
      {
         for (size_t i = 0; i < d_model; ++i)
         {
            if (i % 2 == 0)
            {
               // Seno para posiciones pares
               encoding[pos][i] = std::sin(pos / std::pow(10000.0, (2.0 * i) / d_model));
            }
            else
            {
               // Coseno para posiciones impares
               encoding[pos][i] = std::cos(pos / std::pow(10000.0, (2.0 * (i - 1)) / d_model));
            }
         }
      }
   }

   Matrix getEncoding(size_t seq_len)
   {
      Matrix result(seq_len, d_model);
      for (size_t i = 0; i < seq_len; ++i)
      {
         for (size_t j = 0; j < d_model; ++j)
         {
            result[i][j] = encoding[i][j];
         }
      }
      return result;
   }
};

// Multi-Head Attention (MEJORADO)
class MultiHeadAttention
{
private:
   size_t d_model;
   size_t n_heads;
   size_t d_k;

   Matrix W_Q, W_K, W_V, W_O;

public:
   MultiHeadAttention(size_t d_model, size_t n_heads)
       : d_model(d_model), n_heads(n_heads), d_k(d_model / n_heads),
         W_Q(d_model, d_model), W_K(d_model, d_model),
         W_V(d_model, d_model), W_O(d_model, d_model)
   {

      if (d_model % n_heads != 0)
      {
         throw std::invalid_argument("d_model must be divisible by n_heads");
      }

      // Inicializar pesos
      W_Q.initializeXavier();
      W_K.initializeXavier();
      W_V.initializeXavier();
      W_O.initializeXavier();
   }

   Matrix scaledDotProductAttention(const Matrix &Q, const Matrix &K, const Matrix &V,
                                    const Matrix *mask = nullptr)
   {
      // Calcular scores: QK^T / sqrt(d_k)
      Matrix scores = Q.multiply(K.transpose()).scale(1.0 / std::sqrt(d_k));

      // Aplicar mascara si existe
      if (mask != nullptr)
      {
         for (size_t i = 0; i < scores.getRows(); ++i)
         {
            for (size_t j = 0; j < scores.getCols(); ++j)
            {
               if ((*mask)[i][j] == 0)
               {
                  scores[i][j] = -1e9;
               }
            }
         }
      }

      // Aplicar softmax
      Matrix attention_weights = softmax(scores);

      // Multiplicar por V
      return attention_weights.multiply(V);
   }

   Matrix forward(const Matrix &query, const Matrix &key, const Matrix &value,
                  const Matrix *mask = nullptr)
   {
      // Proyecciones lineales
      Matrix Q = query.multiply(W_Q);
      Matrix K = key.multiply(W_K);
      Matrix V = value.multiply(W_V);

      // Para simplificar, implementamos sin reshape para múltiples cabezas
      Matrix attention_output = scaledDotProductAttention(Q, K, V, mask);

      // Proyeccion final
      return attention_output.multiply(W_O);
   }
};
