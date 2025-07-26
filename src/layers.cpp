#pragma once
#include "matrix.cpp"
#include "utils.cpp"
#include <vector>
#include <cmath>
#include <stdexcept>

class Embedding
{
private:
   Matrix embedding_table;
   size_t vocab_size;
   size_t d_model;

public:
   Embedding(size_t vocab_size, size_t d_model)
       : vocab_size(vocab_size), d_model(d_model),
         embedding_table(vocab_size, d_model)
   {
      embedding_table.initializeXavier();
   }

   Matrix forward(const std::vector<int> &input_ids)
   {
      Matrix result(input_ids.size(), d_model);
      for (size_t i = 0; i < input_ids.size(); ++i)
      {
         if (input_ids[i] >= 0 && input_ids[i] < static_cast<int>(vocab_size))
         {
            for (size_t j = 0; j < d_model; ++j)
            {
               result[i][j] = embedding_table[input_ids[i]][j];
            }
         }
      }
      return result;
   }
};

// Layer Normalization
class LayerNorm
{
private:
   size_t d_model;
   std::vector<double> gamma, beta;
   double eps = 1e-6;

public:
   LayerNorm(size_t d_model) : d_model(d_model)
   {
      gamma.resize(d_model, 1.0);
      beta.resize(d_model, 0.0);
   }

   Matrix forward(const Matrix &input)
   {
      Matrix result(input.getRows(), input.getCols());

      for (size_t i = 0; i < input.getRows(); ++i)
      {
         // Calcular media y varianza
         double mean = 0.0;
         for (size_t j = 0; j < d_model; ++j)
         {
            mean += input[i][j];
         }
         mean /= d_model;

         double variance = 0.0;
         for (size_t j = 0; j < d_model; ++j)
         {
            variance += std::pow(input[i][j] - mean, 2);
         }
         variance /= d_model;

         // Normalizar
         for (size_t j = 0; j < d_model; ++j)
         {
            result[i][j] = gamma[j] * (input[i][j] - mean) / std::sqrt(variance + eps) + beta[j];
         }
      }

      return result;
   }
};

// Feed Forward Network
class FeedForward
{
private:
   size_t d_model;
   size_t d_ff;
   Matrix W1, W2;
   std::vector<double> b1, b2;

public:
   FeedForward(size_t d_model, size_t d_ff = 2048)
       : d_model(d_model), d_ff(d_ff),
         W1(d_model, d_ff), W2(d_ff, d_model),
         b1(d_ff, 0.0), b2(d_model, 0.0)
   {

      W1.initializeXavier();
      W2.initializeXavier();
   }

   Matrix forward(const Matrix &input)
   {
      // Primera transformacion lineal + ReLU
      Matrix hidden = input.multiply(W1);

      // Agregar bias
      for (size_t i = 0; i < hidden.getRows(); ++i)
      {
         for (size_t j = 0; j < hidden.getCols(); ++j)
         {
            hidden[i][j] += b1[j];
         }
      }

      hidden = relu(hidden);

      // Segunda transformacion lineal
      Matrix output = hidden.multiply(W2);

      // Agregar bias
      for (size_t i = 0; i < output.getRows(); ++i)
      {
         for (size_t j = 0; j < output.getCols(); ++j)
         {
            output[i][j] += b2[j];
         }
      }

      return output;
   }
};
