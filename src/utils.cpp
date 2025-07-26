#pragma once
#include "matrix.cpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iostream>
#include <unordered_map>


// Funcion softmax
Matrix softmax(const Matrix &input)
{
   Matrix result(input.getRows(), input.getCols());

   for (size_t i = 0; i < input.getRows(); ++i)
   {
      // Encontrar el maximo para estabilidad numerica
      double max_val = *std::max_element(input[i].begin(), input[i].end());

      // Calcular exponenciales y suma
      double sum = 0.0;
      for (size_t j = 0; j < input.getCols(); ++j)
      {
         result[i][j] = std::exp(input[i][j] - max_val);
         sum += result[i][j];
      }

      // Normalizar
      for (size_t j = 0; j < input.getCols(); ++j)
      {
         result[i][j] /= sum;
      }
   }

   return result;
}

// Funcion ReLU
Matrix relu(const Matrix &input)
{
   Matrix result(input.getRows(), input.getCols());
   for (size_t i = 0; i < input.getRows(); ++i)
   {
      for (size_t j = 0; j < input.getCols(); ++j)
      {
         result[i][j] = std::max(0.0, input[i][j]);
      }
   }
   return result;
}


class MaskUtils
{
public:
   // Crear mascara de padding (1 donde hay tokens reales, 0 donde hay padding)
   static Matrix createPaddingMask(const std::vector<int> &tokens, int pad_token = 0)
   {
      size_t seq_len = tokens.size();
      Matrix mask(1, seq_len);
      for (size_t i = 0; i < seq_len; ++i)
      {
         mask[0][i] = (tokens[i] != pad_token) ? 1.0 : 0.0;
      }
      return mask;
   }

   // Crear mascara look-ahead (triangular inferior) para el decoder
   static Matrix createLookAheadMask(size_t seq_len)
   {
      Matrix mask(seq_len, seq_len, 0.0);
      for (size_t i = 0; i < seq_len; ++i)
      {
         for (size_t j = 0; j <= i; ++j)
         {
            mask[i][j] = 1.0;
         }
      }
      return mask;
   }

   // Combinar mascaras de padding y look-ahead para el decoder
   static Matrix combineDecoderMasks(const std::vector<int> &tokens, int pad_token = 0)
   {
      size_t seq_len = tokens.size();
      Matrix look_ahead = createLookAheadMask(seq_len);

      // Aplicar mascara de padding
      for (size_t i = 0; i < seq_len; ++i)
      {
         if (tokens[i] == pad_token)
         {
            // Si el token es padding, poner toda la fila a 0
            for (size_t j = 0; j < seq_len; ++j)
            {
               look_ahead[i][j] = 0.0;
            }
         }
      }

      return look_ahead;
   }
};
