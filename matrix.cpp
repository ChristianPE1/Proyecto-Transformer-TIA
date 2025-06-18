#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>

class Matrix
{
private:
   std::vector<std::vector<double>> data;
   size_t rows, cols;

public:
   Matrix(size_t r, size_t c) : rows(r), cols(c)
   {
      data.resize(rows, std::vector<double>(cols, 0.0));
   }

   Matrix(size_t r, size_t c, double val) : rows(r), cols(c)
   {
      data.resize(rows, std::vector<double>(cols, val));
   }

   // Operadores de acceso
   std::vector<double> &operator[](size_t i) { return data[i]; }
   const std::vector<double> &operator[](size_t i) const { return data[i]; }

   // Getters
   size_t getRows() const { return rows; }
   size_t getCols() const { return cols; }

   // Multiplicación de matrices
   Matrix multiply(const Matrix &other) const
   {
      if (cols != other.rows)
      {
         throw std::invalid_argument("Matrix dimensions don't match for multiplication");
      }

      Matrix result(rows, other.cols);
      for (size_t i = 0; i < rows; ++i)
      {
         for (size_t j = 0; j < other.cols; ++j)
         {
            for (size_t k = 0; k < cols; ++k)
            {
               result[i][j] += data[i][k] * other[k][j];
            }
         }
      }
      return result;
   }

   // Transposicion
   Matrix transpose() const
   {
      Matrix result(cols, rows);
      for (size_t i = 0; i < rows; ++i)
      {
         for (size_t j = 0; j < cols; ++j)
         {
            result[j][i] = data[i][j];
         }
      }
      return result;
   }

   // Suma de matrices
   Matrix add(const Matrix &other) const
   {
      if (rows != other.rows || cols != other.cols)
      {
         throw std::invalid_argument("Matrix dimensions don't match for addition");
      }

      Matrix result(rows, cols);
      for (size_t i = 0; i < rows; ++i)
      {
         for (size_t j = 0; j < cols; ++j)
         {
            result[i][j] = data[i][j] + other[i][j];
         }
      }
      return result;
   }

   // Escalado por constante
   Matrix scale(double factor) const
   {
      Matrix result(rows, cols);
      for (size_t i = 0; i < rows; ++i)
      {
         for (size_t j = 0; j < cols; ++j)
         {
            result[i][j] = data[i][j] * factor;
         }
      }
      return result;
   }

   // Inicializacion aleatoria (Xavier/Glorot)
   void initializeXavier()
   {
      std::random_device rd;
      std::mt19937 gen(rd());
      double limit = std::sqrt(6.0 / (rows + cols));
      std::uniform_real_distribution<> dis(-limit, limit);

      for (size_t i = 0; i < rows; ++i)
      {
         for (size_t j = 0; j < cols; ++j)
         {
            data[i][j] = dis(gen);
         }
      }
   }

   // Print matrix (helper para debugging)
   void print() const
   {
      for (size_t i = 0; i < std::min(rows, size_t(5)); ++i)
      {
         for (size_t j = 0; j < std::min(cols, size_t(5)); ++j)
         {
            std::cout << data[i][j] << " ";
         }
         std::cout << std::endl;
      }
   }
};
