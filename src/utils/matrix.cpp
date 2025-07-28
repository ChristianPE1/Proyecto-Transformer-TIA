#include "utils/matrix.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <vector>
#include <algorithm>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
{
    data.resize(rows * cols, 0.0f);
}

Matrix::Matrix(int rows, int cols, float init_val) : rows(rows), cols(cols)
{
    data.resize(rows * cols, init_val);
}

Matrix::Matrix(const std::vector<float> &hostData, int rows, int cols) : rows(rows), cols(cols)
{
    if (hostData.size() != rows * cols)
    {
        throw std::runtime_error("Host data size doesn't match matrix dimensions");
    }
    data = hostData;
}

void Matrix::resize(int newRows, int newCols)
{
    rows = newRows;
    cols = newCols;
    data.resize(rows * cols, 0.0f);
}

void Matrix::fill(float value)
{
    std::fill(data.begin(), data.end(), value);
}

Matrix Matrix::add(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::runtime_error("Matrix dimensions don't match for addition");
    }

    Matrix result(rows, cols);
    
    // Simple CPU addition
    for (int i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}

Matrix Matrix::multiply(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }

    Matrix result(rows, other.cols);
    
    // Simple CPU matrix multiplication
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols; k++) {
                sum += data[i * cols + k] * other.data[k * other.cols + j];
            }
            result.data[i * other.cols + j] = sum;
        }
    }
    
    return result;
}

float Matrix::getElement(int row, int col) const
{
    if (row < 0 || row >= rows || col < 0 || col >= cols)
    {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row * cols + col];
}

void Matrix::setElement(int row, int col, float value)
{
    if (row < 0 || row >= rows || col < 0 || col >= cols)
    {
        throw std::out_of_range("Matrix index out of bounds");
    }
    data[row * cols + col] = value;
}
