#pragma once
#include <vector>
#include <cstddef>

class Matrix {
private:
    std::vector<float> data;
    int rows, cols;
public:
    Matrix() : rows(0), cols(0) {}
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, float init_val);
    Matrix(const std::vector<float> &hostData, int rows, int cols);
    ~Matrix() = default;
    Matrix(const Matrix &other) = default;
    Matrix &operator=(const Matrix &other) = default;
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    float *getData() { return data.data(); }
    const float *getData() const { return data.data(); }
    const std::vector<float> &getDataVector() const { return data; }
    std::vector<float> &getDataVector() { return data; }
    bool isEmpty() const { return rows == 0 || cols == 0; }
    void resize(int newRows, int newCols);
    void fill(float value);
    Matrix add(const Matrix &other) const;
    Matrix multiply(const Matrix &other) const;
    float getElement(int row, int col) const;
    void setElement(int row, int col, float value);
};
