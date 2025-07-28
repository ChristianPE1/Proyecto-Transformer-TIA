#pragma once
#include <vector>
#include <cstddef>
#include <fstream>

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
    int getSize() { return data.size(); }
    float& operator[](int i) { return data[i]; }
    const float& operator[](int i) const { return data[i]; }

    // Funciones auxiliares para serialización de matrices
    static void saveMatrix(std::ofstream& file, const Matrix& matrix) {
        int rows = matrix.getRows();
        int cols = matrix.getCols();

        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

        const float* data = matrix.getData();
        file.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(float));
    }

    static Matrix loadMatrix(std::ifstream& file) {
        int rows, cols;

        // Leer dimensiones
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));

        // Crear matriz y leer datos
        Matrix matrix(rows, cols);
        float* data = matrix.getData();
        file.read(reinterpret_cast<char*>(data), rows * cols * sizeof(float));

        return matrix;
    }
};
