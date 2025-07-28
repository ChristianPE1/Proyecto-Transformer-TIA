#pragma once
#include "../utils/matrix.hpp"
#include <vector>
#include <cstddef>

class CrossEntropyLoss {
public:
    static float compute_loss(const Matrix& predictions, const std::vector<int>& labels);
    static Matrix compute_gradients(const Matrix& predictions, const std::vector<int>& labels);
};
