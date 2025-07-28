#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "transformer/vit_mnist.hpp"

struct ClassificationResult {
    int predicted_class;
    float confidence;
    std::vector<std::pair<float, int>> top_predictions;
    std::string image_path;
    bool success;
    std::string error_message;
};

struct ImageStats {
    float min_value;
    float max_value;
    int white_pixels;
    int black_pixels;
    int total_pixels;
};

class Classifier {
private:
    ViTMNIST model;

    std::vector<float> loadProcessedImage(const std::string& filename);
    ImageStats analyzeImageData(const std::vector<float>& image_data);
    std::vector<float> applySoftmax(const std::vector<float>& logits);
    std::vector<std::pair<float, int>> getTopPredictions(const std::vector<float>& probabilities, int top_k = 3);

public:

    Classifier(ViTMNIST& model) : model(model) {}



    ~Classifier() = default;

    ClassificationResult classifySingleImage(const std::string& image_path);

    std::vector<ClassificationResult> classifyAllProcessedImages(const std::string& processed_dir);

    ClassificationResult classifyImageData(const std::vector<float>& image_data, const std::string& image_name = "");

    void printClassificationResult(const ClassificationResult& result);
    void printSummaryResults(const std::vector<ClassificationResult>& results);

    void setVerbose(bool verbose) { this->verbose = verbose; }

private:
    bool verbose = true;
};


