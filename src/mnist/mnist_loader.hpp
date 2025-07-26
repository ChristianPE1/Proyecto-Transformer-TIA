#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

struct MNISTData {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

class MNISTLoader {
public:
    MNISTData load(const std::string& images_path, const std::string& labels_path) {
        MNISTData data;
        std::ifstream images_file(images_path, std::ios::binary);
        if (!images_file.is_open()) {
            throw std::runtime_error("Failed to open images file: " + images_path);
        }
        int magic_number, num_images, rows, cols;
        images_file.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
        images_file.read(reinterpret_cast<char*>(&num_images), sizeof(int));
        images_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        images_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        magic_number = __builtin_bswap32(magic_number);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        std::cout << "MNIST Images: " << num_images << " images of " << rows << "x" << cols << std::endl;
        for (int i = 0; i < num_images; i++) {
            std::vector<float> image(rows * cols);
            std::vector<unsigned char> buffer(rows * cols);
            images_file.read(reinterpret_cast<char*>(buffer.data()), rows * cols);
            for (int j = 0; j < rows * cols; j++) {
                image[j] = static_cast<float>(buffer[j]) / 255.0f;
            }
            data.images.push_back(image);
        }
        images_file.close();
        std::ifstream labels_file(labels_path, std::ios::binary);
        if (!labels_file.is_open()) {
            throw std::runtime_error("Failed to open labels file: " + labels_path);
        }
        int label_magic, num_labels;
        labels_file.read(reinterpret_cast<char*>(&label_magic), sizeof(int));
        labels_file.read(reinterpret_cast<char*>(&num_labels), sizeof(int));
        label_magic = __builtin_bswap32(label_magic);
        num_labels = __builtin_bswap32(num_labels);
        std::cout << "MNIST Labels: " << num_labels << " labels" << std::endl;
        for (int i = 0; i < num_labels; i++) {
            unsigned char label;
            labels_file.read(reinterpret_cast<char*>(&label), 1);
            data.labels.push_back(static_cast<int>(label));
        }
        labels_file.close();
        return data;
    }
    std::vector<std::vector<std::vector<float>>> create_patches(const std::vector<std::vector<float>>& images, int patch_size) {
        std::vector<std::vector<std::vector<float>>> patches;
        for (const auto& image : images) {
            std::vector<std::vector<float>> image_patches;
            for (int i = 0; i < 28; i += patch_size) {
                for (int j = 0; j < 28; j += patch_size) {
                    std::vector<float> patch;
                    for (int pi = 0; pi < patch_size; ++pi) {
                        for (int pj = 0; pj < patch_size; ++pj) {
                            int idx = (i + pi) * 28 + (j + pj);
                            if (idx < image.size()) {
                                patch.push_back(image[idx]);
                            }
                        }
                    }
                    image_patches.push_back(patch);
                }
            }
            patches.push_back(image_patches);
        }
        return patches;
    }
};