#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <array>
#include <cstdint>

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

        magic_number = static_cast<int>(read_big_endian_uint32(images_file));
        num_images = static_cast<int>(read_big_endian_uint32(images_file));
        rows = static_cast<int>(read_big_endian_uint32(images_file));
        cols = static_cast<int>(read_big_endian_uint32(images_file));

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

        label_magic = static_cast<int>(read_big_endian_uint32(labels_file));
        num_labels = static_cast<int>(read_big_endian_uint32(labels_file));

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

    static uint32_t read_big_endian_uint32(std::ifstream& stream) {
        std::array<uint8_t, 4> bytes;
        stream.read(reinterpret_cast<char*>(bytes.data()), 4);
        return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    }
};
