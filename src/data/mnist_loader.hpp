#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <array>
#include <cstdint>
#include <algorithm>
#include <complex>
#include <sstream>

struct MNISTData {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

class MNISTLoader {
private:
    // Estructura para el header de archivos .npy
    struct NPYHeader {
        char magic[6];
        uint8_t major_version;
        uint8_t minor_version;
        uint16_t header_len;
        std::string dtype;
        std::vector<size_t> shape;
        bool fortran_order;
    };

    // Función para parsear el header de archivos .npy
    NPYHeader parse_npy_header(std::ifstream& file) {
        NPYHeader header;
        
        // Leer magic number
        file.read(header.magic, 6);
        if (std::string(header.magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid NPY file format");
        }
        
        // Leer versiones
        file.read(reinterpret_cast<char*>(&header.major_version), 1);
        file.read(reinterpret_cast<char*>(&header.minor_version), 1);
        
        // Leer longitud del header
        file.read(reinterpret_cast<char*>(&header.header_len), 2);
        
        // Leer el diccionario del header
        std::vector<char> header_dict(header.header_len);
        file.read(header_dict.data(), header.header_len);
        
        std::string header_str(header_dict.begin(), header_dict.end());
        
        // Parsear dtype
        size_t dtype_pos = header_str.find("'descr':");
        if (dtype_pos != std::string::npos) {
            size_t start = header_str.find("'", dtype_pos + 8) + 1;
            size_t end = header_str.find("'", start);
            header.dtype = header_str.substr(start, end - start);
        }
        
        // Parsear shape
        size_t shape_pos = header_str.find("'shape':");
        if (shape_pos != std::string::npos) {
            size_t start = header_str.find("(", shape_pos) + 1;
            size_t end = header_str.find(")", start);
            std::string shape_str = header_str.substr(start, end - start);
            
            // Parsear números de la tupla
            std::stringstream ss(shape_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
                if (!item.empty() && item != ",") {
                    header.shape.push_back(std::stoull(item));
                }
            }
        }
        
        // Parsear fortran_order
        header.fortran_order = header_str.find("'fortran_order': True") != std::string::npos;
        
        return header;
    }

public:
    // Método principal para cargar datos MNIST clásico (formato ubyte)
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
    
    // Método para cargar datos en formato NPY (OrganCMNIST)
    MNISTData load_npy(const std::string& images_path, const std::string& labels_path) {
        MNISTData data;
        
        // Cargar imágenes
        std::ifstream images_file(images_path, std::ios::binary);
        if (!images_file.is_open()) {
            throw std::runtime_error("Failed to open images file: " + images_path);
        }
        
        NPYHeader img_header = parse_npy_header(images_file);
        
        // Verificar que tenemos las dimensiones esperadas (N, H, W) o (N, H, W, C)
        if (img_header.shape.size() < 3) {
            throw std::runtime_error("Invalid image shape in NPY file");
        }
        
        size_t num_images = img_header.shape[0];
        size_t height = img_header.shape[1];
        size_t width = img_header.shape[2];
        size_t channels = (img_header.shape.size() > 3) ? img_header.shape[3] : 1;
        size_t pixels_per_image = height * width * channels;
        
        std::cout << "NPY Images: " << num_images << " images of " << height << "x" << width;
        if (channels > 1) std::cout << "x" << channels;
        std::cout << std::endl;
        
        // Leer datos de imagen
        if (img_header.dtype == "<f4" || img_header.dtype == "<f8") {
            // Datos en formato float
            for (size_t i = 0; i < num_images; i++) {
                std::vector<float> image(pixels_per_image);
                if (img_header.dtype == "<f4") {
                    std::vector<float> buffer(pixels_per_image);
                    images_file.read(reinterpret_cast<char*>(buffer.data()), pixels_per_image * sizeof(float));
                    image = buffer;
                } else { // f8 (double)
                    std::vector<double> buffer(pixels_per_image);
                    images_file.read(reinterpret_cast<char*>(buffer.data()), pixels_per_image * sizeof(double));
                    for (size_t j = 0; j < pixels_per_image; j++) {
                        image[j] = static_cast<float>(buffer[j]);
                    }
                }
                
                // Normalizar si es necesario (si los valores están en rango 0-255)
                bool needs_normalization = false;
                for (float val : image) {
                    if (val > 1.0f) {
                        needs_normalization = true;
                        break;
                    }
                }
                
                if (needs_normalization) {
                    for (float& val : image) {
                        val /= 255.0f;
                    }
                }
                
                data.images.push_back(image);
            }
        } else if (img_header.dtype == "|u1" || img_header.dtype == "u1") {
            // Datos en formato uint8
            for (size_t i = 0; i < num_images; i++) {
                std::vector<float> image(pixels_per_image);
                std::vector<uint8_t> buffer(pixels_per_image);
                images_file.read(reinterpret_cast<char*>(buffer.data()), pixels_per_image);
                for (size_t j = 0; j < pixels_per_image; j++) {
                    image[j] = static_cast<float>(buffer[j]) / 255.0f;
                }
                data.images.push_back(image);
            }
        } else {
            throw std::runtime_error("Unsupported dtype for images: " + img_header.dtype);
        }
        
        images_file.close();
        
        // Cargar etiquetas
        std::ifstream labels_file(labels_path, std::ios::binary);
        if (!labels_file.is_open()) {
            throw std::runtime_error("Failed to open labels file: " + labels_path);
        }
        
        NPYHeader lbl_header = parse_npy_header(labels_file);
        
        if (lbl_header.shape.size() != 1) {
            throw std::runtime_error("Invalid label shape in NPY file");
        }
        
        size_t num_labels = lbl_header.shape[0];
        std::cout << "NPY Labels: " << num_labels << " labels" << std::endl;
        
        // Leer etiquetas
        if (lbl_header.dtype == "<i4" || lbl_header.dtype == "<i8") {
            // Enteros de 32 o 64 bits
            for (size_t i = 0; i < num_labels; i++) {
                if (lbl_header.dtype == "<i4") {
                    int32_t label;
                    labels_file.read(reinterpret_cast<char*>(&label), sizeof(int32_t));
                    data.labels.push_back(static_cast<int>(label));
                } else {
                    int64_t label;
                    labels_file.read(reinterpret_cast<char*>(&label), sizeof(int64_t));
                    data.labels.push_back(static_cast<int>(label));
                }
            }
        } else if (lbl_header.dtype == "|u1" || lbl_header.dtype == "u1") {
            // Enteros de 8 bits
            for (size_t i = 0; i < num_labels; i++) {
                uint8_t label;
                labels_file.read(reinterpret_cast<char*>(&label), 1);
                data.labels.push_back(static_cast<int>(label));
            }
        } else {
            throw std::runtime_error("Unsupported dtype for labels: " + lbl_header.dtype);
        }
        
        labels_file.close();
        return data;
    }
    
    // Método de conveniencia para cargar OrganCMNIST completo (train + test)
    std::pair<MNISTData, MNISTData> load_organc_mnist(const std::string& data_dir) {
        std::string train_images = data_dir + "/X_train.npy";
        std::string train_labels = data_dir + "/y_train.npy";
        std::string test_images = data_dir + "/X_test.npy";
        std::string test_labels = data_dir + "/y_test.npy";
        
        MNISTData train_data = load_npy(train_images, train_labels);
        MNISTData test_data = load_npy(test_images, test_labels);
        
        return std::make_pair(train_data, test_data);
    }
    
    std::vector<std::vector<std::vector<float>>> create_patches(const std::vector<std::vector<float>>& images, int patch_size, int img_height = 28, int img_width = 28) {
        std::vector<std::vector<std::vector<float>>> patches;
        for (const auto& image : images) {
            std::vector<std::vector<float>> image_patches;
            for (int i = 0; i < img_height; i += patch_size) {
                for (int j = 0; j < img_width; j += patch_size) {
                    std::vector<float> patch;
                    for (int pi = 0; pi < patch_size; ++pi) {
                        for (int pj = 0; pj < patch_size; ++pj) {
                            int row = i + pi;
                            int col = j + pj;
                            if (row < img_height && col < img_width) {
                                int idx = row * img_width + col;
                                if (idx < image.size()) {
                                    patch.push_back(image[idx]);
                                } else {
                                    patch.push_back(0.0f); // Padding con ceros
                                }
                            } else {
                                patch.push_back(0.0f); // Padding con ceros
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
