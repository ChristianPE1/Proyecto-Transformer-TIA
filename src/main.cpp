#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "transformer/vit_mnist.hpp"
#include "data/mnist_loader.hpp"
#include "training/classification_loss.hpp"
#include "utils/trainer.hpp"

// Función para cargar imagen binaria procesada por Python
std::vector<float> load_processed_image(const std::string& filename) {
    std::cout << "Loading processed binary image: " << filename << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open processed image file: " << filename << std::endl;
        return std::vector<float>(28 * 28, 0.0f);
    }
    
    // Leer datos binarios de float
    std::vector<float> image_data(28 * 28);
    file.read(reinterpret_cast<char*>(image_data.data()), 28 * 28 * sizeof(float));
    
    if (file.gcount() != 28 * 28 * sizeof(float)) {
        std::cerr << "Error: File size incorrect for 28x28 float image" << std::endl;
        file.close();
        return std::vector<float>(28 * 28, 0.0f);
    }
    
    file.close();
    
    // Verificar rango de datos
    auto minmax = std::minmax_element(image_data.begin(), image_data.end());
    std::cout << "Image data range: [" << *minmax.first << ", " << *minmax.second << "]" << std::endl;
    
    // Contar píxeles blancos y negros
    int white_pixels = std::count_if(image_data.begin(), image_data.end(), [](float v) { return v > 0.5f; });
    std::cout << "White pixels: " << white_pixels << ", Black pixels: " << (784 - white_pixels) << std::endl;
    
    std::cout << "Successfully loaded processed image: " << filename << std::endl;
    return image_data;
}

// Función para clasificar una sola imagen
void classify_single_image(ViTMNIST& model, const std::string& image_path) {
    std::cout << "\n=== Single Image Classification ===" << std::endl;
    
    // Cargar imagen procesada por Python
    std::vector<float> image_data = load_processed_image(image_path);
    
    if (image_data.empty() || image_data.size() != 784) {
        std::cerr << "Error: Invalid image data" << std::endl;
        return;
    }
    
    // Crear matriz de la imagen
    Matrix image(image_data, 28, 28);
    
    // Forward pass
    Matrix logits = model.forward(image);
    
    // Obtener predicción
    std::vector<float> pred_data = logits.getDataVector();
    
    // Aplicar softmax para obtener probabilidades
    float max_logit = *std::max_element(pred_data.begin(), pred_data.end());
    float sum_exp = 0.0f;
    
    for (int i = 0; i < 10; ++i) {
        pred_data[i] = std::exp(pred_data[i] - max_logit);
        sum_exp += pred_data[i];
    }
    
    for (int i = 0; i < 10; ++i) {
        pred_data[i] /= sum_exp;
    }
    
    // Encontrar la clase con mayor probabilidad
    int predicted_class = std::max_element(pred_data.begin(), pred_data.end()) - pred_data.begin();
    float confidence = pred_data[predicted_class];
    
    // Mostrar resultados
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Predicted digit: " << predicted_class << std::endl;
    std::cout << "Confidence: " << (confidence * 100.0f) << "%" << std::endl;
    
    // Mostrar top 3 predicciones
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < 10; ++i) {
        scores.push_back({pred_data[i], i});
    }
    
    std::sort(scores.rbegin(), scores.rend()); // Ordenar descendente
    
    std::cout << "Top 3 predictions:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "  " << scores[i].second << ": " << (scores[i].first * 100.0f) << "%" << std::endl;
    }
}

// Función para buscar y clasificar todas las imágenes procesadas
void classify_all_processed_images(ViTMNIST& model) {
    std::cout << "\n=== Clasificando todas las imágenes procesadas ===" << std::endl;
    
    std::string processed_dir = "processed_images";
    
    // Verificar si existe el directorio
    if (!std::filesystem::exists(processed_dir)) {
        std::cout << "❌ No se encontró el directorio: " << processed_dir << std::endl;
        std::cout << "Por favor ejecuta: python3 convert_images.py" << std::endl;
        return;
    }
    
    // Buscar todos los archivos .bin en el directorio
    std::vector<std::string> bin_files;
    for (const auto& entry : std::filesystem::directory_iterator(processed_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            bin_files.push_back(entry.path().string());
        }
    }
    
    if (bin_files.empty()) {
        std::cout << "❌ No se encontraron archivos .bin en " << processed_dir << std::endl;
        std::cout << "Por favor ejecuta: python3 convert_images.py" << std::endl;
        return;
    }
    
    // Ordenar archivos alfabéticamente
    std::sort(bin_files.begin(), bin_files.end());
    
    std::cout << "Encontrados " << bin_files.size() << " archivos de imágenes procesadas:" << std::endl;
    
    // Clasificar cada imagen
    for (const auto& file_path : bin_files) {
        classify_single_image(model, file_path);
    }
}

int main(int argc, char** argv) {

    std::cout << "=== Vision Transformer for MNIST Classification ===" << std::endl;

    std::cout << "CPU version" << std::endl;

    std::cout << "Loading MNIST dataset..." << std::endl;
    MNISTLoader loader;
    std::string train_images_path = "data/train-images.idx3-ubyte";
    std::string train_labels_path = "data/train-labels.idx1-ubyte";
    std::string test_images_path = "data/t10k-images.idx3-ubyte";
    std::string test_labels_path = "data/t10k-labels.idx1-ubyte";

    if (argc == 5) {
        train_images_path = argv[1];
        train_labels_path = argv[2];
        test_images_path = argv[3];
        test_labels_path = argv[4];
    }

    // Cargar datos de entrenamiento
    MNISTData train_data = loader.load(train_images_path, train_labels_path);
    std::cout << "Loaded training data: " << train_data.images.size() << " images and "
        << train_data.labels.size() << " labels" << std::endl;

    // Cargar datos de test
    MNISTData test_data = loader.load(test_images_path, test_labels_path);
    std::cout << "Loaded test data: " << test_data.images.size() << " images and "
        << test_data.labels.size() << " labels" << std::endl;

    int patch_size = 7;  // Patches más grandes 
    int embed_dim = 64;   // Dimensión más grande pero manejable
    int num_heads = 2;    // 2 heads
    int num_layers = 3;   // 2 capas
    int mlp_hidden_layers_size = 96;
    int num_classes = 10;

    ViTMNIST vit_model(
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        mlp_hidden_layers_size,
        num_classes);

    std::cout << "Vision Transformer for MNIST initialized" << std::endl;

    // Training parameters - configuracion para mejor aprendizaje
    int num_epochs = 10;
    int batch_size = 64;
    float learning_rate = 0.001f;
    int save_each_epoch = 5;

    Trainer trainer(num_epochs, batch_size, learning_rate);
    //trainer.train(vit_model, train_data, test_data, save_each_epoch);
    
    // Cargar pesos del modelo entrenado
    std::cout << "Loading trained model weights..." << std::endl;
    vit_model.load_weights("./vit-10.bin");
    std::cout << "Model weights loaded successfully!" << std::endl;
    
    // Clasificar todas las imágenes procesadas automáticamente
    classify_all_processed_images(vit_model);

    return 0;
}