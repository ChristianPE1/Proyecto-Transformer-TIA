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
#include "utils/classifier.hpp"  

ViTMNIST train_mnist();

ViTMNIST train_fashion();


int main(int argc, char** argv) {
    std::cout << "=== Vision Transformer for MNIST Classification ===" << std::endl;
    std::cout << "CPU version" << std::endl;


    train_fashion();

    /*
    try {
        auto model = ViTMNIST::load_pretrained_model();
        std::cout << "Modelo preentrenado cargado exitosamente" << std::endl;
   
        // Crear instancia del clasificador
        Classifier classifier(model);

        // Configurar verbosidad (opcional)
        classifier.setVerbose(true);

        std::vector<ClassificationResult> all_results = classifier.classifyAllProcessedImages(std::string(IMAGES_DIR));

        // Mostrar resumen de resultados
        classifier.printSummaryResults(all_results);

    }
    catch (const std::exception& e) {
        std::cerr << "Error cargando modelo preentrenado: " << e.what() << std::endl;
    }
    */

    return 0;
}

ViTMNIST train_mnist() {
    std::cout << "Loading MNIST dataset..." << std::endl;
    MNISTLoader loader;
    std::string data_path = std::string(DATA_DIR);
    std::string train_images_path = data_path + "mnist/train-images.idx3-ubyte";
    std::string train_labels_path = data_path + "mnist/train-labels.idx1-ubyte";
    std::string test_images_path = data_path + "mnist/t10k-images.idx3-ubyte";
    std::string test_labels_path = data_path +"mnist/t10k-labels.idx1-ubyte";

    // Cargar datos de entrenamiento
    MNISTData train_data = loader.load(train_images_path, train_labels_path);
    std::cout << "Loaded training data: " << train_data.images.size() << " images and "
        << train_data.labels.size() << " labels" << std::endl;

    // Cargar datos de test
    MNISTData test_data = loader.load(test_images_path, test_labels_path);
    std::cout << "Loaded test data: " << test_data.images.size() << " images and "
        << test_data.labels.size() << " labels" << std::endl;

    int patch_size = 7;
    int embed_dim = 64;
    int num_heads = 2;
    int num_layers = 3;
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

    // Training parameters
    int num_epochs = 10;
    int batch_size = 64;
    float learning_rate = 0.001f;
    int save_each_epoch = 5;

    Trainer trainer(num_epochs, batch_size, learning_rate);
    trainer.train(vit_model, train_data, test_data, save_each_epoch);

    return vit_model;
}

ViTMNIST train_fashion() {
    std::cout << "Loading MNIST dataset..." << std::endl;
    MNISTLoader loader;
    std::string data_path = std::string(DATA_DIR);
    std::string train_images_path = data_path + "fashion/train-images-idx3-ubyte";
    std::string train_labels_path = data_path + "fashion/train-labels-idx1-ubyte";
    std::string test_images_path = data_path + "fashion/t10k-images-idx3-ubyte";
    std::string test_labels_path = data_path + "fashion/t10k-labels-idx1-ubyte";

    // Cargar datos de entrenamiento
    MNISTData train_data = loader.load(train_images_path, train_labels_path);
    std::cout << "Loaded training data: " << train_data.images.size() << " images and "
        << train_data.labels.size() << " labels" << std::endl;

    // Cargar datos de test
    MNISTData test_data = loader.load(test_images_path, test_labels_path);
    std::cout << "Loaded test data: " << test_data.images.size() << " images and "
        << test_data.labels.size() << " labels" << std::endl;

    int patch_size = 7;
    int embed_dim = 64;
    int num_heads = 2;
    int num_layers = 3;
    int mlp_hidden_layers_size = 32;
    int num_classes = 10;

    ViTMNIST vit_model(
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        mlp_hidden_layers_size,
        num_classes);

    std::cout << "Vision Transformer for MNIST initialized" << std::endl;

    // Training parameters
    int num_epochs = 10;
    int batch_size = 64;
    float learning_rate = 0.001f;
    int save_each_epoch = 1;

    Trainer trainer(num_epochs, batch_size, learning_rate);
    trainer.train(vit_model, train_data, test_data, save_each_epoch);

    return vit_model;
}

