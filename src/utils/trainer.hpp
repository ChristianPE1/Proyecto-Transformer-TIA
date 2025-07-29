#include <vector>
#include <fstream>
#include <string>
#include <numeric>
#include <random>
#include "data/mnist_loader.hpp"
#include "transformer/vit_mnist.hpp"
#include "training/classification_loss.hpp"


class Trainer {
    private:
        int num_epochs;
        int batch_size;
        float learning_rate;
        std::mt19937 gen;

    public:

        Trainer(int num_epochs, int batch_size, float learning_rate);
        void train(ViTMNIST& model, const MNISTData& train_data, const MNISTData& test_data, int save_each_epoch);


    private:
        // Funcion para evaluar el modelo en datos de test
        float evaluate_model(ViTMNIST& model, const MNISTData& test_data);

};