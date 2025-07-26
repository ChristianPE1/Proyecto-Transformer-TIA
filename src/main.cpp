#include <iostream>
#include <fstream>
#include<algorithm>
#include <cmath>
#include "transformer/vit_mnist.hpp"
#include "data/mnist_loader.hpp"
#include "training/classification_loss.hpp"


int main()
{
    try
    {
        std::cout << "=== Vision Transformer for MNIST Classification ===" << std::endl;

        std::cout << "CPU version" << std::endl;

        std::cout << "Loading MNIST dataset..." << std::endl;
        MNISTLoader loader;
        
        // Cargar datos de entrenamiento
        MNISTData train_data = loader.load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        std::cout << "Loaded training data: " << train_data.images.size() << " images and " 
                  << train_data.labels.size() << " labels" << std::endl;

        int patch_size = 14;  // Patches más grandes 
        int embed_dim = 64;   // Dimensión más grande pero manejable
        int num_heads = 2;    // 2 heads
        int num_layers = 2;   // 2 capas
        int num_classes = 10;

        ViTMNIST vit_model(patch_size, embed_dim, num_heads, num_layers, num_classes);
        std::cout << "Vision Transformer for MNIST initialized" << std::endl;

        // Training parameters - configuracion para mejor aprendizaje
        int num_epochs = 20;
        int batch_size = 64;
        float learning_rate = 0.001f;
        int max_samples = 6400;

        std::cout << "Training parameters:" << std::endl;
        std::cout << "  Epochs: " << num_epochs << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Learning rate: " << learning_rate << std::endl;
        
        // archivo para guardar el accuracy
        std::ofstream metrics_file("accuracy.txt");
        if (!metrics_file.is_open()) {
            std::cerr << "Warning: Could not open metrics file for writing" << std::endl;
        }
        metrics_file << "Epoch,Train_Loss,Train_Accuracy,Test_Accuracy,Train_Samples\n";

        // Training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\nEpoch " << (epoch + 1) << "/" << num_epochs << std::endl;
            
            float epoch_loss = 0.0f;
            int num_batches = 0;
            int correct_predictions = 0;
            int total_samples = 0;
            
            // Procesa imágenes en lotes (solo datos de entrenamiento)
            for (size_t i = 0; i < std::min((size_t)max_samples, train_data.images.size()); i += batch_size) {
                size_t batch_end = std::min(i + batch_size, train_data.images.size());
                size_t current_batch_size = batch_end - i;
                
                // Create batch matrices
                std::vector<float> batch_images(current_batch_size * 28 * 28);
                std::vector<int> batch_labels(current_batch_size);
                
                // Fill batch data
                for (size_t j = 0; j < current_batch_size; ++j) {
                    size_t img_idx = i + j;
                    std::copy(train_data.images[img_idx].begin(), 
                             train_data.images[img_idx].end(),
                             batch_images.begin() + j * 784);
                    batch_labels[j] = train_data.labels[img_idx];
                }
                
                float batch_loss = 0.0f;
                int batch_correct = 0;
                
                // Procesa cada imagen en el lote
                for (size_t j = 0; j < current_batch_size; ++j) {
                    // Extract single image from batch
                    std::vector<float> single_image(batch_images.begin() + j * 784,
                                                   batch_images.begin() + (j + 1) * 784);
                    
                    Matrix image(single_image, 28, 28);
                    
                    // Forward pass
                    Matrix logits = vit_model.forward(image);
                    
                    // Get predicted class (simple por ahora)
                    std::vector<float> pred_data = logits.getDataVector();
                    int predicted_class = std::max_element(pred_data.begin(), pred_data.end()) - pred_data.begin();
                    
                    if (predicted_class == batch_labels[j]) {
                        batch_correct++;
                    }
                    
                    // Compute loss
                    std::vector<int> single_label = {batch_labels[j]};
                    float loss = CrossEntropyLoss::compute_loss(logits, single_label);
                    
                    if (!std::isnan(loss) && !std::isinf(loss) && loss < 100.0f) {
                        batch_loss += loss;
                        
                        // Compute gradients and backward pass
                        Matrix grad = CrossEntropyLoss::compute_gradients(logits, single_label);
                        vit_model.backward(grad);
                    }
                }
                
                // Update weights once per batch
                vit_model.update_weights(learning_rate);
                
                // Update totals
                correct_predictions += batch_correct;
                total_samples += current_batch_size;
                epoch_loss += batch_loss;
                
                num_batches++;
                if (num_batches % 5 == 0 || num_batches == 1) {  // Report more frequently
                    float current_accuracy = (float)correct_predictions / total_samples * 100.0f;
                    std::cout << "Batch " << num_batches << ", samples: " << total_samples 
                              << ", avg_loss: " << (epoch_loss / total_samples) 
                              << ", acc: " << current_accuracy << "%" << std::endl;
                }
            }
            
            float avg_epoch_loss = epoch_loss / total_samples;
            float train_accuracy = (float)correct_predictions / total_samples * 100.0f;
            
            // Evaluar en datos de test
            
            std::cout << "Epoch " << (epoch + 1) << " completed." << std::endl;
            std::cout << "  Training - Loss: ";
            if (std::isnan(avg_epoch_loss) || std::isinf(avg_epoch_loss)) {
                std::cout << "NaN (ERROR!)" << std::endl;
            } else {
                std::cout << avg_epoch_loss << ", Accuracy: " << train_accuracy << "%" << std::endl;
            }
            
            // Log metrics to file
            if (metrics_file.is_open()) {
                metrics_file << (epoch + 1) << "," << avg_epoch_loss << "," 
                            << train_accuracy << "," << test_accuracy << "," << total_samples << "\n";
                metrics_file.flush();
            }
        }

        std::cout << "\nTraining completed!" << std::endl;
        
        // Close metrics file
        if (metrics_file.is_open()) {
            metrics_file.close();
            std::cout << "Training and test metrics saved to accuracy.txt" << std::endl;
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}