#include "trainer.hpp"

Trainer::Trainer(
    int num_epochs,
    int batch_size,
    float learning_rate)
    : gen(std::random_device{}()),
    num_epochs(num_epochs),
    batch_size(batch_size),
    learning_rate(learning_rate) {}

void Trainer::train(
    ViTMNIST& model,
    const MNISTData& train_data,
    const MNISTData& test_data,
    int save_each_epoch) {

    if (train_data.images.size() != train_data.labels.size() || 
        test_data.images.size() != test_data.labels.size())
        throw std::invalid_argument("E");

    std::cout << "Training parameters:" << std::endl;
    std::cout << "  Epochs: " << num_epochs << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;

    // archivo para guardar el accuracy
    std::ofstream metrics_file("metrics.csv", std::ofstream::trunc);
    float best_test_accuracy{};

    if (!metrics_file.is_open()) {
        std::cerr << "Warning: Could not open metrics file for writing" << std::endl;
    }

    metrics_file << "Epoch,Train_Loss,Train_Accuracy,Test_Accuracy\n";

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << num_epochs << std::endl;

        float epoch_loss = 0.0f;
        int num_batches = 0;
        int correct_predictions = 0;
        int total_samples = 0;

        // Random Indices
        std::vector<size_t> random_indices(train_data.images.size());
        std::iota(random_indices.begin(), random_indices.end(), 0);
        std::shuffle(random_indices.begin(), random_indices.end(), gen);

        for (size_t i = 0; i < train_data.images.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, train_data.images.size());
            size_t current_batch_size = batch_end - i;

            // Create batch matrices
            std::vector<int> batch_index(current_batch_size);

            // Fill batch data
            for (size_t j = 0; j < current_batch_size; ++j) {
                size_t img_idx = i + j;
                batch_index[j] = random_indices[img_idx];
            }
            
            float batch_loss = 0.0f;
            int batch_correct = 0;

            // Procesa cada imagen en el lote
            for (size_t j = 0; j < current_batch_size; ++j) {
                // Extract single image from batch
                Matrix image(train_data.images[batch_index[j]], 28, 28);

                // Forward pass
                Matrix logits = model.forward(image);

                // Get predicted class (simple por ahora)
                std::vector<float> pred_data = logits.getDataVector();
                int predicted_class = std::max_element(pred_data.begin(), pred_data.end()) - pred_data.begin();

                if (predicted_class == train_data.labels[batch_index[j]]) {
                    batch_correct++;
                }

                // Compute loss
                std::vector<int> single_label = { train_data.labels[batch_index[j]] };
                float loss = CrossEntropyLoss::compute_loss(logits, single_label);

                if (!std::isnan(loss) && !std::isinf(loss) && loss < 100.0f) {
                    batch_loss += loss;

                    // Compute gradients and backward pass
                    Matrix grad = CrossEntropyLoss::compute_gradients(logits, single_label);
                    model.backward(grad);
                }
            }

            // Update weights once per batch
            model.update_weights(learning_rate);

            // Update totals
            correct_predictions += batch_correct;
            total_samples += current_batch_size;
            epoch_loss += batch_loss;

            num_batches++;
        }

        float avg_epoch_loss = epoch_loss / total_samples;
        float train_accuracy = (float)correct_predictions / total_samples;

        // Evaluar en datos de test
        float test_accuracy = evaluate_model(model, test_data);

        std::cout << "Epoch " << (epoch + 1) << " completed." << std::endl;
        std::cout << "  Training - Loss: ";

        if (std::isnan(avg_epoch_loss) || std::isinf(avg_epoch_loss)) {
            std::cout << "NaN (ERROR!)" << std::endl;
        }
        else {
            std::cout << avg_epoch_loss << ", Accuracy: " << train_accuracy * 100.0f << "%" << std::endl;
        }
        std::cout << "  Test Accuracy: " << test_accuracy * 100.f << "%" << std::endl;

        // Log metrics to file
        if (metrics_file.is_open()) {
            metrics_file << (epoch + 1) << "," << avg_epoch_loss << ","
                << train_accuracy << "," << test_accuracy << "\n";
            metrics_file.flush();
        }

        if ((epoch + 1) % save_each_epoch == 0) {
            model.save_weights("vit-" + std::to_string(epoch + 1) + ".bin");
        }
    }

    std::cout << "\nTraining completed!" << std::endl;

    // Close metrics file
    if (metrics_file.is_open()) {
        metrics_file.close();
        std::cout << "Training and test metrics saved to accuracy.csv" << std::endl;
    }
}

float Trainer::evaluate_model(ViTMNIST& model, const MNISTData& test_data) {

    if (test_data.images.size() != test_data.labels.size())
        throw std::invalid_argument("E");

    int correct_predictions = 0;
    int total_samples = 0;
    size_t max_test_samples = test_data.images.size();

    std::cout << "Evaluating on test data..." << std::endl;

    for (size_t i = 0; i < std::min(max_test_samples, test_data.images.size()); ++i) {
        Matrix image(test_data.images[i], 28, 28);

        // Forward pass (solo inferencia, sin backward)
        Matrix logits = model.forward(image);

        // Get predicted class
        std::vector<float> pred_data = logits.getDataVector();
        int predicted_class = std::max_element(pred_data.begin(), pred_data.end()) - pred_data.begin();

        if (predicted_class == test_data.labels[i]) {
            correct_predictions++;
        }
        total_samples++;
    }

    float test_accuracy = (float)correct_predictions / total_samples;
    return test_accuracy;
}




