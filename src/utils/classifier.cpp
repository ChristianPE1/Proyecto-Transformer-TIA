#include "classifier.hpp"


std::vector<float> Classifier::loadProcessedImage(const std::string& filename) {
    if (verbose) {
        std::cout << "Cargando imagen procesada: " << filename << std::endl;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (verbose) {
            std::cerr << "Error: No se puede abrir el archivo: " << filename << std::endl;
        }
        return std::vector<float>();
    }

    // Leer datos binarios de float
    std::vector<float> image_data(28 * 28);
    file.read(reinterpret_cast<char*>(image_data.data()), 28 * 28 * sizeof(float));

    if (file.gcount() != 28 * 28 * sizeof(float)) {
        if (verbose) {
            std::cerr << "Error: Tamano de archivo incorrecto para imagen 28x28 float" << std::endl;
        }
        file.close();
        return std::vector<float>();
    }

    file.close();
    return image_data;
}

ImageStats Classifier::analyzeImageData(const std::vector<float>& image_data) {
    ImageStats stats;

    if (image_data.empty()) {
        return stats;
    }

    auto minmax = std::minmax_element(image_data.begin(), image_data.end());
    stats.min_value = *minmax.first;
    stats.max_value = *minmax.second;
    stats.white_pixels = std::count_if(image_data.begin(), image_data.end(), [](float v) { return v > 0.5f; });
    stats.black_pixels = image_data.size() - stats.white_pixels;
    stats.total_pixels = image_data.size();

    return stats;
}

std::vector<float> Classifier::applySoftmax(const std::vector<float>& logits) {
    std::vector<float> probabilities = logits;

    // Aplicar softmax para obtener probabilidades
    float max_logit = *std::max_element(probabilities.begin(), probabilities.end());
    float sum_exp = 0.0f;

    for (int i = 0; i < 10; ++i) {
        probabilities[i] = std::exp(probabilities[i] - max_logit);
        sum_exp += probabilities[i];
    }

    for (int i = 0; i < 10; ++i) {
        probabilities[i] /= sum_exp;
    }

    return probabilities;
}

std::vector<std::pair<float, int>> Classifier::getTopPredictions(const std::vector<float>& probabilities, int top_k) {
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < static_cast<int>(probabilities.size()); ++i) {
        scores.push_back({ probabilities[i], i });
    }

    std::sort(scores.rbegin(), scores.rend()); // Ordenar descendente

    // Retornar solo los top_k
    if (static_cast<int>(scores.size()) > top_k) {
        scores.resize(top_k);
    }

    return scores;
}

ClassificationResult Classifier::classifySingleImage(const std::string& image_path) {
    ClassificationResult result;
    result.image_path = image_path;
    result.success = false;

    // Cargar imagen procesada
    std::vector<float> image_data = loadProcessedImage(image_path);

    if (image_data.empty() || image_data.size() != 784) {
        result.error_message = "Datos de imagen invalidos";
        return result;
    }

    return classifyImageData(image_data, image_path);
}

ClassificationResult Classifier::classifyImageData(const std::vector<float>& image_data, const std::string& image_name) {
    ClassificationResult result;
    result.image_path = image_name;
    result.success = false;

    if (image_data.size() != 784) {
        result.error_message = "Tamano de imagen incorrecto (debe ser 784 pixeles)";
        return result;
    }

    try {
        // Crear matriz de la imagen
        Matrix image(image_data, 28, 28);

        // Forward pass
        Matrix logits = model.forward(image);

        // Obtener prediccion
        std::vector<float> logits_data = logits.getDataVector();

        // Aplicar softmax
        std::vector<float> probabilities = applySoftmax(logits_data);

        // Encontrar la clase con mayor probabilidad
        int predicted_class = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
        float precision = probabilities[predicted_class];

        // Obtener top predictions
        std::vector<std::pair<float, int>> top_predictions = getTopPredictions(probabilities, 3);

        // Llenar resultado
        result.predicted_class = predicted_class;
        result.precision = precision;
        result.top_predictions = top_predictions;
        result.success = true;

        if (verbose && !image_name.empty()) {
            ImageStats stats = analyzeImageData(image_data);
            std::cout << "Rango de datos: [" << stats.min_value << ", " << stats.max_value << "]" << std::endl;
            std::cout << "PPixeles blancos: " << stats.white_pixels << ", negros: " << stats.black_pixels << std::endl;
        }

    }
    catch (const std::exception& e) {
        result.error_message = std::string("Error durante clasificacion: ") + e.what();
    }

    return result;
}

std::vector<ClassificationResult> Classifier::classifyAllProcessedImages(const std::string& processed_dir) {
    std::vector<ClassificationResult> results;

    if (verbose) {
        std::cout << "\n=== Clasificando todas las imagenes procesadas ===" << std::endl;
    }

    // Verificar si existe el directorio
    if (!std::filesystem::exists(processed_dir)) {
        if (verbose) {
            std::cout << "? No se encontro el directorio: " << processed_dir << std::endl;
            std::cout << "Por favor ejecuta: python3 convert_images.py" << std::endl;
        }
        return results;
    }

    // Buscar todos los archivos .bin en el directorio
    std::vector<std::string> bin_files;
    for (const auto& entry : std::filesystem::directory_iterator(processed_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            bin_files.push_back(entry.path().string());
        }
    }

    if (bin_files.empty()) {
        if (verbose) {
            std::cout << "? No se encontraron archivos .bin en " << processed_dir << std::endl;
            std::cout << "Por favor ejecuta: python3 convert_images.py" << std::endl;
        }
        return results;
    }

    // Ordenar archivos alfab�ticamente
    std::sort(bin_files.begin(), bin_files.end());

    if (verbose) {
        std::cout << "Encontrados " << bin_files.size() << " archivos de imagenes procesadas:" << std::endl;
    }

    // Clasificar cada imagen
    for (const auto& file_path : bin_files) {
        ClassificationResult result = classifySingleImage(file_path);
        results.push_back(result);

        if (verbose) {
            printClassificationResult(result);
        }
    }

    return results;
}

void Classifier::printClassificationResult(const ClassificationResult& result) {
    if (!result.success) {
        std::cout << "? Error clasificando " << result.image_path << ": " << result.error_message << std::endl;
        return;
    }

    std::cout << "\n--- Resultado de Clasificacion ---" << std::endl;
    std::cout << "Imagen: " << result.image_path << std::endl;
    std::cout << "Digito predicho: " << result.predicted_class << std::endl;
    std::cout << "Precision: " << (result.precision * 100.0f) << "%" << std::endl;

    std::cout << "Top 3 predicciones:" << std::endl;
    for (const auto& prediction : result.top_predictions) {
        std::cout << "  " << prediction.second << ": " << (prediction.first * 100.0f) << "%" << std::endl;
    }
}

void Classifier::printSummaryResults(const std::vector<ClassificationResult>& results) {
    std::cout << "\n=== Resumen de Resultados ===" << std::endl;

    int successful = 0;
    int failed = 0;
    std::vector<int> class_counts(10, 0);

    for (const auto& result : results) {
        if (result.success) {
            successful++;
            class_counts[result.predicted_class]++;
        }
        else {
            failed++;
        }
    }

    std::cout << "Total de imagenes: " << results.size() << std::endl;
    std::cout << "Clasificadas exitosamente: " << successful << std::endl;
    std::cout << "Fallidas: " << failed << std::endl;

    if (successful > 0) {
        std::cout << "\nDistribucion de predicciones:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            if (class_counts[i] > 0) {
                std::cout << "Digito " << i << ": " << class_counts[i] << " predicciones" << std::endl;
            }
        }
    }
}

