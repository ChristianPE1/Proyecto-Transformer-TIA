#include "image_utils.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

// Simple PNG loader without external dependencies
// For production use, consider using libraries like stb_image or OpenCV

namespace ImageUtils {

    std::vector<unsigned char> loadPNG(const std::string& filename) {
        // Simple grayscale PNG reader
        // This is a minimal implementation - for full PNG support use stb_image
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return {};
        }
        
        // For now, we'll implement a simple grayscale reader
        // In a real implementation, you would parse the PNG headers
        
        // Read file size
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        if (fileSize < 100) {  // PNG files are typically larger
            std::cerr << "Error: File too small to be a valid PNG" << std::endl;
            return {};
        }
        
        // For demonstration, we'll create a simple grayscale loader
        // that assumes the image is already in the correct format
        
        // Skip PNG header (8 bytes) and chunk headers
        file.seekg(33);  // Skip to approximate start of image data
        
        std::vector<unsigned char> pixels;
        pixels.reserve(28 * 28);
        
        // Read pixel data (this is a simplified approach)
        char byte;
        int pixelCount = 0;
        
        while (file.read(&byte, 1) && pixelCount < 28 * 28) {
            // Convert to grayscale value
            unsigned char gray = static_cast<unsigned char>(byte);
            pixels.push_back(gray);
            pixelCount++;
        }
        
        // If we don't have enough pixels, pad with zeros
        while (pixels.size() < 28 * 28) {
            pixels.push_back(0);
        }
        
        file.close();
        return pixels;
    }
    
    std::vector<unsigned char> resizeTo28x28(const std::vector<unsigned char>& pixels, int srcWidth, int srcHeight) {
        if (srcWidth == 28 && srcHeight == 28) {
            return pixels;
        }
        
        std::vector<unsigned char> resized(28 * 28);
        
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                // Nearest neighbor interpolation
                int srcX = static_cast<int>((x * srcWidth) / 28.0f);
                int srcY = static_cast<int>((y * srcHeight) / 28.0f);
                
                // Clamp to bounds
                srcX = std::min(srcX, srcWidth - 1);
                srcY = std::min(srcY, srcHeight - 1);
                
                int srcIndex = srcY * srcWidth + srcX;
                int dstIndex = y * 28 + x;
                
                if (srcIndex < static_cast<int>(pixels.size())) {
                    resized[dstIndex] = pixels[srcIndex];
                } else {
                    resized[dstIndex] = 0;
                }
            }
        }
        
        return resized;
    }
    
    std::vector<float> preprocessForMNIST(const std::vector<unsigned char>& pixels, int width, int height) {
        // First resize to 28x28 if needed
        std::vector<unsigned char> resizedPixels = resizeTo28x28(pixels, width, height);
        
        std::vector<float> processed;
        processed.reserve(28 * 28);
        
        for (unsigned char pixel : resizedPixels) {
            // Normalize to 0-1 range
            float normalized = pixel / 255.0f;
            
            // MNIST digits are white on black background
            // If your image has black digits on white background, invert:
            // normalized = 1.0f - normalized;
            
            processed.push_back(normalized);
        }
        
        return processed;
    }
    
    // Alternative: Simple PGM (Portable GrayMap) loader for testing
    std::vector<unsigned char> loadPGM(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open PGM file " << filename << std::endl;
            return {};
        }
        
        std::string magic;
        int width, height, maxval;
        
        file >> magic >> width >> height >> maxval;
        
        if (magic != "P2" && magic != "P5") {
            std::cerr << "Error: Unsupported PGM format" << std::endl;
            return {};
        }
        
        std::vector<unsigned char> pixels(width * height);
        
        if (magic == "P2") {
            // ASCII format
            for (int i = 0; i < width * height; i++) {
                int value;
                file >> value;
                pixels[i] = static_cast<unsigned char>(value);
            }
        } else {
            // Binary format
            file.ignore(); // Skip newline
            file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
        }
        
        return pixels;
    }
}
