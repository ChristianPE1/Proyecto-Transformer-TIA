#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <vector>
#include <string>

/**
 * Utility functions for image loading and processing
 */
namespace ImageUtils {
    
    /**
     * Load a PNG image and convert it to grayscale
     * @param filename Path to the PNG file
     * @return Vector of pixel values (0-255), empty if loading failed
     */
    std::vector<unsigned char> loadPNG(const std::string& filename);
    
    /**
     * Convert grayscale image to MNIST format (normalized 0-1, inverted if needed)
     * @param pixels Raw pixel data (0-255)
     * @param width Image width
     * @param height Image height
     * @return Vector of normalized pixel values suitable for MNIST model
     */
    std::vector<float> preprocessForMNIST(const std::vector<unsigned char>& pixels, int width, int height);
    
    /**
     * Resize image to 28x28 using simple nearest neighbor interpolation
     * @param pixels Input pixel data
     * @param srcWidth Original width
     * @param srcHeight Original height
     * @return Resized pixel data (28x28)
     */
    std::vector<unsigned char> resizeTo28x28(const std::vector<unsigned char>& pixels, int srcWidth, int srcHeight);
}

#endif // IMAGE_UTILS_HPP
