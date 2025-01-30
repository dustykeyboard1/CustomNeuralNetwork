#include "Utils.h"
#include <random>
#include <algorithm>
#include <iostream>

std::vector<int> Utils::generateShuffledIndices(int size) {
    // Create vector without shuffling first
    std::vector<int> indices;
    
    // Reserve space
    indices.reserve(size);
    
    // Fill with sequential numbers
    for (int i = 0; i < size; i++) {
        indices.push_back(i);
    }
    
    try {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        std::shuffle(indices.begin(), indices.end(), gen);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during shuffle: " << e.what() << std::endl;
        throw;
    }
    
    return indices;
}
