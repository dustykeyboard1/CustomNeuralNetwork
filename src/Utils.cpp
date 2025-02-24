#include "Utils.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

namespace Utils {
// Generate random indices for batch processing
    std::vector<int> Utils::generateShuffledIndices(int minIdx, int maxIdx) {
        std::vector<int> indices;
        // Only include indices that have enough history and future space
        for (int i = minIdx; i <= maxIdx; i++) {
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

    // Save training loss history to CSV file
    void Utils::writeLossToFile(const std::vector<float>& losses, const std::string& filename) {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        outFile << "epoch,loss\n";
        
        for (size_t i = 0; i < losses.size(); ++i) {
            outFile << i << "," << losses[i] << "\n";
        }
        
        outFile.close();
    }

    // Display training progress with progress bar and loss
    void Utils::printProgress(int current, int total, float loss) {
        const int barWidth = 50;
        float progress = (float)current / total;
        int pos = barWidth * progress;

        std::cout << "\r[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "% "
                << "Loss: " << std::fixed << std::setprecision(4) << loss << std::flush;
        
        if (current == total) std::cout << std::endl;
    }

    DataSplit Utils::splitDataset(const float* data, int totalSamples, int numFeatures) {
        // Calculate sizes for each split
        int trainSize = static_cast<int>(0.8f * totalSamples);
        int validSize = static_cast<int>(0.1f * totalSamples);
        int testSize = totalSamples - trainSize - validSize;
        
        // Allocate memory for each split
        float* trainData = new float[trainSize * numFeatures];
        float* validData = new float[validSize * numFeatures];
        float* testData = new float[testSize * numFeatures];
        
        // Copy data into splits
        std::memcpy(trainData, data, trainSize * numFeatures * sizeof(float));
        std::memcpy(validData, 
                    data + (trainSize * numFeatures), 
                    validSize * numFeatures * sizeof(float));
        std::memcpy(testData, 
                    data + ((trainSize + validSize) * numFeatures), 
                    testSize * numFeatures * sizeof(float));
        
        return {
            trainData,
            validData,
            testData,
            trainSize,
            validSize,
            testSize
        };
    }
}