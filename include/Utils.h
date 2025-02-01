#pragma once
#include <vector>
#include <string>

struct DataSplit {
    float* trainData;
    float* validData;
    float* testData;
    int trainSize;
    int validSize;
    int testSize;
};

namespace Utils {
    std::vector<int> generateShuffledIndices(int minIdx, int maxIdx);
    void writeLossToFile(const std::vector<float>& losses, const std::string& filename);
    void printProgress(int current, int total, float loss);
    DataSplit splitDataset(const float* data, int totalSamples, int numFeatures);
}