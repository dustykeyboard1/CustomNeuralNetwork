#pragma once
#include <vector>
#include <string>

class Utils {
public:
    static std::vector<int> generateShuffledIndices(int size);
    static void writeLossToFile(const std::vector<float>& losses, const std::string& filename);
    static void printProgress(int current, int total, float loss);
};
