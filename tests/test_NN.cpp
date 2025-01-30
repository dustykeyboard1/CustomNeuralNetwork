#include <iostream>
#include "NeuralNet.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

std::vector<float> loadCSV(const std::string& filename, int& numDays, int& numFeatures) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
        return std::vector<float>();
    }
    
    std::cout << "File opened successfully" << std::endl;
    std::string line;
    std::vector<float> data;
    
    // Skip header line
    std::getline(file, line);
    std::cout << "Header line: " << line << std::endl;
    
    // Read data lines
    int lineCount = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        // Skip date column
        std::getline(ss, value, ',');
        
        // Read in order: Close,Volume,Open,High,Low
        std::getline(ss, value, ','); float close = std::stof(value);
        std::getline(ss, value, ','); float volume = std::stof(value);
        std::getline(ss, value, ','); float open = std::stof(value);
        std::getline(ss, value, ','); float high = std::stof(value);
        std::getline(ss, value, ','); float low = std::stof(value);
        
        // Store in OHLC order
        data.push_back(open);
        data.push_back(high);
        data.push_back(low);
        data.push_back(close);
        data.push_back(volume);
        
        lineCount++;
    }
    
    // Calculate dimensions
    numFeatures = 5; // OHLC = 4 features
    numDays = lineCount;
    
    return data;
}

void testNeuralNetInitialization() {
    int numDays, numFeatures;
    std::vector<float> trainingData = loadCSV("C:/Users/Michael/Downloads/HistoricalData_1738275321928.csv", numDays, numFeatures);
    
    // Print some debug info
    std::cout << "Loaded " << numDays << " days of data with " << numFeatures << " features each." << std::endl;
    
    // Create and initialize the neural network
    NeuralNet net;
    
    // Example configuration
    int lookback = 5;  // Use 5 days of history to predict
    int numPredictions = 2;  // Predict both high and low
    int batchSize = 32;
    float learningRate = 0.001f;
    int numEpochs = 1;
    
    // Define network architecture
    int hiddenLayers = 2;
    int neurons[] = {8, 4};  // Two hidden layers with 8 and 4 neurons
    
    // Initialize network
    net.initialize(lookback * numFeatures, neurons, hiddenLayers, numPredictions);
    
    // Define which indices we want to predict (high=1, low=2 in our OHLC order)
    std::vector<int> targetIndices = {1, 2};  // High and Low indices
    
    // Train the network
    net.train(trainingData.data(),  // Pointer to the training data
             numDays,               // Total number of days
             lookback,              // Number of days to look back
             numFeatures,           // Number of features (OHLC = 4)
             numPredictions,        // Number of values to predict (2 for high/low)
             batchSize,
             learningRate,
             numEpochs,
             targetIndices.data()); // Pass the target indices
}

// void testTrainInitialization() {
//     const int numSamples = 10;
//     const int numFeatures = 5;
//     const int hiddenSize = 16;
//     const int outputFeatures = 2;
//     const int batchSize = 5;
//     const int epochs = 1;
//     const float learningRate = 0.01f;

//     // Define inputs as a 10x5 feature matrix (flattened to 1D array)
//     float inputs[numSamples * numFeatures] = {
//         0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f, 1.6f, 1.8f, 2.0f,
//         2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f,
//         4.2f, 4.4f, 4.6f, 4.8f, 5.0f, 5.2f, 5.4f, 5.6f, 5.8f, 6.0f,
//         6.2f, 6.4f, 6.6f, 6.8f, 7.0f, 7.2f, 7.4f, 7.6f, 7.8f, 8.0f,
//         8.2f, 8.4f, 8.6f, 8.8f, 9.0f, 9.2f, 9.4f, 9.6f, 9.8f, 10.0f
//     };

//     // Define regression targets for 10 samples with 2 outputs per sample
//     float targets[numSamples * outputFeatures] = {
//         1.5f, 2.5f, 2.0f, 3.0f, 2.5f, 3.5f, 3.0f, 4.0f, 3.5f, 4.5f,
//         4.0f, 5.0f, 4.5f, 5.5f, 5.0f, 6.0f, 5.5f, 6.5f, 6.0f, 7.0f
//     };

//     // Initialize the neural network
//     NeuralNet nn;
//     nn.initialize(numSamples, numFeatures, hiddenSize, outputFeatures);

//     // Debug: Verify the training process
//     std::cout << "[TEST] Starting training for 1 epoch with debug statements.\n";
//     nn.train(inputs, targets, numSamples, numFeatures, batchSize, epochs, learningRate);
//     std::cout << "[TEST] Training process completed successfully.\n";
// }

int main() {
    testNeuralNetInitialization();
    // std::cout << "------------Train Test---------------" << std::endl;
    // testTrainInitialization();
    return 0;
}
