#include "Layer.h"
#include <iostream>
#include <cassert>

void testLayerConnections() {
    Layer inputLayer(5, 10);
    Layer hiddenLayer(10, 7);
    Layer outputLayer(7, 1);

    inputLayer.setNextLayer(&hiddenLayer);
    hiddenLayer.setNextLayer(&outputLayer);

    // Verify connections
    assert(inputLayer.getNextLayer() == &hiddenLayer);
    assert(hiddenLayer.getPrevLayer() == &inputLayer);
    assert(hiddenLayer.getNextLayer() == &outputLayer);
    assert(outputLayer.getPrevLayer() == &hiddenLayer);

    std::cout << "Layer connections passed.\n";
}

int main() {
    testLayerConnections();
    std::cout << "Passed All tests" << std::endl;
    return 0;
}