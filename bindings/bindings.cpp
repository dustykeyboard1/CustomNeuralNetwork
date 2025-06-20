#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "NeuralNet.h"

namespace py = pybind11;

PYBIND11_MODULE(neuralnet, m)
{
    py::class_<NeuralNet>(m, "NeuralNet")
        .def(py::init<int, int, std::vector<int>, int>(),  // optional constructor
             py::arg("inputSize"), py::arg("outputSize"), py::arg("hiddenLayerSizes"),
             py::arg("batchSize"))
        .def("initialize", &NeuralNet::initialize, py::arg("inputSize"), py::arg("outputSize"),
             py::arg("hiddenLayerSizes"), py::arg("batchSize"))
        .def(
            "train",
            [](NeuralNet& self, const std::vector<float>& inputs, const std::vector<float>& targets,
               int numSamples, int inputSize, int outputSize, int batchSize, float learningRate,
               int numEpochs, float clipThreshold, float decayRate,
               const std::vector<int>& batchIndices)
            {
                return self.train(inputs.data(), targets.data(), numSamples, inputSize, outputSize,
                                  batchSize, learningRate, numEpochs, clipThreshold, decayRate,
                                  batchIndices);
            },
            py::arg("inputs"), py::arg("targets"), py::arg("numSamples"), py::arg("inputSize"),
            py::arg("outputSize"), py::arg("batchSize"), py::arg("learningRate"),
            py::arg("numEpochs"), py::arg("clipThreshold"), py::arg("decayRate"),
            py::arg("batchIndices"));
}