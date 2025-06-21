#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "NeuralNet.h"

namespace py = pybind11;

PYBIND11_MODULE(neuralnet, m)
{
    py::class_<NeuralNet>(m, "NeuralNet")
        .def(py::init<>())
        .def("initialize", &NeuralNet::initialize, py::arg("input_size"),
             py::arg("hidden_layer_sizes"), py::arg("output_features"))
        .def(
            "train",
            [](NeuralNet& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> inputs,
               py::array_t<float, py::array::c_style | py::array::forcecast> targets,
               int numSamples, int inputSize, int outputSize, int batchSize, float learningRate,
               int numEpochs, float clipThreshold, float decayRate,
               const std::vector<int>& batchIndices)
            {
                const float* inputs_ptr = inputs.data();
                const float* targets_ptr = targets.data();

                return self.train(inputs_ptr, targets_ptr, numSamples, inputSize, outputSize,
                                  batchSize, learningRate, numEpochs, clipThreshold, decayRate,
                                  batchIndices);
            },
            py::arg("inputs"), py::arg("targets"), py::arg("numSamples"), py::arg("inputSize"),
            py::arg("outputSize"), py::arg("batchSize"), py::arg("learningRate"),
            py::arg("numEpochs"), py::arg("clipThreshold"), py::arg("decayRate"),
            py::arg("batchIndices"));
}
