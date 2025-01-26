#ifndef LOSS_OPS_H
#define LOSS_OPS_H

namespace LossOps {
    float MeanSquaredError(const float* yTrue, const float* yPred, int size);
    float gpuCrossEntropyLoss(const float* yTrue, const float* yPred, int batchSize, int numClasses);
}

#endif
