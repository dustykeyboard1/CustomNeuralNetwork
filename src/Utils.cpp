#include "Utils.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

namespace Utils
{

// Display training progress with progress bar and loss
void Utils::printProgress(int current, int total, float loss)
{
    const int barWidth = 50;
    float progress = (float)current / total;
    int pos = barWidth * progress;

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << "Loss: " << std::fixed << std::setprecision(4) << loss << std::flush;

    if (current == total) std::cout << std::endl;
}
}  // namespace Utils