#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "include/matrix.h"
#include "include/threads.h"

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " -threads N" << std::endl;
    std::cout << "  -threads N : maximum number of concurrent threads (required)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example: " << progName << " -threads 4" << std::endl;
}

int main(int argc, char* argv[]) {
    int maxThreads = 0;
    std::string progName = argc > 0 ? argv[0] : "conv_filter";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-threads" && i + 1 < argc) {
            try {
                maxThreads = std::stoi(argv[++i]);
                if (maxThreads <= 0) {
                    std::cerr << "Error: number of threads must be positive" << std::endl;
                    return 1;
                }
            } catch (...) {
                std::cerr << "Error: invalid thread count format" << std::endl;
                return 1;
            }
        }
    }

    if (maxThreads <= 0) {
        std::cerr << "Error: must specify number of threads" << std::endl;
        printUsage(progName.c_str());
        return 1;
    }

    try {
        std::cout << "=== Convolution Filter (Variant 13) ===" << std::endl;
        std::cout << "Max concurrent threads: " << maxThreads << std::endl;

        Matrix input;
        input.fillFromConsole();

        int windowSize;
        std::cout << "Enter window size (odd positive integer): ";
        std::cin >> windowSize;
        if (!std::cin || windowSize <= 0 || windowSize % 2 == 0) {
            std::cerr << "Error: window size must be a positive odd integer." << std::endl;
            return 1;
        }

        Matrix kernel(windowSize, windowSize);
        std::cout << "Enter convolution kernel elements (" << windowSize << "x" << windowSize 
                  << ") row by row (double values):" << std::endl;
        for (int r = 0; r < windowSize; ++r) {
            for (int c = 0; c < windowSize; ++c) {
                double v;
                std::cin >> v;
                kernel.setValue(r, c, v);
            }
        }

        int iterations;
        std::cout << "Enter number of filter applications (K > 0): ";
        std::cin >> iterations;
        if (!std::cin || iterations <= 0) {
            std::cerr << "Error: iterations must be positive." << std::endl;
            return 1;
        }

        std::cout << "\n--- Original Matrix ---\n";
        input.print();

        Matrix bufferA = input;
        Matrix bufferB(input.getRows(), input.getCols());

        ThreadControl threadManager(maxThreads);

        std::cout << "\n--- Processing ---\n";
        std::cout << "Window size: " << windowSize << ", Iterations: " << iterations 
                  << ", Matrix: " << input.getRows() << "x" << input.getCols() << std::endl;

        bool ok = threadManager.applyConvolution(bufferA, bufferB, kernel, iterations);
        if (!ok) {
            std::cerr << "Error: convolution failed." << std::endl;
            return 1;
        }
        Matrix& result = (iterations % 2 == 0) ? bufferA : bufferB;
        std::cout << "\n--- Result ---\n";
        result.print();

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
