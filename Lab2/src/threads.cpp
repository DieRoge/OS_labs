#include "../include/threads.h"
#include <iostream>
#include <functional>
#include <cassert>

ThreadControl::ThreadControl(int maxThreads_) : maxThreads(maxThreads_) {
    if (maxThreads <= 0) throw std::invalid_argument("maxThreads must be positive");
}

ThreadControl::~ThreadControl() {
    for (auto &t : workers) {
        if (t.joinable()) t.join();
    }
}

void ThreadControl::workerFunc(const Matrix* src, Matrix* dst, const Matrix* kernel,
                               std::atomic<int>* nextRow, int totalRows) {
    while (true) {
        int row = nextRow->fetch_add(1, std::memory_order_relaxed);
        if (row >= totalRows) break;
        for (int col = 0; col < dst->getCols(); ++col) {
            double v = dst->applyConvolutionAt(row, col, *kernel, *src);
            dst->setValue(row, col, v);
        }
    }
}

bool ThreadControl::applyConvolution(Matrix& bufferA, Matrix& bufferB, const Matrix& kernel, int iterations) {
    if (bufferA.getRows() != bufferB.getRows() || bufferA.getCols() != bufferB.getCols()) {
        std::cerr << "Error: buffer sizes mismatch" << std::endl;
        return false;
    }
    if (kernel.getRows() != kernel.getCols() || kernel.getRows() % 2 == 0) {
        std::cerr << "Error: kernel must be square with odd size" << std::endl;
        return false;
    }
    if (iterations <= 0) {
        std::cerr << "Error: iterations must be positive" << std::endl;
        return false;
    }

    Matrix* src = &bufferA;
    Matrix* dst = &bufferB;

    int rows = src->getRows();
    int cols = src->getCols();

    for (int iter = 0; iter < iterations; ++iter) {
        std::atomic<int> nextRow(0);
        workers.clear();
        workers.reserve(maxThreads);

        for (int t = 0; t < maxThreads; ++t) {
            workers.emplace_back(&ThreadControl::workerFunc, this, src, dst, &kernel, &nextRow, rows);
        }

        for (auto &th : workers) {
            if (th.joinable()) th.join();
        }
        std::swap(src, dst);
    }
    return true;
}
