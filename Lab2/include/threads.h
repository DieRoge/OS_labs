#ifndef THREADS_H
#define THREADS_H

#include <vector>
#include <thread>
#include <atomic>

#include "matrix.h"

class ThreadControl {
public:
    ThreadControl(int maxThreads);
    ~ThreadControl();

    bool applyConvolution(Matrix& bufferA, Matrix& bufferB, const Matrix& kernel, int iterations);

private:
    int maxThreads;
    std::vector<std::thread> workers;

    void workerFunc(const Matrix* src, Matrix* dst, const Matrix* kernel,
                    std::atomic<int>* nextRow, int totalRows);
};

#endif
