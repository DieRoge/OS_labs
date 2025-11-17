#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);

    int getRows() const;
    int getCols() const;

    double getValue(int row, int col) const;
    void setValue(int row, int col, double value);

    void print() const;
    void fillFromConsole();

    double applyConvolutionAt(int row, int col, const Matrix& kernel, const Matrix& source) const;

private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;
};

#endif
