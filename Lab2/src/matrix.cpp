#include "../include/matrix.h"
#include <stdexcept>
#include <iomanip>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("Negative matrix size");
    data.assign(rows, std::vector<double>(cols, 0.0));
}

int Matrix::getRows() const { return rows; }
int Matrix::getCols() const { return cols; }

double Matrix::getValue(int row, int col) const {
    if (row >= 0 && row < rows && col >= 0 && col < cols) return data[row][col];
    return 0.0;
}

void Matrix::setValue(int row, int col, double value) {
    if (row >= 0 && row < rows && col >= 0 && col < cols) data[row][col] = value;
}

void Matrix::print() const {
    std::cout << "Matrix " << rows << " x " << cols << ":\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << data[r][c] << "\t";
        }
        std::cout << "\n";
    }
}

void Matrix::fillFromConsole() {
    std::cout << "Input number of rows: ";
    std::cin >> rows;
    std::cout << "Input number of cols: ";
    std::cin >> cols;
    if (!std::cin || rows <= 0 || cols <= 0) {
        throw std::invalid_argument("rows and cols must be positive integers");
    }
    data.assign(rows, std::vector<double>(cols));
    std::cout << "Input matrix elements row by row (double):\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cin >> data[r][c];
        }
    }
}

double Matrix::applyConvolutionAt(int row, int col, const Matrix& kernel, const Matrix& source) const {
    int kRows = kernel.getRows();
    int kCols = kernel.getCols();
    if (kRows != kCols || kRows % 2 == 0) {
        throw std::invalid_argument("Kernel must be square with odd size");
    }
    int half = kRows / 2;
    double sum = 0.0;
    for (int kr = 0; kr < kRows; ++kr) {
        for (int kc = 0; kc < kCols; ++kc) {
            int srcR = row + (kr - half);
            int srcC = col + (kc - half);
            double srcVal = source.getValue(srcR, srcC);
            double kval = kernel.getValue(kr, kc);
            sum += kval * srcVal;
        }
    }
    return sum;
}
