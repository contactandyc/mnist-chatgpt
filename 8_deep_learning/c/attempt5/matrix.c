#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdlib.h>
#include "matrix.h"

float** createZeroMatrix(int rows, int cols) {
    int i, j;
    float** matrix = (float**)malloc(rows * sizeof(float*));

    for (i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
        for (j = 0; j < cols; j++) {
            matrix[i][j] = 0.0f;
        }
    }
    return matrix;
}

void freeMatrix(float** matrix, int rows) {
    int i;
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void matrixMultiply(float** mat1, float* mat2, float* result, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0;  // Initialize the result array
        for (int j = 0; j < columns; j++) {
            result[i] += mat1[i][j] * mat2[j];
        }
    }
}

float* vectorAdd(float* A, float* B, int size) {
    int i;
    float* result = (float*)malloc(size * sizeof(float));

    for (i = 0; i < size; i++) {
        result[i] = A[i] + B[i];
    }
    return result;
}

float* vectorSubtract(float* A, float* B, int size) {
    int i;
    float* result = (float*)malloc(size * sizeof(float));

    for (i = 0; i < size; i++) {
        result[i] = A[i] - B[i];
    }
    return result;
}

float** outerProduct(float* A, float* B, int sizeA, int sizeB) {
    int i, j;
    float** result = createZeroMatrix(sizeA, sizeB);

    for (i = 0; i < sizeA; i++) {
        for (j = 0; j < sizeB; j++) {
            result[i][j] = A[i] * B[j];
        }
    }
    return result;
}

float** matrixSubtract(float** A, float** B, int row, int col) {
    int i, j;
    float** result = createZeroMatrix(row, col);

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

float* matrixVectorMultiply(float** A, float* B, int rowA, int colA) {
    int i, j;
    float* result = (float*)calloc(rowA, sizeof(float));

    for (i = 0; i < rowA; i++) {
        for (j = 0; j < colA; j++) {
            result[i] += A[i][j] * B[j];
        }
    }
    return result;
}

float** transpose(float** A, int row, int col) {
    int i, j;
    float** result = createZeroMatrix(col, row);

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

float* elementwiseMultiply(float* A, float* B, int size) {
    int i;
    float* result = (float*)malloc(size * sizeof(float));

    for (i = 0; i < size; i++) {
        result[i] = A[i] * B[i];
    }
    return result;
}

float dotProduct(float* A, float* B, int size) {
    int i;
    float result = 0.0;

    for (i = 0; i < size; i++) {
        result += A[i] * B[i];
    }
    return result;
}

#endif /* _MATRIX_H */