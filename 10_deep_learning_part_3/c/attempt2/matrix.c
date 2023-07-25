#include "matrix.h"
#include "ac_allocator.h"
#include <stdlib.h>
#include <math.h>

float** createZeroMatrix(int rows, int cols) {
    int i, j;
    float** matrix = (float**)ac_malloc(rows * sizeof(float*));

    for (i = 0; i < rows; i++) {
        matrix[i] = (float*)ac_malloc(cols * sizeof(float));
        for (j = 0; j < cols; j++) {
            matrix[i][j] = 0.0f;
        }
    }
    return matrix;
}

void freeMatrix(float** matrix, int rows) {
    int i;
    for (i = 0; i < rows; i++) {
        ac_free(matrix[i]);
    }
    ac_free(matrix);
}

void vectorSubtractWithLearningRate(float* a, float* b, int size, float learning_rate) {
    for (int i = 0; i < size; i++) {
        a[i] -= learning_rate * b[i];
    }
}

float** outerProduct(float* a, float* b, float **result, int a_len, int b_len) {
    for (int i = 0; i < a_len; i++) {
        for (int j = 0; j < b_len; j++) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

void matrixSubtract(float** a, float** b, int rows, int cols, float learning_rate) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i][j] -= learning_rate * b[i][j];
        }
    }
}

float* matrixVectorMultiply(float** a, float* b, float *result, int a_rows, int a_cols) {
    for (int i = 0; i < a_rows; i++) {
        float r = 0.0;
        for (int j = 0; j < a_cols; j++) {
            r += a[i][j] * b[j];
        }
        result[i] = r;
    }
    return result;
}

float** transpose(float** a, float **result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = a[i][j];
        }
    }
    return result;
}

float* elementwiseMultiply(float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] * b[i];
    }
    return a;
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
    for (i = 0; i < size; i++) {
        A[i] = A[i] + B[i];
    }
    return A;
}

float dotProduct(float* A, float* B, int size) {
    int i;
    float result = 0.0;

    for (i = 0; i < size; i++) {
        result += A[i] * B[i];
    }
    return result;
}
