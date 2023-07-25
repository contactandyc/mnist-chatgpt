#include <stdlib.h>
#include "matrix.h"
#include "ac_allocator.h"

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

float* vectorSubtract(float* a, float* b, int size) {
    float* result = (float*)ac_malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

float** outerProduct(float* a, float* b, int a_len, int b_len) {
    float** result = createZeroMatrix(a_len, b_len);
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

float* matrixVectorMultiply(float** a, float* b, int a_rows, int a_cols) {
    float* result = (float*)ac_calloc(a_rows*sizeof(float));
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            result[i] += a[i][j] * b[j];
        }
    }
    return result;
}

float** transpose(float** a, int rows, int cols) {
    float** result = createZeroMatrix(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = a[i][j];
        }
    }
    return result;
}

float* elementwiseMultiply(float* a, float* b, int size) {
    float* result = (float*)ac_calloc(size*sizeof(float));
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
    return result;
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
    float* result = (float*)ac_malloc(size * sizeof(float));

    for (i = 0; i < size; i++) {
        result[i] = A[i] + B[i];
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
