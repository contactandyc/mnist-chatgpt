#ifndef _MATRIX_H
#define _MATRIX_H

float** createZeroMatrix(int rows, int cols);
void freeMatrix(float** matrix, int rows);
void matrixMultiply(float** mat1, float* mat2, float* result, int rows, int columns);
float* vectorAdd(float* A, float* B, int size);

float* vectorSubtract(float* a, float* b, int size);
float** outerProduct(float* a, float* b, int a_len, int b_len);
void matrixSubtract(float** a, float** b, int rows, int cols, float learning_rate);
float* matrixVectorMultiply(float** a, float* b, int a_rows, int a_cols);
float** transpose(float** a, int rows, int cols);
float* elementwiseMultiply(float* a, float* b, int size);

float dotProduct(float* A, float* B, int size);

#endif /* _MATRIX_H */