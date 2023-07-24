#ifndef _MATRIX_H
#define _MATRIX_H

float** createZeroMatrix(int rows, int cols);
void freeMatrix(float** matrix, int rows);
void matrixMultiply(float** mat1, float* mat2, float* result, int rows, int columns);
float* vectorAdd(float* A, float* B, int size);
float* vectorSubtract(float* A, float* B, int size);
float** outerProduct(float* A, float* B, int sizeA, int sizeB);
float** matrixSubtract(float** A, float** B, int row, int col);
float* matrixVectorMultiply(float** A, float* B, int rowA, int colA);
float** transpose(float** A, int row, int col);
float* elementwiseMultiply(float* A, float* B, int size);
float dotProduct(float* A, float* B, int size);

#endif /* _MATRIX_H */