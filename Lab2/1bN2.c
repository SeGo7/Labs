#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "./timer.h"

void PrintMatrix(int* mat, int rows, int cols, int rank) {
    printf("My rank: %d Matrix: \n", rank);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            printf("%d ", mat[i*cols + j]);
        } printf("\n");
    }
}

void PrintVector(int* vec, int rows, int rank) {
    printf("My rank: %d Vector: \n", rank);
    for (int j = 0; j < rows; j++){
            printf("%d ", vec[j]);
    } printf("\n");
}

void InputDim(int *n, int my_rank)
{
    if (my_rank == 0)
    {
        printf("Enter dimension: ");
        scanf("%d", n);
    }
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void InputMatrix(int *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = rand() % 10;
    }
}

void InputVector(int *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % 10;
    }
}

void DistributeMatrixColumns(int *matrix, int *local_mat, int rows, int cols, int comm_sz, int my_rank) {
    int cols_per_proc = cols / comm_sz;
    int remainder = cols % comm_sz;

    int *sendcounts = malloc(comm_sz * sizeof(int));
    int *displs = malloc(comm_sz * sizeof(int));

    int offset = 0;
    for (int i = 0; i < comm_sz; i++) {
        sendcounts[i] = rows * (cols_per_proc + (i < remainder ? 1 : 0));
        displs[i] = offset;
        offset += sendcounts[i];
    }

    MPI_Scatterv(matrix, sendcounts, displs, MPI_INT, local_mat, sendcounts[my_rank], MPI_INT, 0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);
}

void DistributeVectorColumns(int *vec, int *local_vec, int cols, int comm_sz, int my_rank) {
    int cols_per_proc = cols / comm_sz;
    int remainder = cols % comm_sz;

    int *sendcounts = malloc(comm_sz * sizeof(int));
    int *displs = malloc(comm_sz * sizeof(int));

    int offset = 0;
    for (int i = 0; i < comm_sz; i++) {
        sendcounts[i] = cols_per_proc + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    MPI_Scatterv(vec, sendcounts, displs, MPI_INT, local_vec, sendcounts[my_rank], MPI_INT, 0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);
}

void PartialMatVecMult(int *local_mat, int *local_vec, int *local_res, int rows, int local_cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < local_cols; j++) {
            local_res[i] += local_mat[i * local_cols + j] * local_vec[j];
        }
    }
}

void CollectPartialResults(int *local_res, int *final_res, int rows, int comm_sz, int my_rank) {
    int *recvcounts = malloc(comm_sz * sizeof(int));
    int *displs = malloc(comm_sz * sizeof(int));

    for (int i = 0; i < comm_sz; i++) {
        recvcounts[i] = rows;
        displs[i] = i * rows;
    }

    MPI_Reduce(local_res, final_res, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
}

void PrintResult(int *res, int rows, int my_rank) {
    if (my_rank == 0) {
        printf("Result vector:\n");
        for (int i = 0; i < rows; i++) {
            printf("%d ", res[i]);
        }
        printf("\n");
    }
}

int main()
{
    int my_rank, comm_sz;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int rows, cols;
    InputDim(&rows, my_rank);
    InputDim(&cols, my_rank);

    double start, finish;
    double full_start, full_finish;

    GET_TIME(full_start);

    int cols_per_proc = cols / comm_sz + (my_rank < cols % comm_sz ? 1 : 0);
    int *local_mat = malloc(rows * cols_per_proc * sizeof(int));
    int *local_vec = malloc(cols_per_proc * sizeof(int));
    int *local_res = calloc(rows, sizeof(int));
    int *final_res = NULL;

    if (my_rank == 0) {
        int *matrix = malloc(rows * cols * sizeof(int));
        int *vec = malloc(cols * sizeof(int));
        InputMatrix(matrix, rows, cols);
        InputVector(vec, cols);

        printf("Matrix:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d ", matrix[i * cols + j]);
            }
            printf("\n");
        }

        printf("Vector:\n");
        for (int i = 0; i < cols; i++) {
            printf("%d ", vec[i]);
        }
        printf("\n");

        DistributeMatrixColumns(matrix, local_mat, rows, cols, comm_sz, my_rank);
        DistributeVectorColumns(vec, local_vec, cols, comm_sz, my_rank);

        free(matrix);
        free(vec);
    } else {
        DistributeMatrixColumns(NULL, local_mat, rows, cols, comm_sz, my_rank);
        DistributeVectorColumns(NULL, local_vec, cols, comm_sz, my_rank);
    }

    GET_TIME(start);

    PartialMatVecMult(local_mat, local_vec, local_res, rows, cols_per_proc);

    if (my_rank == 0) {
        final_res = calloc(rows, sizeof(int));
    }

    CollectPartialResults(local_res, final_res, rows, comm_sz, my_rank);

    GET_TIME(finish);
    GET_TIME(full_finish);

    PrintResult(final_res, rows, my_rank);

    free(local_mat);
    free(local_vec);
    free(local_res);
    if (my_rank == 0) {
        free(final_res);
    }

    MPI_Finalize();
    return 0;
}
