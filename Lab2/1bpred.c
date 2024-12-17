#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void matrix_vector_column_split(int* mat, int* vec, int* result, int rows, int cols, int rank, int size) {
    int cols_per_proc = cols / size;
    int* local_mat = (int*)malloc(rows * cols_per_proc * sizeof(int));
    int* local_result = (int*)calloc(rows, sizeof(int));
    int* local_vec = (int*)malloc(cols_per_proc * sizeof(int));

    MPI_Scatter(mat, rows * cols_per_proc, MPI_INT, local_mat, rows * cols_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(vec, cols_per_proc, MPI_INT, local_vec, cols_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols_per_proc; j++) {
            local_result[i] += local_mat[i * cols_per_proc + j] * local_vec[j];
        }
    }

    MPI_Reduce(local_result, result, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local_mat);
    free(local_result);
    free(local_vec);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows, cols;
    int* mat = NULL;
    int* vec = NULL;
    int* result = NULL;

    if (rank == 0) {
        printf("Enter the number of rows and columns of the matrix: ");
        scanf("%d %d", &rows, &cols);

        if (cols % size != 0) {
            printf("The number of columns must be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        mat = (int*)malloc(rows * cols * sizeof(int));
        vec = (int*)malloc(cols * sizeof(int));
        result = (int*)malloc(rows * sizeof(int));

        printf("Enter the elements of the matrix row by row:\n");
        for (int i = 0; i < rows * cols; i++) {
            scanf("%d", &mat[i]);
        }

        printf("Enter the elements of the vector:\n");
        for (int i = 0; i < cols; i++) {
            scanf("%d", &vec[i]);
        }
    }

    matrix_vector_column_split(mat, vec, result, rows, cols, rank, size);

    if (rank == 0) {
        printf("Result: ");
        for (int i = 0; i < rows; i++) {
            printf("%d ", result[i]);
        }
        printf("\n");

        free(mat);
        free(vec);
        free(result);
    }

    MPI_Finalize();
    return 0;
}