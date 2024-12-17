#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void InputDim(int *n, int my_rank) {
    if (my_rank == 0) {
        scanf("%d", n);
    }
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_vector_column_split(int** mat, int* vec, int* result, int rows, int cols, int rank, int size) {
    int cols_per_proc = cols / size;
    int* local_mat = (int*)malloc(cols_per_proc * sizeof(int));
    int* local_result = (int*)calloc(rows, sizeof(int));
    int* local_vec = (int*)malloc(cols_per_proc * sizeof(int));

    MPI_Scatter(vec, cols_per_proc, MPI_INT, local_vec, cols_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows; i++) {
        if (rank == 0) {
            MPI_Scatter(mat[i], cols_per_proc, MPI_INT, local_mat, cols_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Scatter(NULL, cols_per_proc, MPI_INT, local_mat, cols_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
        }

        for (int j = 0; j < cols_per_proc; j++) {
            local_result[i] += local_mat[j] * local_vec[j];
        }
    }


    MPI_Reduce(local_result, result, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local_mat);
    free(local_result);
    free(local_vec);
}

int main() {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int rows, cols;
    InputDim(&rows, rank);
    InputDim(&cols, rank);

    int** mat = NULL;
    int* vec = NULL;
    int* result = NULL;

    if (rank == 0) {
        if (cols % size != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        mat = (int **)malloc(rows * sizeof(int *));
        for (int i = 0; i < rows; i++) {
            mat[i] = (int *)malloc(cols * sizeof(int));
            if (mat[i] == NULL) {
                printf("Memory allocation failed for row %d\n", i);
                return 1;
            }
        }
        vec = (int*)malloc(cols * sizeof(int));
        result = (int*)malloc(rows * sizeof(int));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = rand() % 10;
            }
        }

        for (int i = 0; i < cols; i++) {
            vec[i] = rand() % 10;
        }

    }

    matrix_vector_column_split(mat, vec, result, rows, cols, rank, size);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        printf("Elapsed time: %f seconds\n", end_time - start_time);

        // for (int i = 0; i < rows; i++) {
        //     for (int j = 0; j < cols; j++) 
        //     {
        //         printf("%d ", mat[i][j]);
        //     }
        //     printf("\n");
        // }

        // printf("Vector: ");
        // for (int i = 0; i < cols; i++) {
        //     printf("%d ", vec[i]);
        // } printf("\n");

        // printf("Result: ");
        // for (int i = 0; i < rows; i++) {
        //     printf("%d ", result[i]);
        // }
        // printf("\n");

        free(mat);
        free(vec);
        free(result);
    }
    MPI_Finalize();
    return 0;
}