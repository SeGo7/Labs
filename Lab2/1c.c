#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void PrintMatrix(int* mat, int rows, int cols, int rank) {
    printf("My rank: %d Matrix: \n", rank);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            printf("%d ", mat[i*cols + j]);
        } printf("\n");
    }
}

void PrintVector(int* vec, int rows, int rank) {
    printf("My rank: %d Vector: ", rank);
    for (int j = 0; j < rows; j++){
            printf("%d ", vec[j]);
    } printf("\n");
}

void BuildSize(int len, int sq_size, int *sizes) { 
    for (int i = 0; i < sq_size; i++) {
        sizes[i] = len / sq_size;
        if (i < len % sq_size) {
            sizes[i]++;
        }
    }
}

void BuildDisplacements(int sq_size, int *displs, int *sizes) {
    displs[0] = 0;
    for (int i = 1; i < sq_size; i++) {
        displs[i] = displs[i - 1] + sizes[i - 1];
    }
}

void InputDim(int *n, int my_rank) {
    if (my_rank == 0) {
        scanf("%d", n);
    }
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_vector_block_split(int* mat, int* vec, int* result, int rows, int cols, int* dip_cols, int* dip_rows, int my_rank, int size) {
    int* local_result = (int*)calloc(rows, sizeof(int));

    
    int sq_size = (int)sqrt(size);
    int block_r = (int)(my_rank / (sq_size) );
    int block_c = (int)(my_rank % (sq_size) );
    
    for (int i = dip_rows[block_r]; i < (block_r == sq_size-1 ? rows : dip_rows[block_r+1]); i++) {
        for (int j = dip_cols[block_c]; j < (block_c == sq_size-1 ? cols : dip_cols[block_c+1]); j++) {
            local_result[i] += mat[i*cols+j] * vec[j];
            if (my_rank==1){
                //printf("My rank: %d x  %d %d\n", my_rank, mat[i*cols+j], vec[j]);
            }
        }
    }
    MPI_Reduce(local_result, result, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local_result);
}

int main() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int sq_size = (int)sqrt(size);
    if (sq_size * sq_size != size) {
        if (rank == 0) {
            printf("The number of processes must be a perfect square.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows, cols;
    InputDim(&rows, rank);
    InputDim(&cols, rank);

    int* mat = (int*)malloc(rows * cols * sizeof(int));
    int* vec = (int*)malloc(cols * sizeof(int));
    int* result = NULL;


    int *block_rows = calloc(sq_size, sizeof(int));
    int *dip_rows = calloc(sq_size, sizeof(int));
    int *block_cols = calloc(sq_size, sizeof(int));
    int *dip_cols = calloc(sq_size, sizeof(int));

    BuildSize(rows, sq_size, block_rows);
    BuildDisplacements(sq_size, dip_rows, block_rows);
    BuildSize(cols, sq_size, block_cols);
    BuildDisplacements(sq_size, dip_cols, block_cols);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        result = (int*)malloc(rows * sizeof(int));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i*cols + j] = rand() % 10;
            }
        }

        for (int i = 0; i < cols; i++) {
            vec[i] = rand() % 10;
        }
    }
    
    MPI_Bcast(mat, rows*cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vec, cols, MPI_INT, 0, MPI_COMM_WORLD);
    matrix_vector_block_split(mat, vec, result, rows, cols, dip_rows, dip_cols, rank, size);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        printf("Elapsed time: %f seconds\n", end_time - start_time);
        // printf("Matrix: \n");
        // for (int i = 0; i < rows; i++) {
        //     for (int j = 0; j < cols; j++) 
        //     {
        //         printf("%d ", mat[i*cols+j]);
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