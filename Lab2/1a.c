#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void PrintMatrix(int* mat, int rows, int cols, int rank) {
    printf("My rank: %d Matrix: \n", rank);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            printf("%d ", mat[i*cols + j]);
        } printf("\n");
    }
}

void PrintVector2(int* vec, int rows, int rank) {
    printf("My rank: %d Vector: \n", rank);
    for (int j = 0; j < rows; j++){
            printf("%d ", vec[j]);
    } printf("\n");
}

void InputDim(int *n, int my_rank) {
    if (my_rank == 0) {
        scanf("%d", n);
    }
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void InputVector(int *v, int n, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            v[i] = rand() % 10;
        }
    }
    MPI_Bcast(v, n, MPI_INT, 0, MPI_COMM_WORLD);
}

void PrintDistributedVector(int *v, int n, int my_rank, int *sizes, int *displs) {
    int *temp = calloc(n, sizeof(int));
    MPI_Gatherv(v, sizes[my_rank], MPI_INT, temp, sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            printf("%d ", temp[i]);
        }
        printf("\n");
    }
    free(temp);
}

void InputMatrix(int r, int c, int *mat, int my_rank, int *sizes, int *displs) {
    if (my_rank == 0) {
        int *temp = calloc(r * c, sizeof(int));
        for (int i = 0; i < r * c; i++) {
            temp[i] = rand() % 10;
        }
        MPI_Scatterv(temp, sizes, displs, MPI_INT, mat, sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
        free(temp);
    } else {
        MPI_Scatterv(NULL, sizes, displs, MPI_INT, mat, sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void PrintDistributedMatrix(int r, int c, int *local_mat, int my_rank, int *sizes, int *displs) {
    if (my_rank == 0) {
        int *temp = calloc(r * c, sizeof(int));
        MPI_Gatherv(local_mat, sizes[my_rank], MPI_INT, temp, sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);
        printf("Matrix: \n");
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                printf("%d ", temp[i * c + j]);
            }
            printf("\n");
        }
        free(temp);
    } else {
        MPI_Gatherv(local_mat, sizes[my_rank], MPI_INT, NULL, sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void PrintVector(int *v, int n, int my_rank) {
    if (my_rank == 0) {
        printf("Vector: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", v[i]);
        }
        printf("\n");
    }
}

void MatVecMult(int *mat, int *vec, int *res, int local_r, int c, int my_rank) {
    // PrintMatrix(mat, local_r, c, my_rank);
    // PrintVector2(vec, c, my_rank);
    for (int i = 0; i < local_r; i++) {
        res[i] = 0;
        for (int j = 0; j < c; j++) {
            res[i] += mat[i * c + j] * vec[j];
        }
    }
}

void BuildSize(int r, int c, int comm_sz, int *sizes) {
    for (int i = 0; i < comm_sz; i++) {
        sizes[i] = r / comm_sz;
        if (i < r % comm_sz) {
            sizes[i]++;
        }
        sizes[i] *= c;
    }
}

void BuildDisplacements(int comm_sz, int *displs, int *sizes) {
    displs[0] = 0;
    for (int i = 1; i < comm_sz; i++) {
        displs[i] = displs[i - 1] + sizes[i - 1];
    }
}

int main() {
    int comm_sz, my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double start_time = MPI_Wtime();

    int r, c;
    InputDim(&r, my_rank);
    InputDim(&c, my_rank);

    int *sizes_mat = calloc(comm_sz, sizeof(int));
    int *displacements_mat = calloc(comm_sz, sizeof(int));
    int *sizes_vec = calloc(comm_sz, sizeof(int));
    int *displacements_vec = calloc(comm_sz, sizeof(int));

    BuildSize(r, c, comm_sz, sizes_mat);
    BuildDisplacements(comm_sz, displacements_mat, sizes_mat);
    BuildSize(r, 1, comm_sz, sizes_vec);
    BuildDisplacements(comm_sz, displacements_vec, sizes_vec);

    int *mat = calloc(sizes_mat[my_rank], sizeof(int));
    int *vec = calloc(c, sizeof(int));
    InputVector(vec, c, my_rank);
    InputMatrix(r, c, mat, my_rank, sizes_mat, displacements_mat);

    PrintVector(vec, c, my_rank);
    PrintDistributedMatrix(r, c, mat, my_rank, sizes_mat, displacements_mat);

    int *res = calloc(r / comm_sz + 1, sizeof(int));
    MatVecMult(mat, vec, res, sizes_mat[my_rank] / c, c, my_rank);

    PrintDistributedVector(res, r, my_rank, sizes_vec, displacements_vec);
    
    free(mat);
    free(vec);
    free(res);
    free(sizes_mat);
    free(displacements_mat);
    free(sizes_vec);
    free(displacements_vec);

    if (my_rank == 0) {
        double end_time = MPI_Wtime();
        printf("Elapsed time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}