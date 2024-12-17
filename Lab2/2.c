#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

void initialize_matrix(int* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = rand() % 10; // Случайные числа от 0 до 9
    }
}

void print_matrix(int* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void cannon_algorithm(int* A, int* B, int* C, int n, int sqrt_p, int rank) {
    int block_size = n / sqrt_p; // Размер блока
    int* local_A = (int*)malloc(block_size * block_size * sizeof(int));
    int* local_B = (int*)malloc(block_size * block_size * sizeof(int));
    int* local_C = (int*)calloc(block_size * block_size, sizeof(int));

    MPI_Comm grid_comm, row_comm, col_comm;

    // Создаём декартовую топологию процессов
    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {1, 1}; // Замкнутая топология (циклический сдвиг)
    int coords[2];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    // Создаём подкоммуникаторы для строк и столбцов
    MPI_Comm_split(grid_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(grid_comm, coords[1], coords[0], &col_comm);




    for (int i = 0; i < sqrt_p; i++) {
        for (int j = 0; j < sqrt_p; j++) {
            int target_rank;
            MPI_Cart_rank(grid_comm, (int[]){i, j}, &target_rank);
            // printf("i: %d j: %d My rank: %d Target: %d\n", i, j, rank, target_rank);
            if (target_rank == rank) {
                for (int bi = 0; bi < block_size; bi++) {
                    for (int bj = 0; bj < block_size; bj++) {
                        local_A[bi * block_size + bj] = A[(i * block_size + bi) * n + j * block_size + bj];
                        local_B[bi * block_size + bj] = B[(i * block_size + bi) * n + j * block_size + bj];
                    }
                }
                break;
            }
        }
    }

    // Выполняем начальные циклические сдвиги
    MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_INT, (coords[0] - coords[1] + sqrt_p) % sqrt_p, 0,
                         (coords[0] + coords[1]) % sqrt_p, 0, row_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_INT, (coords[1] - coords[0] + sqrt_p) % sqrt_p, 0,
                         (coords[1] + coords[0]) % sqrt_p, 0, col_comm, MPI_STATUS_IGNORE);

    // Основной цикл алгоритма Кэннона
    for (int step = 0; step < sqrt_p; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    local_C[i * block_size + j] += local_A[i * block_size + k] * local_B[k * block_size + j];
                }
            }
        }

        // Циклический сдвиг блоков
        MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_INT, (coords[1] + 1) % sqrt_p, 0,
                             (coords[1] - 1 + sqrt_p) % sqrt_p, 0, row_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_INT, (coords[0] + 1) % sqrt_p, 0,
                             (coords[0] - 1 + sqrt_p) % sqrt_p, 0, col_comm, MPI_STATUS_IGNORE);
    }


    if (rank == 0) {
        for (int i = 0; i < sqrt_p; i++) {
            for (int j = 0; j < sqrt_p; j++) {
                int source_rank;
                MPI_Cart_rank(grid_comm, (int[]){i, j}, &source_rank);
                if (source_rank != 0) {
                    MPI_Recv(local_C, block_size * block_size, MPI_INT, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                for (int bi = 0; bi < block_size; bi++) {
                    for (int bj = 0; bj < block_size; bj++) {
                        C[(i * block_size + bi) * n + j * block_size + bj] = local_C[bi * block_size + bj];
                    }
                }
            }
        }
    } else {
        MPI_Send(local_C, block_size * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sqrt_p = (int)sqrt(size);
    if (sqrt_p * sqrt_p != size) {
        if (rank == 0) {
            printf("The number of processes must be a perfect square.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 4; // Размерность матриц (например, 4x4)
    int* A = (int*)malloc(n * n * sizeof(int));
    int* B = (int*)malloc(n * n * sizeof(int));
    int* C = NULL;

    if (rank == 0) {
        C = (int*)calloc(n * n, sizeof(int));

        initialize_matrix(A, n, n);
        initialize_matrix(B, n, n);

        printf("Matrix A:\n");
        print_matrix(A, n, n);

        printf("Matrix B:\n");
        print_matrix(B, n, n);
    }

    MPI_Bcast(A, n*n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n*n, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    cannon_algorithm(A, B, C, n, sqrt_p, rank);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Matrix C (Result):\n");
        print_matrix(C, n, n);

        printf("Elapsed time: %f seconds\n", end_time - start_time);

        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}