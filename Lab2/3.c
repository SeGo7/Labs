#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOLERANCE 1e-6  // Заданная точность
#define MAX_ITER 10000  // Максимальное число итераций

// Функция источника тепла f(x, y)
double f(double x, double y) {
    return 0.0; // Задаётся пользователем, здесь пример с f(x, y) = 0
}

// Инициализация пластинки
void initialize_plate(double* plate, int nx, int ny, double boundary_value) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                plate[i * ny + j] = boundary_value; // Границы
            } else {
                plate[i * ny + j] = 0.0; // Внутренние точки
            }
        }
    }
}

// Печать области
void print_plate(double* plate, int nx, int ny) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%.2f ", plate[i * ny + j]);
        }
        printf("\n");
    }
}

// Метод Гаусса-Зейделя с волновой схемой
void gauss_seidel_wave(double* local_plate, int nx, int ny, int rank, int size, double boundary_value) {
    int local_nx = nx / size + 2; // С учётом соседних строк
    int local_ny = ny;

    double* new_plate = (double*)malloc(local_nx * local_ny * sizeof(double));
    double global_diff;

    int iter = 0;
    do {
        global_diff = 0.0;

        for (int i = 1; i < local_nx - 1; i++) {
            for (int j = 1; j < local_ny - 1; j++) {
                double old_value = local_plate[i * local_ny + j];
                new_plate[i * local_ny + j] = 0.25 * (
                    local_plate[(i - 1) * local_ny + j] +
                    local_plate[(i + 1) * local_ny + j] +
                    local_plate[i * local_ny + j - 1] +
                    local_plate[i * local_ny + j + 1] -
                    f((double)i / nx, (double)j / ny));
                global_diff += fabs(new_plate[i * local_ny + j] - old_value);
            }
        }

        // Обмен граничными данными между процессами
        if (rank > 0) {
            MPI_Send(new_plate + local_ny, local_ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(new_plate, local_ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(new_plate + (local_nx - 2) * local_ny, local_ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(new_plate + (local_nx - 1) * local_ny, local_ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Сравнение разницы на всех процессах
        double local_diff = global_diff;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Копирование нового состояния в старое
        for (int i = 1; i < local_nx - 1; i++) {
            for (int j = 1; j < local_ny; j++) {
                local_plate[i * local_ny + j] = new_plate[i * local_ny + j];
            }
        }

        iter++;
    } while (global_diff > TOLERANCE && iter < MAX_ITER);

    free(new_plate);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 100; // Размер сетки по x
    int ny = 100; // Размер сетки по y
    double boundary_value = 100.0; // Граничное значение температуры

    if (nx % size != 0) {
        if (rank == 0) {
            printf("The number of processes must divide the grid size evenly.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_nx = nx / size + 2; // +2 для соседних строк
    double* local_plate = (double*)malloc(local_nx * ny * sizeof(double));

    if (rank == 0) {
        double* global_plate = (double*)malloc(nx * ny * sizeof(double));
        initialize_plate(global_plate, nx, ny, boundary_value);

        // Раздача блоков остальным процессам
        for (int p = 1; p < size; p++) {
            MPI_Send(global_plate + p * (nx / size - 1) * ny, (nx / size + 2) * ny, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }

            // Скопировать свой блок
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < ny; j++) {
                local_plate[i * ny + j] = global_plate[i * ny + j];
            }
        }

        free(global_plate);
    } else {
        MPI_Recv(local_plate, local_nx * ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double start_time = MPI_Wtime();
    gauss_seidel_wave(local_plate, nx, ny, rank, size, boundary_value);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Elapsed time: %f seconds\n", end_time - start_time);
    }

    free(local_plate);
    MPI_Finalize();
    return 0;
}