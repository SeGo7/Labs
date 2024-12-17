#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Определение размеров сетки
#define NX 20  // Количество точек в направлении x
#define NY 20  // Количество точек в направлении y
#define MAX_ITER 10000  // Максимальное количество итераций
#define TOL 1e-6  // Точность сходимости

// Функция для инициализации сетки с граничными условиями
void initialize_grid(double u[NX+1][NY+1], double c) {
    // Установка граничных условий (Дирихле)
    for (int i = 0; i <= NX; i++) {
        u[i][0] = c;  // u(x, 0) = c
        u[i][NY] = c; // u(x, 1) = c
    }
    for (int j = 0; j <= NY; j++) {
        u[0][j] = c;  // u(0, y) = c
        u[NX][j] = c; // u(1, y) = c
    }

    // Инициализация внутренних точек (начальное приближение)
    for (int i = 1; i < NX; i++) {
        for (int j = 1; j < NY; j++) {
            u[i][j] = 0.0; // Начальное приближение
        }
    }
}

// Метод Гаусса-Зейделя для решения уравнения
void gauss_seidel(double u[NX+1][NY+1], double f[NX+1][NY+1]) {
    // Обновление значений сетки с использованием метода Гаусса-Зейделя
    #pragma omp parallel for collapse(2) shared(u, f)
    for (int i = 1; i < NX; i++) {
        for (int j = 1; j < NY; j++) {
            u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - f[i][j]);
        }
    }
}

int main() {
    double u[NX+1][NY+1]; // Сетка для решения
    double f[NX+1][NY+1]; // Источник (для примера, считаем его нулевым)

    // Инициализация сетки
    double c = 100.0; // Значение граничного условия
    initialize_grid(u, c);

    // Инициализация источника (если нужно)
    for (int i = 1; i < NX; i++) {
        for (int j = 1; j < NY; j++) {
            f[i][j] = 0.0; // Например, f(x, y) = 0
        }
    }

    // Итерации до сходимости или максимального числа итераций
    for (int iter = 0; iter < MAX_ITER; iter++) {
        double max_diff = 0.0;

        // Сохраняем старые значения для вычисления сходимости
        double old_u[NX+1][NY+1];
        for (int i = 0; i <= NX; i++) {
            for (int j = 0; j <= NY; j++) {
                old_u[i][j] = u[i][j];
            }
        }

        // Выполняем итерацию Гаусса-Зейделя
        gauss_seidel(u, f);

        // Проверка сходимости
        for (int i = 1; i < NX; i++) {
            for (int j = 1; j < NY; j++) {
                double diff = fabs(u[i][j] - old_u[i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        // Если изменения меньше порога, завершить итерации
        if (max_diff < TOL) {
            printf("Сходимость достигнута после %d итераций.\n", iter);
            break;
        }
    }

    // Вывод решения (можно распечатать или визуализировать)
    printf("Решение сетки:\n");
    for (int i = 0; i <= NX; i++) {
        for (int j = 0; j <= NY; j++) {
            printf("%f ", u[i][j]);
        }
        printf("\n");
    }

    return 0;
}
