#include <stdio.h>
#include <omp.h>

int main() {
    int n = 10;
    int i;
    int a[10];

    // Указываем количество потоков
    omp_set_num_threads(4);

    // Параллельный цикл
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        a[i] = i * i;  // Каждый поток вычисляет часть итераций
        printf("Поток %d выполнил итерацию %d\n", omp_get_thread_num(), i);
    }

    // Вывод результата
    printf("Результат:\n");
    for (i = 0; i < n; i++) {
        printf("a[%d] = %d\n", i, a[i]);
    }

    return 0;
}