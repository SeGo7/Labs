#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <complex.h>

#define MAX_ITER 1000  // Максимальное количество итераций для проверки принадлежности к множеству

// Глобальные переменные
int npoints;
int nthreads;
const double xmin = -2.5, xmax = 1.0;
double ran_x = xmax - xmin;
const double ymin = -1.5, ymax = 1.5;
double ran_y = ymax - ymin;
double step = 1000000000;
FILE *output_file;
pthread_mutex_t mutex;
long long count_find = 0;

// Функция для проверки, принадлежит ли точка множеству Мандельброта
int mandelbrot(double complex c) {
    double complex z = 0 + 0 * I;
    for (int n = 0; n < MAX_ITER; n++) {
        z = z * z + c;
        if (abs(z) >= 2.0) {
            return 0;  // Точка не принадлежит множеству Мандельброта
        }
    }
    return 1;  // Точка принадлежит множеству
}


void* compute_mandelbrot(void* threadid) {
    long tid = (long)threadid;

    double start_x = xmin + tid * ran_x/nthreads;
    double end_x = (tid == nthreads - 1) ? xmax : xmin + (tid + 1) * ran_x/nthreads;

    double start_y = ymin + tid * ran_y/nthreads;
    double end_y = (tid == nthreads - 1) ? ymax : ymin + (tid + 1) * ran_y/nthreads;


    double step_tread_x = ran_x/nthreads/step;
    double step_tread_y = ran_y/nthreads/step;


    for (double x_ = start_x; x_ < end_x; x_+=step_tread_x) {
        for (double y_ = start_y; y_ < end_y; y_+=step_tread_y) {

            double complex c = x_ + y_ * I;

            // Проверяем принадлежность к множеству Мандельброта
            int is_in_set = mandelbrot(c);



            pthread_mutex_lock(&mutex);

            if (count_find == npoints){
                pthread_mutex_unlock(&mutex);
                pthread_exit(NULL);
            }

            if (is_in_set) {
                count_find++;
                fprintf(output_file, "%.16lf,%.16lf\n", x_, y_);
                fflush(output_file);  // Принудительная запись в файл
            }
            pthread_mutex_unlock(&mutex);
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s nthreads npoints\n", argv[0]);
        return 1;
    }

    nthreads = atoi(argv[1]);
    npoints = atoi(argv[2]);

    // Открываем CSV файл для записи
    output_file = fopen("mandelbrot_set.csv", "w");
    if (output_file == NULL) {
        printf("Error: Unable to open output file.\n");
        return 1;
    }
    printf("File opened successfully.\n");

    // Инициализируем мьютекс
    pthread_mutex_init(&mutex, NULL);

    // Массив для хранения потоков
    pthread_t threads[nthreads];

    // Создаем потоки
    for (long t = 0; t < nthreads; t++) {
        int rc = pthread_create(&threads[t], NULL, compute_mandelbrot, (void*)t);
        if (rc) {
            printf("Error: unable to create thread, %d\n", rc);
            return 1;
        }
    }

    // Ждем завершения всех потоков
    for (long t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    // Закрываем файл
    fclose(output_file);
    printf("Mandelbrot set has been written to 'mandelbrot_set.csv'.\n");

    // Освобождаем ресурсы мьютекса
    pthread_mutex_destroy(&mutex);

    return 0;
}