#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

// Глобальные переменные для хранения результатов
long long total_hits = 0; // количество попаданий в окружность
long long ntrials;        // общее количество попыток
int nthreads;             // количество потоков

// Мьютекс для синхронизации доступа к общей переменной total_hits
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Функция, выполняемая каждым потоком
void* monte_carlo_pi(void* threadid) {
    long long hits = 0;  // Локальная переменная для хранения количества попаданий в окружность
    unsigned int seed = (unsigned int)time(NULL) + (unsigned int)threadid; // Уникальное зерно для генератора случайных чисел

    // Определим количество бросков для данного потока
    long long local_trials = ntrials / nthreads;

    for (long long i = 0; i < local_trials; i++) {
        // Генерация случайных координат (x, y) в пределах квадрата [-1, 1]
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;

        // Проверка, попала ли точка в окружность с радиусом 1
        if (x * x + y * y <= 1.0) {
            hits++; // если попала, увеличиваем количество попаданий
        }
    }

    // Блокировка мьютекса для синхронизации доступа к общей переменной
    pthread_mutex_lock(&mutex);
    total_hits += hits; // добавляем результат к общему количеству попаданий
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s nthreads ntrials\n", argv[0]);
        return -1;
    }

    // Получаем количество потоков и общее количество попыток из аргументов командной строки
    nthreads = atoi(argv[1]);
    ntrials = atoll(argv[2]);

    // Массив потоков
    pthread_t threads[nthreads];

    // Создаем потоки
    for (long t = 0; t < nthreads; t++) {
        int rc = pthread_create(&threads[t], NULL, monte_carlo_pi, (void*)t);
        if (rc) {
            printf("Error: unable to create thread, %d\n", rc);
            return -1;
        }
    }

    // Ожидаем завершения всех потоков
    for (long t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    // Вычисляем значение π на основе количества попаданий
    double pi_estimate = 4.0 * (double)total_hits / (double)ntrials;

    // Выводим результат
    printf("Estimated Pi = %.6f\n", pi_estimate);

    // Завершаем программу
    pthread_exit(NULL);
}