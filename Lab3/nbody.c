#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define G 6.67E-11

void calculate_force(float *masses, float *array_x, float *array_y, float *fx, float *fy, int n)
{
    for (int i = 0; i < n; ++i)
    {
        fx[i] = 0.0;
        fy[i] = 0.0;
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            float dx = array_x[j] - array_x[i];
            float dy = array_y[j] - array_y[i];
            float squared_dist = dx * dx + dy * dy;
            float dist = sqrtf(squared_dist);
            float force = G * masses[i] * masses[j] / (squared_dist * dist);
            fx[i] += force * dx;
            fy[i] += force * dy;
            fx[j] -= force * dx;
            fy[j] -= force * dy;
        }
    }
}

void update_points(float *fx, float *fy, float *masses, float *array_x, float *array_y,
                   float *v_x, float *v_y, int n, float delta_t)
{
    for (int i = 0; i < n; ++i)
    {
        array_x[i] += v_x[i] * delta_t;
        array_y[i] += v_y[i] * delta_t;
        v_x[i] += (fx[i] / masses[i]) * delta_t;
        v_y[i] += (fy[i] / masses[i]) * delta_t;
    }
}

void generate_bodies(float *masses, float *array_x, float *array_y, float *v_x, float *v_y, int n)
{
    for (int i = 0; i < n; ++i)
    {
        masses[i] = ((float)rand()) / (RAND_MAX >> 10);
        array_x[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
        array_y[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
        v_x[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
        v_y[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
    }
}

int main(int argc, char *argv[])
{
    int n;
    float t_end;
    n = atoi(argv[1]);
    t_end = atof(argv[2]);
    // scanf("%d %f", &n, &t_end);
    float delta_t = t_end / 100.0;

    float *masses = malloc(n * sizeof(float));
    float *array_x = malloc(n * sizeof(float));
    float *array_y = malloc(n * sizeof(float));
    float *vs_x = malloc(n * sizeof(float));
    float *vs_y = malloc(n * sizeof(float));
    float *fx = malloc(n * sizeof(float));
    float *fy = malloc(n * sizeof(float));

    // for(int i = 0; i < n; ++i)
    // {
    //     scanf("%f %f %f %f %f", &masses[i], &array_x[i], &array_y[i], &vs_x[i], &vs_y[i]);
    // }
    generate_bodies(masses, array_x, array_y, vs_x, vs_y, n);

    float current_time = 0.0;
    while (current_time < t_end)
    {
        printf("%f ", current_time);
        for (int i = 0; i < n; ++i)
        {
            printf("%f %f ", array_x[i], array_y[i]);
        }
        printf("\n");
        calculate_force(masses, array_x, array_y, fx, fy, n);
        update_points(fx, fy, masses, array_x, array_y, vs_x, vs_y, n, delta_t);
        current_time += delta_t;
    }

    free(masses);
    free(array_x);
    free(array_y);
    free(fx);
    free(fy);
    return 0;
}