#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define G 6.67E-11

__global__ void calculate_force_cuda(float *masses, float *array_x, float *array_y, 
                                     float *fx, float *fy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float local_fx = 0.0f;
        float local_fy = 0.0f;

        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float dx = array_x[j] - array_x[i];
                float dy = array_y[j] - array_y[i];
                float squared_dist = dx * dx + dy * dy + 1e-9f; // Avoid division by zero
                float dist = sqrtf(squared_dist);
                float force = G * masses[i] * masses[j] / (squared_dist * dist);
                local_fx += force * dx;
                local_fy += force * dy;
            }
        }

        fx[i] = local_fx;
        fy[i] = local_fy;
    }
}

// CUDA kernel for updating positions and velocities of bodies
__global__ void update_points_cuda(float *fx, float *fy, float *masses, float *array_x, 
                                    float *array_y, float *v_x, float *v_y, int n, float delta_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        v_x[i] += (fx[i] / masses[i]) * delta_t;
        v_y[i] += (fy[i] / masses[i]) * delta_t;
        array_x[i] += v_x[i] * delta_t;
        array_y[i] += v_y[i] * delta_t;
    }
}

// Host function for generating initial body data
void generate_bodies(float *masses, float *array_x, float *array_y, float *v_x, float *v_y, int n) {
    for (int i = 0; i < n; ++i) {
        masses[i] = ((float)rand()) / (RAND_MAX >> 10);
        array_x[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
        array_y[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
        v_x[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
        v_y[i] = 2.0 * ((float)rand()) / RAND_MAX - 1.0;
    }
}

int main(int argc, char *argv[]) {
    int n;
    float t_end;
    n = atoi(argv[1]);
    t_end = atof(argv[2]);
    float delta_t = t_end / 100.0;

    // Host memory allocation
    float *masses = (float *)malloc(n * sizeof(float));
    float *array_x = (float *)malloc(n * sizeof(float));
    float *array_y = (float *)malloc(n * sizeof(float));
    float *v_x = (float *)malloc(n * sizeof(float));
    float *v_y = (float *)malloc(n * sizeof(float));
    float *fx = (float *)malloc(n * sizeof(float));
    float *fy = (float *)malloc(n * sizeof(float));

    generate_bodies(masses, array_x, array_y, v_x, v_y, n);

    // Device memory allocation
    float *d_masses, *d_array_x, *d_array_y, *d_v_x, *d_v_y, *d_fx, *d_fy;
    cudaMalloc((void **)&d_masses, n * sizeof(float));
    cudaMalloc((void **)&d_array_x, n * sizeof(float));
    cudaMalloc((void **)&d_array_y, n * sizeof(float));
    cudaMalloc((void **)&d_v_x, n * sizeof(float));
    cudaMalloc((void **)&d_v_y, n * sizeof(float));
    cudaMalloc((void **)&d_fx, n * sizeof(float));
    cudaMalloc((void **)&d_fy, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_masses, masses, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_x, array_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_y, array_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_x, v_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_y, v_y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Determine CUDA grid and block sizes
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    float current_time = 0.0;
    while (current_time < t_end) {
        // Print current state
        printf("%f ", current_time);
        for (int i = 0; i < n; ++i) {
            printf("%f %f ", array_x[i], array_y[i]);
        }
        printf("\n");

        // Calculate forces on the device
        calculate_force_cuda<<<blocks_per_grid, threads_per_block>>>(d_masses, d_array_x, d_array_y, d_fx, d_fy, n);

        // Update positions and velocities on the device
        update_points_cuda<<<blocks_per_grid, threads_per_block>>>(d_fx, d_fy, d_masses, d_array_x, d_array_y, d_v_x, d_v_y, n, delta_t);

        // Copy updated positions back to host
        cudaMemcpy(array_x, d_array_x, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(array_y, d_array_y, n * sizeof(float), cudaMemcpyDeviceToHost);

        current_time += delta_t;
    }

    // Free device memory
    cudaFree(d_masses);
    cudaFree(d_array_x);
    cudaFree(d_array_y);
    cudaFree(d_v_x);
    cudaFree(d_v_y);
    cudaFree(d_fx);
    cudaFree(d_fy);

    // Free host memory
    free(masses);
    free(array_x);
    free(array_y);
    free(v_x);
    free(v_y);
    free(fx);
    free(fy);

    return 0;
}
