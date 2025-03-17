#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "main.cpp" // Include the header to use the Particle structure

__global__ void updateParticlesKernel(Particle* particles, int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        particles[idx].x += particles[idx].vx * deltaTime;
        particles[idx].y += particles[idx].vy * deltaTime;

        // Wrap around logic
        if (particles[idx].x < 0) particles[idx].x += GRID_SIZE;
        if (particles[idx].x >= GRID_SIZE) particles[idx].x -= GRID_SIZE;
        if (particles[idx].y < 0) particles[idx].y += GRID_SIZE;
        if (particles[idx].y >= GRID_SIZE) particles[idx].y -= GRID_SIZE;
    }
}

void updateParticles(Particle* particles, int numParticles, float deltaTime) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    updateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(particles, numParticles, deltaTime);
}
