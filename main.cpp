#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int GRID_SIZE = 512; // Size of the grid
const int NUM_PARTICLES = 1000; // Number of particles

struct Particle {
    float x, y; // Position
    float vx, vy; // Velocity
};

// Function prototype for the CUDA kernel (defined in the .cu file)
void updateParticles(Particle* particles, int numParticles, float deltaTime);
void initializeParticles(Particle* particles, int numParticles);

int main() {
    srand(static_cast<unsigned int>(time(0)));

    // Allocate host memory
    std::vector<Particle> h_particles(NUM_PARTICLES);
    initializeParticles(h_particles.data(), NUM_PARTICLES);

    // Allocate device memory
    Particle* d_particles;
    cudaMalloc(&d_particles, sizeof(Particle) * NUM_PARTICLES);
    cudaMemcpy(d_particles, h_particles.data(), sizeof(Particle) * NUM_PARTICLES, cudaMemcpyHostToDevice);

    // Game loop
    float deltaTime = 0.016f; // Approx. 60 FPS
    for (int i = 0; i < 1000; ++i) {
        // Call the CUDA kernel defined in the .cu file
        updateParticles(d_particles, NUM_PARTICLES, deltaTime);
        cudaDeviceSynchronize();

        // Optionally: copy back the particles to host memory
        cudaMemcpy(h_particles.data(), d_particles, sizeof(Particle) * NUM_PARTICLES, cudaMemcpyDeviceToHost);

        // Here you would normally render the particles to the screen
        if (i % 100 == 0) {
            std::cout << "Particles positions at step " << i << ":\n";
            for (int j = 0; j < NUM_PARTICLES; ++j) {
                std::cout << "Particle " << j << ": (" << h_particles[j].x << ", " << h_particles[j].y << ")\n";
            }
        }
    }

    // Cleanup
    cudaFree(d_particles);
    return 0;
}
