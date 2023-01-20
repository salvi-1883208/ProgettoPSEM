#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define gridSize 400
#define numParticles 8192
#define blockSize 1024
#define numBlocks (numParticles + blockSize - 1) / blockSize
#define numSteps 10000
#define IMG_NAME "image.ppm"

// kernel to set up the seed for each thread
__global__ void setup_kernel(curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

void saveImage(int* grid, int size);

// GPU kernel to generate initial positions of particles
__global__ void generateParticlesKernel(int* particlePositions, int* grid, curandState* state);

// GPU kernel to update the positions of the particles during diffusion
__global__ void diffuseParticlesKernel(int* particlePositions, curandState* state);

// GPU kernel to check if a particle has come into contact with the aggregate
__global__ void checkContactKernel(int* particlePositions, int* grid);

int main(int argc, char const* argv[]) {
    // allocate memory for the particles positions
    int* particles = (int*)malloc(numParticles * 2 * sizeof(int));

    // allocate memory for the grid
    int* grid = (int*)malloc(gridSize * gridSize * sizeof(int));

    // allocate the grid for both the host and the device
    cudaMallocManaged((void**)&grid, gridSize * gridSize * sizeof(int));

    // allocate memory for the random states
    curandState* d_States;

    // allocate the array of the random states in the device memory
    cudaMalloc((void**)&d_States, numParticles * sizeof(curandState));

    // allocate the particles for both the host and the device
    cudaMallocManaged((void**)&particles, numParticles * 2 * sizeof(int));

    // launch the kernel to set up the seed for each thread
    setup_kernel<<<numBlocks, blockSize>>>(d_States);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // time execution start
    clock_t start = clock();

    // launch the kernel to generate the initial positions of the particles
    generateParticlesKernel<<<numBlocks, blockSize>>>(particles, grid, d_States);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // launch the kernel to diffuse the particles
    for (int i = 0; i < numSteps; i++) {
        diffuseParticlesKernel<<<numBlocks, blockSize>>>(particles, d_States);
        checkContactKernel<<<numBlocks, blockSize>>>(particles, grid);
    }

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // time execution end
    clock_t end = clock();

    // print the execution time
    printf("Execution time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // save the image
    saveImage(grid, gridSize);

    // free the memory
    cudaFree(particles);
    cudaFree(grid);
    cudaFree(d_States);

    return 0;
}

// GPU kernel to generate initial positions of particles
__global__ void generateParticlesKernel(int* particlePositions, int* grid, curandState* state) {
    // Generate a unique index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the grid and place the seed point
    if (index < numParticles) {
        // Get the current position of the particle
        int x = (int)particlePositions[2 * index];
        int y = (int)particlePositions[2 * index + 1];

        // Place the seed point randomly on the grid
        if (index == 0) {
            x = gridSize / 2;
            y = gridSize / 2;
            grid[x + y * gridSize] = 1;
        }
        // Place the rest of the particles randomly on the grid
        else {
            x = curand(&state[index]) % gridSize;
            y = curand(&state[index]) % gridSize;
        }

        // Store the initial position in the particlePositions buffer
        particlePositions[2 * index] = x;
        particlePositions[2 * index + 1] = y;
    }
}

__device__ void moveParticle(int* x, int* y, int m);

// GPU kernel to update the positions of the particles during diffusion
__global__ void diffuseParticlesKernel(int* particlePositions, curandState* state) {
    // Generate a unique index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Update the position of the particle using the diffusion algorithm
    if (index < numParticles) {
        // Get the current position of the particle
        int x = particlePositions[2 * index];
        int y = particlePositions[2 * index + 1];

        if (x == -1 && y == -1)
            return;

        // Update the position using random walk
        moveParticle(&x, &y, curand(&state[index]) % 8);

        // Check if the particle is outside the grid
        if (x < 0)
            x = 0;
        if (x >= gridSize)
            x = gridSize - 1;
        if (y < 0)
            y = 0;
        if (y >= gridSize)
            y = gridSize - 1;

        // Store the new position in the particlePositions buffer
        particlePositions[2 * index] = x;
        particlePositions[2 * index + 1] = y;
    }
}

// GPU kernel to check if a particle has come into contact with the aggregate
__global__ void checkContactKernel(int* particlePositions, int* grid) {
    // Generate a unique index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the particle is in contact with an occupied cell
    if (index < numParticles) {
        // Get the current position of the particle
        int x = particlePositions[2 * index];
        int y = particlePositions[2 * index + 1];

        if (x == -1 && y == -1)
            return;

        // Check if the current cell is occupied (i think this is useless)
        if (grid[x + y * gridSize] == 2) {
            // Mark the particle as stuck
            particlePositions[2 * index] = -1;
            particlePositions[2 * index + 1] = -1;
        }
        // Check the 8 neighboring cells
        else {
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
                        if (grid[nx + ny * gridSize] == 1) {
                            // Mark the particle as stuck
                            particlePositions[2 * index] = -1;
                            particlePositions[2 * index + 1] = -1;
                            grid[x + y * gridSize] = 1;
                            break;
                        }
                    }
                }
            }
        }
    }
}

// function to update the position of a particle
__device__ void moveParticle(int* x, int* y, int m) {
    switch (m) {
        case 0:  // top left
            (*x)--;
            (*y)--;
            break;
        case 1:  // top
            (*y)--;
            break;
        case 2:  // top right
            (*x)++;
            (*y)--;
            break;
        case 3:  // right
            (*x)++;
            break;
        case 4:  // bottom right
            (*x)++;
            (*y)++;
            break;
        case 5:  // bottom
            (*y)++;
            break;
        case 6:  // bottom left
            (*x)--;
            (*y)++;
            break;
        case 7:  // left
            (*x)--;
            break;
    }
}

void saveImage(int* grid, int size) {
    // save image to .ppm file
    int count = 0;
    int i, j;
    FILE* fp = fopen(IMG_NAME, "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[j * gridSize + i]) {
                case 1:             // stuck particles
                    color[0] = 255; /* red */
                    color[1] = 255; /* green */
                    color[2] = 255; /* blue */
                    count++;
                    break;
                // case 2:             // seed
                //     color[0] = 255; /* red */
                //     color[1] = 0;   /* green */
                //     color[2] = 0;   /* blue */
                //     break;
                case 0:           // empty spots
                    color[0] = 0; /* red */
                    color[1] = 0; /* green */
                    color[2] = 0; /* blue */
                    break;
                default:            // overlapping
                    color[0] = 255; /* red */
                    color[1] = 0;   /* green */
                    color[2] = 0;   /* blue */
                    count++;
                    break;
            }
            (void)fwrite(color, 1, 3, fp);
        }
    }
    (void)fclose(fp);

    printf("Saved image containing %d particles\n", count - 1);

    return;
}