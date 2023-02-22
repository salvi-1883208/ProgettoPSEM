// dla algorithm implementation in cuda

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define blockSize 1024

// particle object
typedef struct {
    int x;
    int y;
    short int stuck = 0;
} particle;

void saveImage(int *grid, int size);

__device__ void moveParticle(particle *p, int direction);

// kernel to initialize the particles and the random states
__global__ void setup_kernel(particle *particles, int *grid, curandState *state, int gridSize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);

    // generate a random position for the particle
    int x = curand(&state[id]) % gridSize;
    int y = curand(&state[id]) % gridSize;

    // store the position in the particle object
    particles[id].x = x;
    particles[id].y = y;
}

// kernel to move the particles
__global__ void diffusion_kernel(particle *particles, int *grid, int gridSize, curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // copy the particle to a local variable
    particle p = particles[id];

    // if the particle is stuck do nothing
    if (p.stuck) {
        // printf("\nParticle %d stuck at (%d, %d)", id, p.x, p.y);
        particles[id].stuck = 2;
        return;
    }

    // generate a random direction
    moveParticle(&p, curand(&state[id]) % 8);

    // Check if the particle is outside the grid and if it is move it to the
    // opposite side
    if (p.x < 1)
        p.x = 1;
    else if (p.x > gridSize - 2)
        p.x = gridSize - 2;
    if (p.y < 1)
        p.y = 1;
    else if (p.y > gridSize - 2)
        p.y = gridSize - 2;

    // if (p.x == 0 || p.x == gridSize - 1 || p.y == 0 || p.y == gridSize - 1)
    //     printf("\nParticle %d at (%d, %d)", id, p.x, p.y);

    // printf("\nParticle %d at (%d, %d)", id, p.x, p.y);
    // grid[p.y * gridSize + p.x] = true;

    // if (((p.x == (gridSize / 2) - 1) || (p.x == (gridSize / 2) + 1)) && ((p.y == (gridSize / 2) - 1) || (p.y == (gridSize / 2) + 1)))
    //     printf("\nParticle %d at (%d, %d)", id, p.x, p.y);

    // check if the particle is close to a particle that is stuck
    if (grid[(p.y - 1) * gridSize + (p.x - 1)] ||  // top left
        grid[(p.y - 1) * gridSize + p.x] ||        // top
        grid[(p.y - 1) * gridSize + (p.x + 1)] ||  // top right
        grid[p.y * gridSize + (p.x - 1)] ||        // left
        grid[p.y * gridSize + (p.x + 1)] ||        // right
        grid[(p.y + 1) * gridSize + (p.x - 1)] ||  // bottom left
        grid[(p.y + 1) * gridSize + p.x] ||        // bottom
        grid[(p.y + 1) * gridSize + (p.x + 1)]) {  // bottom right

        // if the particle is close to a stuck particle mark it as stuck
        p.stuck = 1;
        // set the grid position to true using atomicAdd
        atomicAdd(&grid[p.y * gridSize + p.x], 1);

        // printf("\nParticle %d stuck at (%d, %d)", id, p.x, p.y);
    }

    // printf("\nA");

    // copy the particle back to the global memory
    particles[id] = p;
}

int main(int argc, char const *argv[]) {
    int gridSize = 1000;
    int numParticles = 12 * 1024;
    int maxIterations = 1000000;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    int stuckParticles = 0;
    int iterations = 0;

    // allocate memory for the grid
    int *h_grid = (int *)malloc(gridSize * gridSize * sizeof(int));
    for (int i = 0; i < gridSize * gridSize; i++)
        h_grid[i] = 0;

    // place the seed in the middle
    h_grid[gridSize / 2 * gridSize + gridSize / 2] = 1;

    // allocate memory for the particles
    particle *h_particles = (particle *)malloc(numParticles * sizeof(particle));

    // allocate the array of the random states in the device memory
    curandState *d_States;
    cudaMalloc((void **)&d_States, numParticles * sizeof(curandState));

    // allocate memory for the grid in the device memory
    int *d_grid;
    cudaMalloc((void **)&d_grid, gridSize * gridSize * sizeof(int));

    // allocate memory for the particles in the device memory
    particle *d_particles;
    cudaMalloc((void **)&d_particles, numParticles * sizeof(particle));

    // copy the grid to the device memory
    cudaMemcpy(d_grid, h_grid, gridSize * gridSize * sizeof(int), cudaMemcpyHostToDevice);

    // copy the particles to the device memory
    cudaMemcpy(d_particles, h_particles, numParticles * sizeof(particle), cudaMemcpyHostToDevice);

    // time execution start
    clock_t start = clock();

    // initialize the particles and the random states
    setup_kernel<<<numBlocks, blockSize>>>(d_particles, d_grid, d_States, gridSize);

    // wait for the initialization to finish
    cudaDeviceSynchronize();

    // TODO maybe remove the stuck particles from the array and launch a new
    // kernel with the remaining particles what happens if two or more particles
    // are stuck in the same position?
    do {
        // move the particles
        // diffusion_kernel<<<1, 100>>>(d_particles, d_grid, gridSize, d_States);
        diffusion_kernel<<<numBlocks, blockSize>>>(d_particles, d_grid, gridSize, d_States);

        // wait for the diffusion to finish
        cudaDeviceSynchronize();

        // copy the particles from the device memory to the host memory
        cudaMemcpy(h_particles, d_particles, numParticles * sizeof(particle), cudaMemcpyDeviceToHost);

        // count the number of stuck particles
        for (int i = 0; i < numParticles; i++)
            if (h_particles[i].stuck == 1) {
                stuckParticles++;
                // printf("\nParticle %d stuck at (%d, %d)", i, h_particles[i].x, h_particles[i].y);
            }

        iterations++;

        // printf("\nIteration %d", iterations);

        // move the particles that are not stuck or have not reached the maximum
        // number of iterations
    } while (stuckParticles < numParticles && iterations < maxIterations);

    // time execution end
    clock_t end = clock();

    // print the execution time
    printf("\nExecution time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // copy the grid from the device memory to the host memory
    cudaMemcpy(h_grid, d_grid, gridSize * gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // save the grid to a file
    saveImage(h_grid, gridSize);

    // free the memory
    free(h_grid);
    free(h_particles);
    cudaFree(d_grid);
    cudaFree(d_particles);
    cudaFree(d_States);

    return 0;
}

__device__ void moveParticle(particle *p, int direction) {
    switch (direction) {
        case 0:  // up
            p->y--;
            break;
        case 1:  // down
            p->y++;
            break;
        case 2:  // left
            p->x--;
            break;
        case 3:  // right
            p->x++;
            break;
        case 4:  // up left
            p->y--;
            p->x--;
            break;
        case 5:  // up right
            p->y--;
            p->x++;
            break;
        case 6:  // down left
            p->y++;
            p->x--;
            break;
        case 7:  // down right
            p->y++;
            p->x++;
            break;
    }
}

void saveImage(int *grid, int size) {
    // save image to .ppm file
    int count = 0;
    int i, j;
    FILE *fp = fopen("opt_out.ppm", "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[j * size + i]) {
                case 0:           // empty
                    color[0] = 0; /* red */
                    color[1] = 0; /* green */
                    color[2] = 0; /* blue */
                    break;
                case 1:             // stuck particles
                    color[0] = 255; /* red */
                    color[1] = 255; /* green */
                    color[2] = 255; /* blue */
                    count++;
                    break;
                default:            // stuck particles
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