// cuda implementation of the dla algorithm in 3 dimensions

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IMG_NAME "out.ppm"

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int* z, int m);

// check if the particle is close to an already stuck particle
__device__ int is_close_to_stuck(int* grid, int x, int y, int z, int gridSize);

// calculate the number of overlapping particles
int calc_over(int* grid, int size);

// write the matrix to a file
void write_matrix_to_file(int* matrix, int dim);

// kernel to set up the seed for each thread
__global__ void setup_kernel(curandState* state, int randomSeed) {
    // calculate thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(randomSeed, id, 0, &state[id]);
}

// kernel to perform the dla algorithm
__global__ void dla_kernel(int* grid, int* skipped, curandState* state, int gridSize, int maxIterations) {
    // calculate thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // copy the random state to the local memory
    curandState localState = state[id];

    // initialize the starting position of the particle
    int x;
    int y;
    int z;

    // if the particle has been generated on a stuck particle, generate a new position
    do {
        x = curand(&localState) % gridSize;
        y = curand(&localState) % gridSize;
        z = curand(&localState) % gridSize;
    } while (grid[z * gridSize * gridSize + y * gridSize + x]);

    // initialize the counter for the number of iterations
    int g = 0;

    // iterate until the particle is attached to the grid or it did more than maxIterations number of iterations
    while ((g <= maxIterations)) {
        // if the particle is outside the grid, move it back inside
        if (x < 1)
            x = 1;
        else if (x > gridSize - 2)
            x = gridSize - 2;
        if (y < 1)
            y = 1;
        else if (y > gridSize - 2)
            y = gridSize - 2;
        if (z < 1)
            z = 1;
        else if (z > gridSize - 2)
            z = gridSize - 2;

        // if the particle is close to an already stuck particle
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++)
                    if (grid[(z + k) * gridSize * gridSize + (y + j) * gridSize + (x + i)]) {
                        // if the particle is close to an already stuck particle, attach it to the grid
                        // (using atomicAdd to count the overlapping particles)
                        atomicAdd(&grid[z * gridSize * gridSize + y * gridSize + x], 1);
                        return;
                    }

        // calculate the random direction of the particle
        int dir = curand(&localState) % 26;

        // move the particle in the random direction
        move_particle(&x, &y, &z, dir);

        // increment the counter for the number of iterations
        g++;
    }

    // if the particle did more than MAX_ITER number of iterations, skip it
    // if I remove this it doesn't work, probably because of in warp divergence
    atomicAdd(skipped, 1);

    return;
}

int main(int argc, char* argv[]) {
    // command line input: grid size, number of particles, number of steps, seed coordinates, block size, random seed
    if ((argc) < 7) {
        printf("Arguments are: square grid size, number of particles times 1024, number of maximum steps, seed coordinates, the number of threads per block, seed for the curand() function.\n");
        return -1;
    }

    // get grid size from args
    int gridSize = atoi(argv[1]) | 1;

    // get number of particles sent from args
    int numParticles = atoi(argv[2]);

    // get number of iterations for each particle from args
    int maxIterations = atoi(argv[3]);

    // get seed coordinates from args
    int si = atoi(argv[4]) - 1;
    int sj = atoi(argv[5]) - 1;
    int sk = atoi(argv[6]) - 1;

    // if given out image coordinates place seed in the middle
    if (si < 0 || sj < 0 || sk < 0 || si > gridSize || sj > gridSize || sk > gridSize) {
        printf("Given outside of image seed coordinates.\n");
        printf("Setting seed coordinates to %d, %d, %d.\n", gridSize / 2, gridSize / 2, gridSize / 2);
        si = (gridSize - 1) / 2;
        sj = (gridSize - 1) / 2;
        sk = (gridSize - 1) / 2;
    }

    // get number of threads per block from args
    int blockSize;
    if (argc >= 8)
        blockSize = atoi(argv[7]);
    else
        blockSize = 1024;  // I am using a 1080, so I can use a maximum of 1024 threads per block

    // calculate the number of particles based on the number of threads per block
    numParticles *= blockSize;

    // if the random seed is given from the command line arguments
    int randomSeed;
    if (argc == 9)
        // get seed for the rand() function from args
        randomSeed = atoi(argv[8]);
    else
        // if the random seed is not given from the command line arguments, use a default value
        randomSeed = 3521;

    // calculate the number of blocks
    int blocks = (numParticles + blockSize - 1) / blockSize;

    // allocate the grid for both the host and the device
    int* grid;
    cudaMallocManaged((void**)&grid, gridSize * gridSize * gridSize * sizeof(int), cudaMemAttachGlobal);

    // initialize the grid
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++)
                grid[k * gridSize * gridSize + j * gridSize + i] = 0;

    // place the seed in si, sj
    grid[sk * gridSize * gridSize + sj * gridSize + si] = 1;

    // allocate the skipped counter for both the host and the device
    int* skipped;
    cudaMallocManaged((void**)&skipped, sizeof(int), cudaMemAttachGlobal);

    // initialize the skipped counter
    *skipped = 0;

    // allocate the array of the random states in the device memory
    curandState* d_state;
    cudaMalloc((void**)&d_state, blocks * blockSize * sizeof(curandState));

    // launch the kernel to set up the seed for each thread
    setup_kernel<<<blocks, blockSize>>>(d_state, randomSeed);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    printf("\nSimulating growth...\n");

    // time execution start
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // initialize the number of overlapping particles
    int over = 0;

    // this implementation ignores the overlapping particles and re-launches them until there are none
    do {  // launch the kernel to perform the dla algorithm
        dla_kernel<<<blocks, blockSize>>>(grid, skipped, d_state, gridSize, maxIterations);

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // get the number of overlapping particles
        over = calc_over(grid, gridSize);

        // if there are more than 1024 overlapping particles
        if (over > 1024) {
            // calculate the number of blocks
            blocks = floor(over / 1024) + 1;
            // calculate the number of threads per block
            blockSize = ceil(((float)over) / ((float)blocks));
        } else {
            // else run only one block with just overlapping threads
            blocks = 1;
            blockSize = over;
        }
        // if there are overlapping particles launch the kernel again until there are none
    } while (over > 0);

    // stop timer for execution time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Simulation finished.\n\n");

    // print the number of skipped particles
    printf("Of %d particles:\n - drawn %d,\n - skipped %d.\n\n", numParticles, numParticles - *skipped, *skipped);

    // print the time to simulate in seconds
    printf("Time to simulate: %f seconds.\n", time / 1000);

    // save the grid as a .ppm image and get the number of skipped particles
    write_matrix_to_file(grid, gridSize);

    // free the memory
    cudaFree(grid);
    cudaFree(skipped);
    cudaFree(d_state);

    return 0;
}

// calculate the number of overlapping particles
int calc_over(int* grid, int size) {
    // initialize the counter
    int tot = 0;
    // iterate over the whole grid
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            for (int k = 0; k < size; ++k)
                // if the particle is overlapping
                if (grid[k * size * size + j * size + i] > 1) {
                    // increment the counter
                    tot += grid[k * size * size + j * size + i] - 1;
                    // set the particle as stuck and not overlapping anymore
                    grid[k * size * size + j * size + i] = 1;
                }
    return tot;
}

// check if the particle is close to a stuck particle
__device__ int is_close_to_stuck(int* grid, int x, int y, int z, int size) {
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
                if (grid[(z + k) * size * size + (y + j) * size + (x + i)])
                    return 1;
}

// save the grid to a file
void write_matrix_to_file(int* matrix, int dim) {
    FILE* fp = fopen("matrix.txt", "w");
    if (fp == NULL) {
        printf("Error opening file %s\n", "matrix.txt");
        exit(1);
    }
    fprintf(fp, "%d\n", dim);

    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            for (int k = 0; k < dim; k++)
                if (matrix[(i * dim * dim) + (j * dim) + k])
                    fprintf(fp, "%d %d %d\n", i, j, k);

    fclose(fp);
}

// move the particle in the random direction (out of 26)
__device__ void move_particle(int* x, int* y, int* z, int m) {
    switch (m) {
        case 0:
            (*x)--;
            (*y)--;
            (*z)--;
            break;
        case 1:
            (*x)--;
            (*y)--;
            break;
        case 2:
            (*x)--;
            (*y)--;
            (*z)++;
            break;
        case 3:
            (*x)--;
            (*z)--;
            break;
        case 4:
            (*x)--;
            break;
        case 5:
            (*x)--;
            (*z)++;
            break;
        case 6:
            (*x)--;
            (*y)++;
            (*z)--;
            break;
        case 7:
            (*x)--;
            (*y)++;
            break;
        case 8:
            (*x)--;
            (*y)++;
            (*z)++;
            break;
        case 9:
            (*y)--;
            (*z)--;
            break;
        case 10:
            (*y)--;
            break;
        case 11:
            (*y)--;
            (*z)++;
            break;
        case 12:
            (*z)--;
            break;
        case 13:
            (*z)++;
            break;
        case 14:
            (*x)++;
            (*y)--;
            (*z)--;
            break;
        case 15:
            (*x)++;
            (*y)--;
            break;
        case 16:
            (*x)++;
            (*y)--;
            (*z)++;
            break;
        case 17:
            (*x)++;
            (*z)--;
            break;
        case 18:
            (*x)++;
            break;
        case 19:
            (*x)++;
            (*z)++;
            break;
        case 20:
            (*x)++;
            (*y)++;
            (*z)--;
            break;
        case 21:
            (*x)++;
            (*y)++;
            break;
        case 22:
            (*x)++;
            (*y)++;
            (*z)++;
            break;
        case 23:
            (*y)++;
            (*z)--;
            break;
        case 24:
            (*y)++;
            break;
        case 25:
            (*y)++;
            (*z)++;
            break;
    }
}
