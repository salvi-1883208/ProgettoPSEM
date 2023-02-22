// cuda implementation of the dla algorithm

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IMG_NAME "out.ppm"

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int m);

// calculate the number of overlapping particles
int calc_over(int* grid, int size);

// save the grid as a .ppm image
void saveImage(int* grid, int size);

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

    // if the particle has been generated on a stuck particle, generate a new position
    do {
        x = curand(&localState) % gridSize;
        y = curand(&localState) % gridSize;
    } while (grid[y * gridSize + x]);

    // printf("Thread (%d, %d) started (%d, %d)\n", blockIdx.x, id, x, y);

    // initialize the counter for the number of iterations
    int i = -1;

    // iterate until the particle is attached to the grid or it did more than maxIterations number of iterations
    while ((i <= maxIterations)) {
        // if the particle is outside the grid, move it back inside
        if (x < 1)
            x = 1;
        else if (x > gridSize - 2)
            x = gridSize - 2;
        if (y < 1)
            y = 1;
        else if (y > gridSize - 2)
            y = gridSize - 2;

        // if the particle is close to an already stuck particle
        if (grid[(y - 1) * gridSize + (x - 1)] ||  // top left
            grid[(y - 1) * gridSize + x] ||        // top
            grid[(y - 1) * gridSize + (x + 1)] ||  // top right
            grid[y * gridSize + (x - 1)] ||        // left
            grid[y * gridSize + (x + 1)] ||        // right
            grid[(y + 1) * gridSize + (x - 1)] ||  // bottom left
            grid[(y + 1) * gridSize + x] ||        // bottom
            grid[(y + 1) * gridSize + (x + 1)]) {  // bottom right

            // if the particle is close to an already stuck particle, attach it to the grid
            // (using atomicAdd to count the overlapping particles)
            atomicAdd(&grid[y * gridSize + x], 1);

            // printf("Thread (%d, %d) finished (%d, %d)\n", blockIdx.x, id, x, y);

            return;
        }

        // calculate the random direction of the particle
        int dir = curand(&localState) % 8;

        // move the particle in the random direction
        move_particle(&x, &y, dir);

        // increment the number of iterations for each time the move is made
        // (I have to do it here because of warp divergence)
        if (maxIterations != 0)
            i++;
    }

    // if the particle did more than MAX_ITER number of iterations, skip it
    // if I remove this it doesn't work, probably because of in warp divergence
    atomicAdd(skipped, 1);

    // printf("Thread (%d, %d) skipped\n", blockIdx.x, id);

    return;
}

int main(int argc, char* argv[]) {
    // command line input: grid size, number of particles, number of steps, seed coordinates, block size, random seed
    if ((argc) < 6) {
        printf("Arguments are: square grid size, number of particles times 1024, number of maximum steps, seed coordinates, the number of threads per block, seed for the curand() function.\n");
        return -1;
    }

    // get grid size from args
    int gridSize = atoi(argv[1]);

    // get number of particles sent from args
    int numParticles = atoi(argv[2]);

    // get number of iterations for each particle from args
    int maxIterations = atoi(argv[3]);

    // get seed coordinates from args
    int si = atoi(argv[4]) - 1;
    int sj = atoi(argv[5]) - 1;

    // if given out image coordinates place seed in the middle
    if (si < 0 || sj < 0 || si > gridSize || sj > gridSize) {
        printf("Given outside of image seed coordinates.\n");
        printf("Setting seed coordinates to %d, %d.\n", gridSize / 2, gridSize / 2);
        si = (gridSize - 1) / 2;
        sj = (gridSize - 1) / 2;
    }

    // get number of threads per block from args
    int blockSize;
    if (argc >= 7)
        blockSize = atoi(argv[6]);
    else
        blockSize = 1024;  // I am using a 1080, so I can use a maximum of 1024 threads per block

    // calculate the number of particles based on the number of threads per block
    numParticles *= blockSize;

    // if the random seed is given from the command line arguments
    int randomSeed;
    if (argc == 8)
        // get seed for the rand() function from args
        randomSeed = atoi(argv[7]);
    else
        // if the random seed is not given from the command line arguments, use a default value
        randomSeed = 3521;

    // calculate the number of blocks
    int blocks = (numParticles + blockSize - 1) / blockSize;

    // allocate the grid for both the host and the device
    int* grid;
    cudaMallocManaged((void**)&grid, gridSize * gridSize * sizeof(int), cudaMemAttachGlobal);

    // initialize the grid
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            grid[i * gridSize + j] = 0;

    // place the seed in si, sj
    grid[sj * gridSize + si] = 1;

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

    // time execution end
    clock_t end = clock();

    printf("Simulation finished.\n\n");

    // print the number of skipped particles
    printf("Of %d particles:\n - drawn %d,\n - skipped %d.\n\n", numParticles, numParticles - *skipped, *skipped);

    // print execution time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    // print the time to simulate in seconds
    printf("Time to simulate: %f seconds.\n", time / 1000);

    // save the grid as a .ppm image and get the number of skipped particles
    saveImage(grid, gridSize);

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
            // if the particle is overlapping
            if (grid[i * size + j] > 1) {
                // increment the counter
                tot += grid[i * size + j] - 1;
                // set the particle as stuck and not overlapping anymore
                grid[i * size + j] = 1;
            }
    return tot;
}

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int m) {
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
    int i, j;
    FILE* fp = fopen(IMG_NAME, "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i)
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[j * size + i]) {
                case 1:             // stuck particles
                    color[0] = 255; /* red */
                    color[1] = 255; /* green */
                    color[2] = 255; /* blue */
                    break;
                case 0:           // empty spots
                    color[0] = 0; /* red */
                    color[1] = 0; /* green */
                    color[2] = 0; /* blue */
                    break;
                default:            // overlapping
                    color[0] = 255; /* red */
                    color[1] = 0;   /* green */
                    color[2] = 0;   /* blue */
                    break;
            }
            (void)fwrite(color, 1, 3, fp);
        }
    (void)fclose(fp);
    return;
}
