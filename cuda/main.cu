// cuda implementation of the dla algorithm

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define IMG_NAME "out.ppm"

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int m);

// atomi CAS for bool
static __inline__ __device__ bool atomicCAS(bool* address, bool compare, bool val);

// save the grid as a .ppm image
int saveImage(bool* grid, int size);

// kernel to set up the seed for each thread
__global__ void setup_kernel(curandState* state, int randomSeed) {
    // calculate thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(randomSeed, id, 0, &state[id]);
}

// kernel to perform the diffusion
__global__ void dla_kernel(bool* grid, curandState* state, int gridSize, int maxIterations) {
    // calculate thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // copy the random state to the local memory
    curandState localState = state[id];

    // initialize the starting position of the particle
    int x;
    int y;

    // initialize the counter for the number of iterations
    int i = 0;

    // if the particle has been generated on a stuck particle, generate a new
    // position
    do {
        x = curand(&localState) % gridSize;
        y = curand(&localState) % gridSize;
    } while (grid[x * gridSize + y] && i++ < maxIterations);

    // iterate until the particle is attached to the grid or it did more than
    // maxIterations number of iterations
    while (i++ < maxIterations) {
        // if the particle is outside the grid, move it back insid
        x = min(max(x, 1), gridSize - 2);
        y = min(max(y, 1), gridSize - 2);

        // if the particle is close to an already stuck particle
        if (grid[(x - 1) * gridSize + (y - 1)] ||  // top left
            grid[(x - 1) * gridSize + y] ||        // top
            grid[(x - 1) * gridSize + (y + 1)] ||  // top right
            grid[x * gridSize + (y - 1)] ||        // left
            grid[x * gridSize + (y + 1)] ||        // right
            grid[(x + 1) * gridSize + (y - 1)] ||  // bottom left
            grid[(x + 1) * gridSize + y] ||        // bottom
            grid[(x + 1) * gridSize + (y + 1)]) {  // bottom right

            // if the particle is close to an already stuck particle, attach it to the grid
            atomicCAS(&grid[x * gridSize + y], 0, 1);

            return;
        }

        // calculate the random direction of the particle and
        // move the particle in the random direction
        move_particle(&x, &y, curand(&localState) % 8);
    }
    // if the particle did more than MAX_ITER number of iterations, skip it

    // I have to do this because of warp divergence, if I remove this it won't work
    __syncwarp();

    return;
}

int main(int argc, char* argv[]) {
    // command line input: grid size, number of particles, number of steps, seed
    // coordinates, block size, random seed
    if ((argc) < 6) {
        printf(
            "Arguments are: square grid size, number of particles times block "
            "size, number of maximum steps, seed coordinates, the number of "
            "threads per block, seed for the curand() function.\n");
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
        blockSize = 1024;  // I am using a 1080, so I can use a maximum of 1024
                           // threads per block

    // calculate the number of particles based on the number of threads per block
    numParticles *= blockSize;

    // if the random seed is given from the command line arguments
    int randomSeed;
    if (argc == 8)
        // get seed for the rand() function from args
        randomSeed = atoi(argv[7]);
    else
        // if the random seed is not given from the command line arguments, use a
        // default value
        randomSeed = 3521;

    // calculate the number of blocks
    int blocks = (numParticles + blockSize - 1) / blockSize;

    // allocate the grid for both the host and the device
    bool* grid;
    cudaMallocManaged((void**)&grid, gridSize * gridSize * sizeof(bool), cudaMemAttachGlobal);

    // initialize the grid
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            grid[i * gridSize + j] = 0;

    // place the seed in si, sj
    grid[si * gridSize + sj] = 1;

    // allocate the array of the random states in the device memory
    curandState* d_state;
    cudaMalloc((void**)&d_state, blocks * blockSize * sizeof(curandState));

    printf("\nSimulating growth...\n");

    // time execution start
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // this implementation ignores the overlapping particles

    // launch the kernel to set up the seed for each thread
    setup_kernel<<<blocks, blockSize>>>(d_state, randomSeed);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // launch the kernel to perform the dla algorithm
    dla_kernel<<<blocks, blockSize>>>(grid, d_state, gridSize, maxIterations);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Simulation finished.\n\n");

    // save the grid as a .ppm image and get the number of stuck particles
    int stuck = saveImage(grid, gridSize);

    // print the number of skipped particles
    printf("Of %d particles:\n - drawn %d,\n - skipped %d.\n\n", numParticles,
           stuck, numParticles - stuck);

    // print the time to simulate in seconds
    printf("Execution time in seconds: %f\n", time / 1000);

    // free the memory
    cudaFree(grid);
    cudaFree(d_state);

    return 0;
}

__device__ const int dx[] = {-1, 0, 1, 1, 1, 0, -1, -1};
__device__ const int dy[] = {-1, -1, -1, 0, 1, 1, 1, 0};

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int m) {
    *x += dx[m];
    *y += dy[m];
}

// save image to .ppm file
int saveImage(bool* grid, int size) {
    int count = 0;
    int i, j;
    FILE* fp = fopen(IMG_NAME, "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i)
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[i * size + j]) {
                case 0:           // empty spots
                    color[0] = 0; /* red */
                    color[1] = 0; /* green */
                    color[2] = 0; /* blue */
                    break;
                default:            // stuck particles
                    color[0] = 255; /* red */
                    color[1] = 255; /* green */
                    color[2] = 255; /* blue */
                    count++;
                    break;
            }
            (void)fwrite(color, 1, 3, fp);
        }
    (void)fclose(fp);
    return count;
}

// copied from stackoverflow
static __inline__ __device__ bool atomicCAS(bool* address, bool compare, bool val) {
    unsigned long long addr = (unsigned long long)address;
    unsigned pos = addr & 3;             // byte position within the int
    int* int_addr = (int*)(addr - pos);  // int-aligned address
    int old = *int_addr, assumed, ival;

    bool current_value;

    do {
        current_value = (bool)(old & ((0xFFU) << (8 * pos)));

        if (current_value != compare)  // If we expected that bool to be different, then
            break;                     // stop trying to update it and just return it's current value

        assumed = old;
        if (val)
            ival = old | (1 << (8 * pos));
        else
            ival = old & (~((0xFFU) << (8 * pos)));
        old = atomicCAS(int_addr, assumed, ival);
    } while (assumed != old);

    return current_value;
}