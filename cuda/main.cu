// cuda implementation of the dla algorithm

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define gridSize 301
#define MAX_ITERATIONS 9999999
#define IMG_NAME "out.ppm"
#define MAX_ITER 100000
#define BLOCKS 30              // number of blocks
#define THREADS_PER_BLOCK 128  // number of threads per block

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int m);

// save the grid as a .ppm image
void saveImage(int* grid, int size);

// kernel to set up the seed for each thread
__global__ void setup_kernel(curandState* state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

// kernel to perform the dla algorithm
__global__ void dla_kernel(int* grid, int* skipped, curandState* state) {
    // calculate thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // initialize the starting position of the particle
    int x;
    int y;

    // if the particle has been generated on a stuck particle, generate a new position
    do {
        x = curand(&state[id]) % gridSize;
        y = curand(&state[id]) % gridSize;
    } while (grid[y * gridSize + x] == 1);

    // print the starting position of the particle
    printf("Thread (%d, %d) started (%d, %d)\n", blockIdx.x, id, x, y);

    // iterate until the particle is attached to the grid or it did more than MAX_ITER number of iterations
    for (int i = 0; i < MAX_ITER; i++) {
        // if the particle not outside the grid &&
        // if the particle is close to an already stuck particle
        if (!(y < 0 || y > (gridSize - 1) || x < 0 || x > (gridSize - 1)) &&  // in bounds
            (grid[(y - 1) * gridSize + (x - 1)] == 1 ||                       // top left
             grid[(y - 1) * gridSize + x] == 1 ||                             // top
             grid[(y - 1) * gridSize + (x + 1)] == 1 ||                       // top right
             grid[y * gridSize + (x - 1)] == 1 ||                             // left
             grid[y * gridSize + (x + 1)] == 1 ||                             // right
             grid[(y + 1) * gridSize + (x - 1)] == 1 ||                       // bottom left
             grid[(y + 1) * gridSize + x] == 1 ||                             // bottom
             grid[(y + 1) * gridSize + (x + 1)] == 1)) {                      // bottom right

            // if the particle is close to an already stuck particle, attach it to the grid
            grid[y * gridSize + x] = 1;
            printf("Thread (%d, %d) finished (%d, %d)\n", blockIdx.x, id, x, y);
            return;
        }

        // create temp variables to check if the move is ok
        int tempX;
        int tempY;

        // decrement the number of iterations for the do while
        i--;

        // while the move is not ok generate a new move TODO do this with a for not a while
        do {
            // save the current position
            tempX = x;
            tempY = y;

            // calculate the random direction of the particle
            int dir = curand(&state[id]) % 8;

            // move the particle in the random direction
            move_particle(&tempX, &tempY, dir);

            // increment the number of iterations for each time the move is generated
            i++;
        } while (tempY < 0 || tempY > (gridSize - 1) || tempX < 0 || tempX > (gridSize - 1));

        // move the particle
        x = tempX;
        y = tempY;
    }

    // if the particle did more than MAX_ITER number of iterations, skip it
    atomicAdd(skipped, 1);

    printf("Thread (%d, %d) skipped\n", blockIdx.x, id);

    return;
}

int main(void) {
    int* grid = (int*)malloc(gridSize * gridSize * sizeof(int));
    int* skipped;
    curandState* d_state;

    // allocate the array of the random states in the device memory
    cudaMalloc((void**)&d_state, BLOCKS * THREADS_PER_BLOCK * sizeof(curandState));

    // allocate the grid for both the host and the device
    cudaMallocManaged((void**)&grid, gridSize * gridSize * sizeof(int), cudaMemAttachGlobal);

    // allocate the skipped counter for both the host and the device
    cudaMallocManaged((void**)&skipped, sizeof(int), cudaMemAttachGlobal);

    // initialize the grid
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            grid[i * gridSize + j] = 0;

    // initialize the skipped counter
    *skipped = 0;

    // place the seed in the middle of the grid
    grid[gridSize * (gridSize / 2) + (gridSize / 2)] = 1;

    // launch the kernel to set up the seed for each thread
    setup_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_state);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // time execution start
    clock_t start = clock();

    // launch the kernel to perform the dla algorithm
    dla_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(grid, skipped, d_state);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // time execution end
    clock_t end = clock();

    // print CPU time
    printf("CPU time in seconds: %f\n", (double)(end - start) / (CLOCKS_PER_SEC));

    // print the number of skipped particles
    printf("Skipped: %d\n", *skipped);

    // save the grid as a .ppm image
    saveImage(grid, gridSize);

    // free the memory
    cudaFree(grid);
    cudaFree(skipped);
    cudaFree(d_state);

    return 0;
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
    int count = 0;
    // save image to .ppm file
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
                case 2:             // seed
                    color[0] = 255; /* red */
                    color[1] = 0;   /* green */
                    color[2] = 0;   /* blue */
                    break;
                // case 3:             // circle close to stuck structure
                //     color[0] = 0;   /* red */
                //     color[1] = 0;   /* green */
                //     color[2] = 255; /* blue */
                //     break;
                default:          // empty spots
                    color[0] = 0; /* red */
                    color[1] = 0; /* green */
                    color[2] = 0; /* blue */
                    break;
            }
            (void)fwrite(color, 1, 3, fp);
        }
    }
    (void)fclose(fp);

    printf("Saved image containing %d particles\n", count);
}
