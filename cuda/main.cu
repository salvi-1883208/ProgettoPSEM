// cuda implementation of the dla algorithm

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define gridSize 701
#define IMG_NAME "out.ppm"
#define MAX_ITER 100000
#define BLOCKS 16               // number of blocks
#define THREADS_PER_BLOCK 1024  // number of threads per block

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int m);

// calculate the number of overlapping particles
int calc_over(int* grid, int size);

// save the grid as a .ppm image
int saveImage(int* grid, int size);

// calculate the thread id
__device__ int calc_id() {
    return (blockIdx.z * gridDim.x * gridDim.y +
            blockIdx.y * gridDim.x +
            blockIdx.x) *
               blockDim.x * blockDim.y * blockDim.z +
           threadIdx.z * blockDim.x * blockDim.y +
           threadIdx.y * blockDim.x +
           threadIdx.x;
}

// kernel to set up the seed for each thread
__global__ void setup_kernel(curandState* state) {
    int id = calc_id();
    curand_init(1234, id, 0, &state[id]);
}

// kernel to perform the dla algorithm
__global__ void dla_kernel(int* grid, int* skipped, curandState* state) {
    // calculate thread id
    int id = calc_id();

    // initialize the starting position of the particle
    int x;
    int y;

    // if the particle has been generated on a stuck particle, generate a new position
    do {
        x = curand(&state[id]) % gridSize;
        y = curand(&state[id]) % gridSize;
    } while (grid[y * gridSize + x] > 0);

    // print the starting position of the particle
    // printf("Thread (%d, %d) started (%d, %d)\n", blockIdx.x, id, x, y);

    // iterate until the particle is attached to the grid or it did more than MAX_ITER number of iterations
    for (int i = 0; i < MAX_ITER; i++) {
        // if the particle not outside the grid &&
        // if the particle is close to an already stuck particle
        if (!(y <= 0 || y >= (gridSize - 1) || x <= 0 || x >= (gridSize - 1)) &&  // in bounds
            (grid[(y - 1) * gridSize + (x - 1)] > 0 ||                            // top left
             grid[(y - 1) * gridSize + x] > 0 ||                                  // top
             grid[(y - 1) * gridSize + (x + 1)] > 0 ||                            // top right
             grid[y * gridSize + (x - 1)] > 0 ||                                  // left
             grid[y * gridSize + (x + 1)] > 0 ||                                  // right
             grid[(y + 1) * gridSize + (x - 1)] > 0 ||                            // bottom left
             grid[(y + 1) * gridSize + x] > 0 ||                                  // bottom
             grid[(y + 1) * gridSize + (x + 1)] > 0)) {                           // bottom right

            // if the particle is close to an already stuck particle, attach it to the grid
            atomicAdd(&grid[y * gridSize + x], 1);
            // printf("Thread (%d, %d) finished (%d, %d)\n", blockIdx.x, id, x, y);
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

    // printf("Thread (%d, %d) skipped\n", blockIdx.x, id);

    return;
}

int main(void) {
    int* grid = (int*)malloc(gridSize * gridSize * sizeof(int));
    int* skipped;
    curandState* d_state;
    int blocks = BLOCKS;
    int threads_per_block = THREADS_PER_BLOCK;

    // allocate the array of the random states in the device memory
    cudaMalloc((void**)&d_state, blocks * threads_per_block * sizeof(curandState));

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
    setup_kernel<<<blocks, threads_per_block>>>(d_state);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // time execution start
    clock_t start = clock();

    int over = 0;

    do {  // launch the kernel to perform the dla algorithm
        dla_kernel<<<blocks, threads_per_block>>>(grid, skipped, d_state);

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // get the number of overlapping particles
        over = calc_over(grid, gridSize);

        printf("Overlapping: %d\n", over);

        // if there are more than 1024 overlapping particles
        if (over > 1024) {
            // calculate the number of blocks
            blocks = floor(over / 1024) + 1;
            // calculate the number of threads per block
            threads_per_block = ceil(((float)over) / ((float)blocks));
        } else {
            // else run only one block with just overlapping threads
            blocks = 1;
            threads_per_block = over;
        }
        // if there are overlapping particles launch the kernel again until there are none
    } while (over > 0);

    // time execution end
    clock_t end = clock();

    // print CPU time
    printf("CPU time in seconds: %f\n", (double)(end - start) / (CLOCKS_PER_SEC));

    // print the number of skipped particles
    printf("Skipped: %d\n", *skipped);

    // save the grid as a .ppm image and get the number of skipped particles
    saveImage(grid, gridSize);

    // free the memory
    cudaFree(grid);
    cudaFree(skipped);
    cudaFree(d_state);

    return 0;
}

int calc_over(int* grid, int size) {
    int tot = 0;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (grid[i * size + j] > 1) {
                tot++;
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

int saveImage(int* grid, int size) {
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

    return count;
}
