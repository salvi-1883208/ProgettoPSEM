#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define gridSize 11
#define MAX_ITERATIONS 9999999
#define IMG_NAME "out.ppm"

__device__ void move_particle(int *x, int *y, int m);

void printGrid(int *grid);

void saveImage(int *grid, int size);

__global__ void setup_kernel(curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ void simulate_kernel(curandState *my_curandstate, int *grid, int *skipped) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float xf = curand_uniform(my_curandstate + idx);
    float yf = curand_uniform(my_curandstate + idx);
    int x = ((int)(xf * 100)) % (gridSize - 1);  // should be random enough (idk tho) (100 is 10 ** (number of digits in gridSize))
    int y = ((int)(yf * 100)) % (gridSize - 1);  // should be random enough (idk tho) (100 is 10 ** (number of digits in gridSize))

    // if the particle has been generated on an already stuck particle generate a new position
    while (grid[y * gridSize + x] == 1 || grid[y * gridSize + x] == 2) {
        xf = curand_uniform(my_curandstate + idx);
        yf = curand_uniform(my_curandstate + idx);
        x = ((int)(xf * 100)) % (gridSize - 1);  // should be random enough (idk tho) (100 is 10 ** (number of digits in gridSize))
        y = ((int)(yf * 100)) % (gridSize - 1);  // should be random enough (idk tho) (100 is 10 ** (number of digits in gridSize))
    }

    // while the particle is not stuck

    int iter = 3;
    printf("idx %d started x %d y %d\n", idx, x, y);
    while (iter < MAX_ITERATIONS) {
        //     printf("A %d, %d\n", x, y);
        // check if the particle is already close to a stuck particle
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++) {
                // if it is make it stuck and finish the thread
                __syncthreads();
                if ((grid[(y + j) * gridSize + (x + i)] == 2) || (grid[(y + j) * gridSize + (x + i)] == 1)) {
                    grid[y * gridSize + x] = 1;
                    __syncthreads();
                    // printf("idx %d, x %d, y %d, m %d\n", idx, x, y, m);
                    printf("idx %d finished (%d, %d)\n", idx, x, y);
                    return;
                }
            }

        // generate random move (8 directions)
        float mf = curand_uniform(my_curandstate + idx);
        int m = ((int)(mf * 10)) % 8;

        // create temp variables to check if move is ok
        int mx = x;
        int my = y;

        // move the particle using the temp variables
        move_particle(&mx, &my, m);

        //     printf("B %d, %d, %d\n", mx, my, m);

        // if the moved particle would go outside of the image generate a new move until it is ok
        while (mx < 0 || my < 0 || mx >= gridSize || my >= gridSize) {
            // printf("idx %d, x %d, y %d, mx %d, my %d, m %d\n", idx, x, y, mx, my, m);
            // reset the moved particle
            mx = x;
            my = y;
            // generate a new move
            mf = curand_uniform(my_curandstate + idx);
            m = ((int)(mf * 10)) % 8;
            // move the particle
            move_particle(&mx, &my, m);
        }

        // the movement is ok
        x = mx;
        y = my;

        iter++;
    }

    skipped++;

    printf("idx %d skipped\n", idx);
}

int main(void) {
    int *grid;
    int *skipped;
    curandState *d_state;

    // allocate the array of the random states in the device
    cudaMalloc(&d_state, sizeof(curandState) * 2);

    // allocate the grid both the host and the device TODO check if it is better to do manually
    cudaMallocManaged((void **)&grid, sizeof(int) * gridSize * gridSize, cudaMemAttachGlobal);
    // already all 0;

    cudaMallocManaged((void **)&skipped, sizeof(int), cudaMemAttachGlobal);

    // place the seed in the middle
    grid[(gridSize / 2) * gridSize + (gridSize / 2)] = 2;

    // launch the kernel to set up the seed for each thread
    setup_kernel<<<1, 32>>>(d_state);

    // time execution start
    clock_t start = clock();

    // launch the kernel to move a random particle
    simulate_kernel<<<1, 32>>>(d_state, grid, skipped);

    // synchronize after all the threads are done
    cudaDeviceSynchronize();

    // end time
    clock_t end = clock();

    printGrid(grid);
    printf("%d\n", *skipped);

    // print CPU time of main function
    printf("CPU time in seconds: %f\n",
           (double)(end - start) / (CLOCKS_PER_SEC));

    saveImage(grid, gridSize);

    return 0;
}

__device__ void move_particle(int *x, int *y, int m) {
    switch (m) {
        case 0:  // top left
            (*x)--;
            (*y)--;
            break;
        case 1:  // top
            (*y)--;
            break;
        case 2:  // top right
            (*y)--;
            (*x)++;
            break;
        case 3:  // left
            (*x)--;
            break;
        case 4:  // right
            (*x)++;
            break;
        case 5:  // bottom left
            (*x)--;
            (*y)++;
            break;
        case 6:  // bottom
            (*y)++;
            break;
        case 7:  // bottom right
            (*x)++;
            (*y)++;
            break;
    }
}

void printGrid(int *grid) {
    int row, columns;
    for (row = 0; row < gridSize; row++) {
        for (columns = 0; columns < gridSize; columns++)
            printf("%d ", grid[row * gridSize + columns]);
        printf("\n");
    }
}

void saveImage(int *grid, int size) {
    // save image to .ppm file
    int i, j;
    FILE *fp = fopen(IMG_NAME, "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[j * gridSize + i]) {
                case 1:             // stuck particles
                    color[0] = 255; /* red */
                    color[1] = 255; /* green */
                    color[2] = 255; /* blue */
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
                case 0:           // circle close to stuck structure
                    color[0] = 0; /* red */
                    color[1] = 0; /* green */
                    color[2] = 0; /* blue */
                    break;
                default:            // empty spots
                    color[0] = 0;   /* red */
                    color[1] = 255; /* green */
                    color[2] = 0;   /* blue */
                    break;
            }
            (void)fwrite(color, 1, 3, fp);
        }
    }
    (void)fclose(fp);
}