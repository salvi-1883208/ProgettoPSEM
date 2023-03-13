// cuda implementation of the dla algorithm in 3 dimensions

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// move the particle in the random direction
__device__ void move_particle(int* x, int* y, int* z, int m);

// atomi CAS for bool
static __inline__ __device__ bool atomicCAS(bool* address, bool compare,
                                            bool val);

// write the matrix to a file
int write_matrix_to_file(bool* matrix, int dim);

// kernel to set up the seed for each thread
__global__ void setup_kernel(curandState* state, int randomSeed) {
  // calculate thread id
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(randomSeed, id, 0, &state[id]);
}

// kernel to perform the dla algorithm
__global__ void dla_kernel(bool* grid, curandState* state, int gridSize,
                           int maxIterations) {
  // calculate thread id
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  // copy the random state to the local memory
  curandState localState = state[id];

  // initialize the starting position of the particle
  int x;
  int y;
  int z;

  // initialize the counter for the number of iterations
  int g = 0;

  // if the particle has been generated on a stuck particle, generate a new
  // position
  do {
    x = curand(&localState) % gridSize;
    y = curand(&localState) % gridSize;
    z = curand(&localState) % gridSize;
  } while (grid[x * gridSize * gridSize + y * gridSize + z] &&
           g <= maxIterations);

  // iterate until the particle is attached to the grid or it did more than
  // maxIterations number of iterations
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
          if (grid[(x + k) * gridSize * gridSize + (y + j) * gridSize +
                   (z + i)]) {
            // if the particle is close to an already stuck particle, attach it
            // to the grid
            atomicCAS(&grid[x * gridSize * gridSize + y * gridSize + z], 0, 1);
            return;
          }

    // calculate the random direction of the particle
    // and move the particle in the random direction
    move_particle(&x, &y, &z, curand(&localState) % 26);

    // increment the counter for the number of iterations
    g++;
  }

  // if the particle did more than MAX_ITER number of iterations, skip it

  // I have to do this because of warp divergence, if I remove this it won't
  // work
  __syncwarp();

  return;
}

int main(int argc, char* argv[]) {
  // command line input: grid size, number of particles, number of steps, seed
  // coordinates, block size, random seed
  if ((argc) < 7) {
    printf(
        "Arguments are: square grid size, number of particles times block "
        "size, number of maximum steps, seed coordinates, the number of "
        "threads per block, seed for the curand() function.\n");
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
  if (si < 0 || sj < 0 || sk < 0 || si > gridSize || sj > gridSize ||
      sk > gridSize) {
    printf("Given outside of image seed coordinates.\n");
    printf("Setting seed coordinates to %d, %d, %d.\n", gridSize / 2,
           gridSize / 2, gridSize / 2);
    si = (gridSize - 1) / 2;
    sj = (gridSize - 1) / 2;
    sk = (gridSize - 1) / 2;
  }

  // get number of threads per block from args
  int blockSize;
  if (argc >= 8)
    blockSize = atoi(argv[7]);
  else
    blockSize = 1024;  // I am using a 1080, so I can use a maximum of 1024
                       // threads per block

  // calculate the number of particles based on the number of threads per block
  numParticles *= blockSize;

  // if the random seed is given from the command line arguments
  int randomSeed;
  if (argc == 9)
    // get seed for the rand() function from args
    randomSeed = atoi(argv[8]);
  else
    // if the random seed is not given from the command line arguments, use a
    // default value
    randomSeed = 3521;

  // calculate the number of blocks
  int blocks = (numParticles + blockSize - 1) / blockSize;

  // allocate the grid for both the host and the device
  bool* grid;
  cudaMallocManaged((void**)&grid,
                    gridSize * gridSize * gridSize * sizeof(bool),
                    cudaMemAttachGlobal);

  // initialize the grid
  for (int i = 0; i < gridSize; i++)
    for (int j = 0; j < gridSize; j++)
      for (int k = 0; k < gridSize; k++)
        grid[i * gridSize * gridSize + j * gridSize + k] = 0;

  // place the seed in si, sj
  grid[si * gridSize * gridSize + sj * gridSize + sk] = 1;

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

  // launch the kernel to set up the seed for each thread
  setup_kernel<<<blocks, blockSize>>>(d_state, randomSeed);

  // wait for the kernel to finish
  cudaDeviceSynchronize();

  // launch the kernel to perform the dla algorithm
  dla_kernel<<<blocks, blockSize>>>(grid, d_state, gridSize, maxIterations);

  // wait for the kernel to finish
  cudaDeviceSynchronize();

  // stop timer for execution time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Simulation finished.\n\n");

  // save the grid as a txt file and get the number of stuck particles
  int stuck = write_matrix_to_file(grid, gridSize);

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

// save the grid to a file
int write_matrix_to_file(bool* matrix, int dim) {
  int count = 0;
  FILE* fp = fopen("matrix.txt", "w");
  if (fp == NULL) {
    printf("Error opening file %s\n", "matrix.txt");
    exit(1);
  }
  fprintf(fp, "%d\n", dim);

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
      for (int k = 0; k < dim; k++)
        if (matrix[(i * dim * dim) + (j * dim) + k]) {
          fprintf(fp, "%d %d %d\n", i, j, k);
          count++;
        }

  fclose(fp);

  return count;
}

// define the offsets for the 26 directions
__device__ const int dx[26] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
                               0,  1,  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0};
__device__ const int dy[26] = {-1, -1, -1, 0,  0, 0, 1, 1, 1, -1, -1, -1, 0,
                               0,  -1, -1, -1, 0, 0, 0, 1, 1, 1,  1,  1,  1};
__device__ const int dz[26] = {-1, 0,  1, -1, 0,  1, -1, 0,  1, -1, 0,  1, -1,
                               1,  -1, 0, 1,  -1, 0, 1,  -1, 0, 1,  -1, 0, 1};

// move the particle in the random direction (out of 26)
__device__ void move_particle(int* x, int* y, int* z, int m) {
  (*x) += dx[m];
  (*y) += dy[m];
  (*z) += dz[m];
}

// copied from stackoverflow
static __inline__ __device__ bool atomicCAS(bool* address, bool compare,
                                            bool val) {
  unsigned long long addr = (unsigned long long)address;
  unsigned pos = addr & 3;             // byte position within the int
  int* int_addr = (int*)(addr - pos);  // int-aligned address
  int old = *int_addr, assumed, ival;

  bool current_value;

  do {
    current_value = (bool)(old & ((0xFFU) << (8 * pos)));

    if (current_value !=
        compare)  // If we expected that bool to be different, then
      break;      // stop trying to update it and just return it's current value

    assumed = old;
    if (val)
      ival = old | (1 << (8 * pos));
    else
      ival = old & (~((0xFFU) << (8 * pos)));
    old = atomicCAS(int_addr, assumed, ival);
  } while (assumed != old);

  return current_value;
}