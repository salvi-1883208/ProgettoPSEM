#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void write_matrix_to_file(int ***matrix, int dim, char *filename);
void move_particle(int *x, int *y, int *z, int m);
int is_close_to_stuck(int ***grid, int x, int y, int z, int size);
void free_matrix(int ***matrix, int dim);

int main(int argc, char const *argv[]) {
    // command line input: grid size, number of particles, number of steps, seed coordinates, random seed
    if ((argc - 1) < 6) {
        printf("Arguments are: cube grid size, number of particles, number of maximum steps, seed coordinates, seed for the rand() function.\n");
        return -1;
    }

    // get grid size from args
    // make it odd using bitwise operations
    int gridSize = atoi(argv[1]) | 1;

    // get number of particles sent from args
    int particles = atoi(argv[2]);

    // get number of iterations for each particle from args
    int iterations = atoi(argv[3]);

    // get seed coordinates from args
    int si = atoi(argv[4]) - 1;
    int sj = atoi(argv[5]) - 1;
    int sk = atoi(argv[6]) - 1;

    // if the random seed is given from the command line arguments
    int randomSeed;
    if (argc == 8)
        // get seed for the rand() function from args
        randomSeed = atoi(argv[7]);
    else
        randomSeed = 3521;

    // allocate the grid
    int ***grid;
    grid = (int ***)malloc(sizeof(int **) * gridSize);
    for (int i = 0; i < gridSize; i++) {
        grid[i] = (int **)malloc(sizeof(int *) * gridSize);
        for (int j = 0; j < gridSize; j++)
            grid[i][j] = (int *)malloc(sizeof(int) * gridSize);
    }

    // if given out image coordinates place seed in the middle
    if (si < 0 || sj < 0 || sk < 0 || si > gridSize || sj > gridSize || sk > gridSize) {
        printf("Given outside of image seed coordinates.\n");
        printf("Setting seed coordinates to %d, %d, %d.\n", gridSize / 2, gridSize / 2, gridSize / 2);
        si = (gridSize - 1) / 2;
        sj = si;
        sk = si;
    }

    // place seed
    grid[si][sj][sk] = 1;

    // set the seed for the rand() function
    srand(randomSeed);

    // number of particles skipped
    int skipped = 0;

    // counters for the average step and max step
    int avgSteps = 0;
    int maxStep = 0;

    printf("\nSimulating growth...\n");

    // time execution start
    clock_t start = clock();

    // for each particle
    for (int p = 0; p < particles; p++) {
        int x;
        int y;
        int z;

        // if the particle has been generated on a stuck particle, generate a new position
        do {
            x = random() % gridSize;
            y = random() % gridSize;
            z = random() % gridSize;
        } while (grid[x][y][z]);

        // number of iterations
        int i = 0;

        while (i < iterations) {
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
            if (is_close_to_stuck(grid, x, y, z, gridSize)) {  // bottom right

                // if the particle is close to an already stuck particle, attach it to the grid
                grid[x][y][z]++;
                break;
            }

            // calculate the random direction of the particle
            int dir = random() % 26;

            // move the particle in the random direction
            move_particle(&x, &y, &z, dir);

            // increment the number of iterations for each time the move is made
            i++;
        }

        // if the particle has done all the iterations, skip it
        if (i >= iterations)
            skipped++;

        // calculate avgSteps and max steps
        avgSteps += i;
        if (i > maxStep)
            maxStep = i;
    }

    // time execution end
    clock_t end = clock();

    printf("Simulation finished.\n\n");

    // print the number of skipped particles
    printf("Of %d particles:\n - drawn %d,\n - skipped %d.\n\n", particles, particles - skipped, skipped);

    // print the average and max steps
    avgSteps = round(avgSteps / (skipped == particles ? particles : (particles - skipped)));
    printf("Average particle steps %d.\n", avgSteps);
    printf("Max particle steps %d.\n", maxStep);

    // print CPU time
    printf("CPU time in seconds: %f\n", (double)(end - start) / (CLOCKS_PER_SEC));

    // save the grid to a file
    write_matrix_to_file(grid, gridSize, "matrix.txt");

    // free the grid
    free_matrix(grid, gridSize);

    return 0;
}

// check if the particle is close to a stuck particle
int is_close_to_stuck(int ***grid, int x, int y, int z, int size) {
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
                if (grid[x + i][y + j][z + k])
                    return 1;
    return 0;
}

// define the offsets for the 26 directions
const int dx[26] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
const int dy[26] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
const int dz[26] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1};

// move the particle in the random direction (out of 26)
void move_particle(int *x, int *y, int *z, int m) {
    (*x) += dx[m];
    (*y) += dy[m];
    (*z) += dz[m];
}

// save the grid to a file
void write_matrix_to_file(int ***matrix, int dim, char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    fprintf(fp, "%d\n", dim);

    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            for (int k = 0; k < dim; k++)
                if (matrix[i][j][k])
                    fprintf(fp, "%d %d %d\n", i, j, k);

    fclose(fp);
}

// free the grid
void free_matrix(int ***matrix, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++)
            free(matrix[i][j]);
        free(matrix[i]);
    }
    free(matrix);
}
