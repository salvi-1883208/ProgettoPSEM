#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMG_NAME "out.ppm"

// save the grid as a .ppm image
void saveImage(int **grid, int size);
void move_particle(int *x, int *y, int m);

int main(int argc, char const *argv[]) {
    // command line input: grid size, number of particles, number of steps, seed coordinates,, random seed
    if ((argc - 1) < 5) {
        printf("Arguments are: square grid size, number of particles, number of maximum steps, seed coordinates, seed for the rand() function.\n");
        return -1;
    }

    // get grid size from args
    int gridSize = atoi(argv[1]);

    // get number of particles sent from args
    int particles = atoi(argv[2]);

    // get number of iterations for each particle from args
    int iterations = atoi(argv[3]);

    // get seed coordinates from args
    int si = atoi(argv[4]) - 1;
    int sj = atoi(argv[5]) - 1;
    
    // if the random seed is given from the command line arguments
    int randomSeed;
    if (argc == 7)
        // get seed for the rand() function from args
        randomSeed = atoi(argv[6]);
    else
        randomSeed = 3521;

    // allocate the grid
    int **grid;
    grid = (int **)malloc(sizeof(int *) * gridSize);
    for (int i = 0; i < gridSize; i++)
        grid[i] = (int *)malloc(sizeof(int) * gridSize);

    // if given out image coordinates place seed in the middle
    if (si < 0 || sj < 0 || si > gridSize || sj > gridSize) {
        printf("Given outside of image seed coordinates.\n");
        printf("Setting seed coordinates to %d, %d.\n", gridSize / 2, gridSize / 2);
        si = (gridSize - 1) / 2;
        sj = (gridSize - 1) / 2;
    }

    // place seed
    grid[si][sj] = 1;

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

        // if the particle has been generated on a stuck particle, generate a new position
        do {
            x = random() % gridSize;
            y = random() % gridSize;
        } while (grid[x][y]);

        // number of iterations
        int i = 0;

        while (i < iterations || iterations == 0) {
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
            if (grid[x - 1][y - 1] ||    // top left
                 grid[x][y - 1] ||       // top
                 grid[x + 1][y - 1] ||   // top right
                 grid[x - 1][y] ||       // left
                 grid[x + 1][y] ||       // right
                 grid[x - 1][y + 1] ||   // bottom left
                 grid[x][y + 1] ||       // bottom
                 grid[x + 1][y + 1]) {   // bottom right

                // if the particle is close to an already stuck particle, attach it to the grid
                grid[x][y]++;
                break;
            }

            // calculate the random direction of the particle
            int dir = random() % 8;

            // move the particle in the random direction
            move_particle(&x, &y, dir);

            // increment the number of iterations for each time the move is made
            i++;
        }

        // if the particle has done all the iterations, skip it
        if (i >= iterations && iterations != 0)
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
    saveImage(grid, gridSize);

    // free the grid
    for (int i = 0; i < gridSize; i++)
        free(grid[i]);
    free(grid);

    return 0;
}

// move the particle in the random direction
void move_particle(int *x, int *y, int m) {
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

void saveImage(int **grid, int size) {
    // save image to .ppm file
    int i, j;
    FILE *fp = fopen(IMG_NAME, "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[j][i]) {
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
                default:            // overlapping (should not happen)
                    color[0] = 255; /* red */
                    color[1] = 0;   /* green */
                    color[2] = 0;   /* blue */
                    break;
            }
            (void)fwrite(color, 1, 3, fp);
        }
    }
    (void)fclose(fp);
}