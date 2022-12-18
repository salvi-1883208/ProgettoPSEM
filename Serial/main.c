#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*
Place seed (center?)

Start firing particles one at a time

Particle moves randomly (8 possible moves)

If a particle goes too far from the stuck structure
(it is randomly placed from a distance from the structure?)
(deleted?)

If a particle comes close enough to a stuck particle it becomes stuck

Repeat (when to stop?)
*/

void printGrid(int size, int **grid) {
    // print the grid
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            printf("%d    ", grid[i][j]);
        printf("\n");
    }
    printf("\n");
}

void moveParticle(int *i, int *j, int m) {
    switch (m) {
        case 0:  // top left
            (*j)--;
            (*i)--;
            break;
        case 1:  // top
            (*i)--;
            break;
        case 2:  // top right
            (*i)--;
            (*j)++;
            break;
        case 3:  // left
            (*j)--;
            break;
        case 4:  // right
            (*j)++;
            break;
        case 5:  // bottom left
            (*j)--;
            (*i)++;
            break;
        case 6:  // bottom
            (*i)++;
            break;
        case 7:  // bottom right
            (*i)++;
            (*j)++;
            break;
    }
}

int main(int argc, char *argv[]) {
    // command line input: grid size, number of particles, number of steps, seed coordinates, close radius, random seed
    if ((argc - 1) < 6) {
        printf("Arguments are: square grid size, number of particles, number of maximum steps, seed coordinates, radius for a point to be consedered close to the stuck structure, seed for the rand() function.\n");
        return -1;
    }
    // time execution start
    clock_t start = clock();

    // get grid size from args
    int size = atoi(argv[1]);

    // get number of particles sent from args
    int particles = atoi(argv[2]);

    // get number of iterations for each particle from args
    int iterations = atoi(argv[3]);

    // get seed coordinates from args
    int si = atoi(argv[4]) - 1;
    int sj = atoi(argv[5]) - 1;

    // get close radius from args
    int closeRadius = atoi(argv[6]) - 1;
    if ((closeRadius < 0) || (closeRadius >= size))
        closeRadius = size / 10;  // I choose this value randomly

    // if the random seed is given from the command line arguments
    int randomSeed;
    if (argc == 8)
        // get seed for the rand() function from args
        randomSeed = atoi(argv[6]);
    else
        randomSeed = 3521;

    // allocate the grid
    int **grid;
    grid = (int **)malloc(sizeof(int *) * size);
    for (int i = 0; i < size; i++)
        grid[i] = (int *)malloc(sizeof(int) * size);

    // if given out image coordinates place seed in the middle
    if (si < 0 || sj < 0 || si > size || sj > size) {
        printf("Given outside of image seed coordinates.\n");
        printf("Setting seed coordinates to %d, %d.\n", size / 2, size / 2);
        si = (size - 1) / 2;
        sj = (size - 1) / 2;
    }

    // set close particles in the grid to 3
    for (int i = -closeRadius; i <= closeRadius; i++)
        for (int j = -closeRadius; j <= closeRadius; j++)
            if (!((i + si < 0) || (j + sj < 0) || (i + si >= size) || (j + sj >= size)))
                grid[si + i][sj + j] = 3;

    // place seed
    grid[si][sj] = 2;

    // set the seed for the rand() function
    srand(randomSeed);

    // conunter for the culled particles
    int culled = 0;

    // send particles number of particles
    for (int x = 0; x < particles; x++) {
        // generate random i,j position in the image
        int i = rand() % (size);
        int j = rand() % (size);

        // if the particle has been generated on an already stuck particle
        if ((grid[i][j] == 1) || (grid[i][j] == 2)) {
            // x--; // if commented skip this particle, if not generate new coordinates
            continue;
        }

        // if generated particle is not close to the stuck structure generate new coordinates
        if ((grid[i][j] != 3)) {
            x--;
            continue;
        }

        // the generated particle is not stuck
        bool stuck = false;

        // the number of moves a particle did
        int steps = 0;

        // if the particle has been generated close to a stuck one do not move it
        for (int g = -1; g <= 1; g++) {
            for (int k = -1; k <= 1; k++) {
                // don't check the neighbour cells that go outside of the image
                if (i + g < 0 || i + g >= size || j + k < 0 || j + k >= size)
                    continue;
                // if close one is stuck it becomes stuck
                if ((grid[i + g][j + k] == 1) || (grid[i + g][j + k] == 2)) {
                    stuck = true;
                    break;
                }
            }
            if (stuck)
                break;
        }

        // while the particle is not stuck and has not moved more than steps times,
        // move randomly
        while (!stuck && ((steps < iterations) || (iterations == 0))) {
            // generate move
            int m = rand() % (8);

            // increment number of steps done
            steps++;

            // temp variables for checking if the movement is ok
            int mi = i;
            int mj = j;

            // move particle
            moveParticle(&mi, &mj, m);

            // change the move if it is not ok until it is
            while (true) {
                // if the moved particle goes outside of the image
                if (!((mi < 0) || (mj < 0) || (mi >= size) || (mj >= size)))
                    // if the moved particle goes too far away from the structure
                    if (!grid[mi][mj] == 0)
                        break;
                // generate new move
                m = rand() % (8);
                mi = i;
                mj = j;
                moveParticle(&mi, &mj, m);
            }

            // move the particle
            i = mi;
            j = mj;

            // if the particle is close to a stuck particle it becomes stuck
            for (int g = -1; g <= 1; g++) {
                for (int k = -1; k <= 1; k++) {
                    if (i + g < 0 || i + g >= size || j + k < 0 || j + k >= size)
                        continue;
                    if ((grid[i + g][j + k] == 1) || (grid[i + g][j + k] == 2)) {
                        stuck = true;
                        break;
                    }
                }
                if (stuck)
                    break;
            }
        }

        // if the particle finished before doing more than iterations steps
        if ((steps < iterations) || (iterations == 0)) {
            // set close particles to 3
            for (int a = -closeRadius; a <= closeRadius; a++)
                for (int b = -closeRadius; b <= closeRadius; b++)
                    if (!((i + a < 0) || (j + b < 0) || (i + a >= size) || (j + b >= size)))
                        if ((grid[i + a][j + b] == 0))
                            grid[i + a][j + b] = 3;

            // place the stuck particle in the grid
            grid[i][j] = 1;
        }  // else the particle was culled, so increment counter
        else if (steps >= iterations)
            culled++;

        // if a particle did more than iteration steps it is culled
    }

    // print the number of culled particles
    printf("Culled %d particles.\n", culled);

    // end time
    clock_t end = clock();

    // save image to .ppm file
    int i, j;
    FILE *fp = fopen("out.ppm", "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[i][j]) {
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
                case 3:            // close to stuck structure
                    color[0] = 0;  /* red */
                    color[1] = 0;  /* green */
                    color[2] = 96; /* blue */
                    break;
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

    // printGrid(size, grid);

    // print CPU time of main function
    printf("CPU time in seconds: %f\n",
           (double)(end - start) / (CLOCKS_PER_SEC));

    return 0;
}
