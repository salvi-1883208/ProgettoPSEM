#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMG_NAME "out.ppm"

// save the grid as a .ppm image
void saveImage(int **grid, int size);
void move_particle(int *x, int *y, int m);

int main(int argc, char const *argv[]) {
    int gridSize = 350;
    int sx = gridSize / 2;
    int sy = gridSize / 2;
    int randomSeed = 1234;
    int particles = 8192;
    int iterations = 100000;

    // allocate the grid
    int **grid;
    grid = (int **)malloc(sizeof(int *) * gridSize);
    for (int i = 0; i < gridSize; i++)
        grid[i] = (int *)malloc(sizeof(int) * gridSize);

    // place the seed
    grid[sx][sy] = 1;

    // set the seed for the rand() function
    srand(randomSeed);

    // number of particles skipped
    int skipped = 0;

    // time execution start
    clock_t start = clock();

    for (int p = 0; p < particles; p++) {
        int x;
        int y;

        // if the particle has been generated on a stuck particle, generate a new position
        do {
            x = random() % gridSize;
            y = random() % gridSize;
        } while (grid[x][y] > 0);

        // printf("HERE\n");

        int i = 0;

        while (i < iterations) {
            // if the particle not outside the grid &&
            // if the particle is close to an already stuck particle
            if ((x > 0 && x < gridSize - 1 && y > 0 && y < gridSize - 1) &&
                (grid[x - 1][y - 1] > 0 ||   // top left
                 grid[x][y - 1] > 0 ||       // top
                 grid[x + 1][y - 1] > 0 ||   // top right
                 grid[x - 1][y] > 0 ||       // left
                 grid[x + 1][y] > 0 ||       // right
                 grid[x - 1][y + 1] > 0 ||   // bottom left
                 grid[x][y + 1] > 0 ||       // bottom
                 grid[x + 1][y + 1] > 0)) {  // bottom right

                // if the particle is close to an already stuck particle, attach it to the grid
                grid[x][y]++;
                break;
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
                int dir = random() % 8;

                // move the particle in the random direction
                move_particle(&tempX, &tempY, dir);

                // increment the number of iterations for each time the move is generated
                i++;
            } while (tempY < 0 || tempY > (gridSize - 1) || tempX < 0 || tempX > (gridSize - 1));

            // move the particle
            x = tempX;
            y = tempY;

            // increment the number of iterations for each time the move is made
            i++;
        }

        // if the particle has done all the iterations, skip it
        if (i >= iterations)
            skipped++;
    }

    // time execution end
    clock_t end = clock();

    // print CPU time
    printf("CPU time in seconds: %f\n", (double)(end - start) / (CLOCKS_PER_SEC));

    // print the number of particles skipped
    printf("Skipped: %d\n", skipped);

    // save the grid to a file
    saveImage(grid, gridSize);

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
    int count = 0;
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
                    count++;
                    break;
                case 0:           // empty spots
                    color[0] = 0; /* red */
                    color[1] = 0; /* green */
                    color[2] = 0; /* blue */
                    break;
                // case 3:             // circle close to stuck structure
                //     color[0] = 0;   /* red */
                //     color[1] = 0;   /* green */
                //     color[2] = 255; /* blue */
                //     break;
                default:            // overlapping (should not happen)
                    color[0] = 255; /* red */
                    color[1] = 0;   /* green */
                    color[2] = 0;   /* blue */
                    count += grid[j][i];
                    break;
            }
            (void)fwrite(color, 1, 3, fp);
        }
    }
    (void)fclose(fp);

    printf("Saved image containing %d particles\n", count - 1);
}