#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char *argv[]) {
    // get grid size from args
    int size = atoi(argv[1]);

    // get number of particles sent from args
    int particles = atoi(argv[2]);

    // allocate the grid
    int **grid;
    grid = (int **)malloc(sizeof(int *) * size);
    for (int i = 0; i < size; i++)
        grid[i] = (int *)malloc(sizeof(int) * size);

    // place seed in the middle
    grid[(size - 1) / 2][(size - 1) / 2] = 2;

    srand(3521);  // the seed for the rand() function

    // send particles number of particles
    for (int x = 0; x < particles; x++) {
        // generate random i,j position in the image
        int i = rand() % (size);
        int j = rand() % (size);

        // if the particle has been generated on an already stuck particle
        if (grid[i][j] != 0) {
            // x--; // if commented skip this particle, if not generate new coordinates
            continue;
        }

        // the generated particle is not stuck
        bool stuck = false;

        // if the particle has been generated close to a stuck one do not move
        for (int g = -1; g <= 1; g++)
            for (int k = -1; k <= 1; k++) {
                if (i + g < 0 || i + g >= size || j + k < 0 || j + k >= size)
                    continue;
                stuck |= grid[i + g][j + k] != 0;
            }

        // while the particle is not stuck move randomly
        while (!stuck) {
            // generate move
            int m = rand() % (8);

            // if the particle is on one of the edges
            if ((i <= 0 && (m <= 3)) ||                          // top
                (i >= size - 1 && (m >= 5)) ||                   // bottom
                (j <= 0 && (m == 0 || m == 3 || m == 5)) ||      // left
                (j >= size - 1 && (m == 2 || m == 4 || m == 7))  // right
            ) {
                // teleport the particle randomly
                int i = rand() % (size);
                int j = rand() % (size);
                continue;
            }

            // move particle
            switch (m) {
                case 0:  // top left
                    j--;
                    i--;
                    break;
                case 1:  // top
                    i--;
                    break;
                case 2:  // top right
                    i--;
                    j++;
                    break;
                case 3:  // left
                    j--;
                    break;
                case 4:  // right
                    j++;
                    break;
                case 5:  // bottom left
                    j--;
                    i++;
                    break;
                case 6:  // bottom
                    i++;
                    break;
                case 7:  // bottom right
                    i++;
                    j++;
                    break;
            }

            // if the particle is close to a stuck particle it becomes stuck
            for (int g = -1; g <= 1; g++)
                for (int k = -1; k <= 1; k++) {
                    if (i + g < 0 || i + g >= size || j + k < 0 || j + k >= size)
                        continue;
                    stuck |= grid[i + g][j + k] != 0;
                }
        }
        // place the stuck particle in the grid
        grid[i][j] = 1;
    }

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
}
