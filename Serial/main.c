#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define IMG_NAME "out.ppm"
/*
TODO generate a new particle only on the edge of the close range
*/

int gridSize;

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

int bound(int a) {
    if (a < 0)
        return 0;
    if (a >= gridSize)
        return gridSize - 1;
    return a;
}

int boundI(int a) {
    if (a <= 0 || a >= gridSize - 1)
        return -1;
    return a;
}

// Implementing Mid-Point Circle Drawing Algorithm
void drawCircle(int **grid, int i, int j, int radius, int color) {
    int x = radius, y = 0;

    // Printing the initial point on the axes
    // after translation

    grid[bound(x + i)][bound(y + j)] = color;
    grid[bound(i)][bound(-x + j)] = color;
    grid[bound(-x + i)][bound(j)] = color;

    // When radius is zero only a single
    // point will be printed
    if (radius > 0) {
        grid[bound(x + i)][bound(-y + j)] = color;
        grid[bound(y + i)][bound(x + j)] = color;
        grid[bound(-y + i)][bound(x + j)] = color;
    }

    // Initialising the value of P
    int P = 1 - radius;
    while (x > y) {
        y++;

        // Mid-point is inside or on the perimeter
        if (P <= 0)
            P = P + 2 * y + 1;

        // Mid-point is outside the perimeter
        else {
            x--;
            P = P + 2 * y - 2 * x + 1;
        }

        // All the perimeter points have already been printed
        if (x < y)
            break;

        // Printing the generated point and its reflection
        // in the other octants after translation
        grid[bound(x + i)][bound(y + j)] = color;
        grid[bound(-x + i)][bound(y + j)] = color;
        grid[bound(x + i)][bound(-y + j)] = color;
        grid[bound(-x + i)][bound(-y + j)] = color;

        // If the generated point is on the line x = y then
        // the perimeter points have already been printed
        if (x != y) {
            grid[bound(y + i)][bound(x + j)] = color;
            grid[bound(-y + i)][bound(x + j)] = color;
            grid[bound(y + i)][bound(-x + j)] = color;
            grid[bound(-y + i)][bound(-x + j)] = color;
        }
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
                case 2:             // seed
                    color[0] = 255; /* red */
                    color[1] = 0;   /* green */
                    color[2] = 0;   /* blue */
                    break;
                case 3:             // circle close to stuck structure in
                    color[0] = 0;   /* red */
                    color[1] = 0;   /* green */
                    color[2] = 255; /* blue */
                    break;
                case 4:            // circle close to stuck structure out
                    color[0] = 0;  /* red */
                    color[1] = 96; /* green */
                    color[2] = 0;  /* blue */
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
}

void generatePointInCircle(int *i, int *j, int ci, int cj, int radius) {
    // allocte the array with the circle coordinates inside
    int size = 2 * ceil(3.14 * radius * 2);
    int *circle = malloc(sizeof(int) * size);

    for (int k = 0; k < size; k++)
        circle[k] = -1;

    int g = 0;

    int x = radius, y = 0;

    // Printing the initial point on the axes
    // after translation
    circle[g] = boundI(x + ci);
    circle[++g] = boundI(y + cj);

    circle[++g] = boundI(ci);
    circle[++g] = boundI(-x + cj);

    circle[++g] = boundI(-x + ci);
    circle[++g] = boundI(cj);

    // When radius is zero only a single
    // point will be printed
    if (radius > 0) {
        circle[++g] = boundI(x + ci);
        circle[++g] = boundI(-y + cj);

        circle[++g] = boundI(y + ci);
        circle[++g] = boundI(x + cj);

        circle[++g] = boundI(-y + ci);
        circle[++g] = boundI(x + cj);
    }

    // Initialising the value of P
    int P = 1 - radius;
    while (x > y) {
        y++;

        // Mid-point is inside or on the perimeter
        if (P <= 0)
            P = P + 2 * y + 1;

        // Mid-point is outside the perimeter
        else {
            x--;
            P = P + 2 * y - 2 * x + 1;
        }

        // All the perimeter points have already been printed
        if (x < y)
            break;

        // Printing the generated point and its reflection
        // in the other octants after translation
        circle[++g] = boundI(x + ci);
        circle[++g] = boundI(y + cj);

        circle[++g] = boundI(-x + ci);
        circle[++g] = boundI(y + cj);

        circle[++g] = boundI(x + ci);
        circle[++g] = boundI(-y + cj);

        circle[++g] = boundI(-x + ci);
        circle[++g] = boundI(-y + cj);

        // If the generated point is on the line x = y then
        // the perimeter points have already been printed
        if (x != y) {
            circle[++g] = boundI(y + ci);
            circle[++g] = boundI(x + cj);

            circle[++g] = boundI(-y + ci);
            circle[++g] = boundI(x + cj);

            circle[++g] = boundI(y + ci);
            circle[++g] = boundI(-x + cj);

            circle[++g] = boundI(-y + ci);
            circle[++g] = boundI(-x + cj);
        }
    }
    // printf("\n");
    // for (int l = 0; l < size; l += 2)
    //     printf("%d, %d, %d\n", l, circle[l], circle[l + 1]);
    // printf("\n");

    // return;
    int cont = 0;
    int point = rand() % size;
    if (point % 2 != 0)
        point--;
    while ((circle[point] == -1 || circle[point + 1] == -1) && cont < size * 3) {
        // printf("%d\n", point);
        cont++;
        point = rand() % size;
        if (point % 2 != 0)
            point--;
    }
    if (circle[point] == -1 || circle[point + 1] == -1) {
        // *i = rand() % 2 == 0 ? 1 : gridSize - 1;
        // *j = rand() % 2 == 0 ? 1 : gridSize - 1;

        // printf("E\n");
        *i = rand() % (gridSize - 1);
        *j = rand() % (gridSize - 1);
    } else {
        *i = circle[point];
        *j = circle[point + 1];
    }
    // printf("P %d, %d\n", point, size);
    free(circle);
    // printf("G %d, %d, %d\n", *i, *j, point);
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
    gridSize = atoi(argv[1]);

    // get number of particles sent from args
    int particles = atoi(argv[2]);

    // get number of iterations for each particle from args
    int iterations = atoi(argv[3]);

    // get seed coordinates from args
    int si = atoi(argv[4]) - 1;
    int sj = atoi(argv[5]) - 1;

    // get close radius from args
    int closeRadius = atoi(argv[6]) - 1;
    if ((closeRadius < 0) || (closeRadius >= gridSize))
        closeRadius = gridSize / 5;  // I chose this value randomly

    // if the random seed is given from the command line arguments
    int randomSeed;
    if (argc == 8)
        // get seed for the rand() function from args
        randomSeed = atoi(argv[7]);
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

    // draw the first circle around the center
    drawCircle(grid, si, sj, closeRadius, 3);

    // place seed
    grid[si][sj] = 2;

    // set the seed for the rand() function
    srand(randomSeed);

    // conunter for the culled particles
    int culled = 0;

    int maxDistance = 0;

    int closeRadiusT = closeRadius;
    // send particles number of particles
    for (int x = 0; x < particles; x++) {
        // generate random position at distance from center
        int i, j;
        generatePointInCircle(&i, &j, si, sj, closeRadiusT - 2);
        // printf("G %d, %d\n", i, j);

        // if the particle has been generated on an already stuck particle
        if ((grid[i][j] == 1) || (grid[i][j] == 2)) {
            x--;
            continue;
        }

        // the generated particle is not stuck
        bool stuck = false;

        // the number of moves a particle did
        int steps = 0;

        int m;

        // while the particle is not stuck and has not moved more than steps times,
        // move randomly
        while (!stuck && ((steps < iterations) || (iterations == 0))) {
            // printf("A\n");
            // increment number of steps done
            steps++;

            // temp variables for checking if the movement is ok
            int mi = i;
            int mj = j;

            // move particle with a random move
            m = rand() % (8);
            moveParticle(&mi, &mj, m);

            // change the move if it is not ok until it is
            // printf("O %d, %d, %d\n", i, j, m);
            // printf("M %d, %d, %d\n", mi, mj, m);

            while (true) {
                // if the moved particle doesn't go outside of the image
                // printf("O2 %d, %d, %d\n", i, j, m);
                // printf("M2 %d, %d, %d\n", mi, mj, m);
                // if (sqrt(((j - sj) * (j - sj)) + ((i - si) * (i - si))) >= closeRadiusT ||
                //     mi < 0 || mj < 0) {
                //     printf("O3 %d, %d, %d\n", i, j, m);
                //     printf("M3 %d, %d, %d\n", mi, mj, m);
                //     grid[mi][mj] = 2;
                //     grid[i][j] = 4;
                //     saveImage(grid, gridSize);
                //     return -1;
                // }
                if (!((mi < 0) || (mj < 0) || (mi >= gridSize) || (mj >= gridSize))) {
                    // printf("B\n");
                    // if the moved particle doesn't go on the circle
                    if (!(grid[mi][mj] == 3)) {
                        // printf("B1\n");

                        // 0 2 5 7
                        // if the moved particle doesn't go through the circle
                        if (!(m == 0 && grid[i - 1][j] == 3 && grid[i][j - 1] == 3) &&
                            !(m == 5 && grid[i + 1][j] == 3 && grid[i][j - 1] == 3) &&
                            !(m == 2 && grid[i - 1][j] == 3 && grid[i][j + 1] == 3) &&
                            !(m == 7 && grid[i + 1][j] == 3 && grid[i][j + 1] == 3)) {
                            // printf("B2\n");
                            break;
                        }
                    }
                }
                // try moving the particle with a different move
                mi = i;
                mj = j;
                m = rand() % (8);
                moveParticle(&mi, &mj, m);
            }
            // printf("C\n");

            // move the particle
            i = mi;
            j = mj;

            // if the particle is close to a stuck particle it becomes stuck
            for (int g = -1; g <= 1; g++) {
                for (int k = -1; k <= 1; k++) {
                    if (i + g < 0 || i + g >= gridSize || j + k < 0 || j + k >= gridSize)
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
            // delete the old circle
            drawCircle(grid, si, sj, closeRadiusT, 0);
            // printf("B %d, %d,\n", i, j);

            // increment radius
            float d = sqrt((j - sj) * (j - sj) + (i - si) * (i - si));
            int dist = ceil(d);
            // printf("D %d, %d, %d, %d\n", dist, maxDistance, closeRadiusT, closeRadius);
            if (dist > maxDistance) {
                closeRadiusT = dist + closeRadius;
                maxDistance = dist;
            }

            // draw the update circle
            drawCircle(grid, si, sj, closeRadiusT, 3);

            // place the stuck particle in the grid
            grid[i][j] = 1;

            // if (x % 20 == 0)
            //     saveImage(grid, gridSize);

            // sleep(1);

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
    saveImage(grid, gridSize);

    // print CPU time of main function
    printf("CPU time in seconds: %f\n",
           (double)(end - start) / (CLOCKS_PER_SEC));

    printf("Saved image as %s\n", IMG_NAME);

    return 0;
}
