#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define IMG_NAME "out.ppm"

int gridSize;

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

// bound a particle inside the grid
int boundI(int a) {
    if (a <= 0 || a >= gridSize - 1)
        return -1;
    return a;
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
                case 3:             // circle close to stuck structure
                    color[0] = 0;   /* red */
                    color[1] = 0;   /* green */
                    color[2] = 255; /* blue */
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

// generate an array of points of a circle with ci, cj as center and radius radius
int generateCircleFromCenter(int **out, int ci, int cj, int radius) {
    // allocte the array with the circle coordinates inside
    int size = 2 * ceil(3.14 * radius * 2);
    int *circle = malloc(sizeof(int) * size);

    // initialize the array to a specific impossible value
    for (int k = 0; k < size; k++)
        circle[k] = -1;

    // counter for populating the array
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
    *out = circle;

    return size;
}

void generatePointInCircle(int *i, int *j, int ci, int cj, int radius) {
    // get the array of points of the circle and its size
    int *circle = NULL;
    int size = generateCircleFromCenter(&circle, ci, cj, radius);

    // counter generate a point size * 3 times if not ok
    int cont = 0;

    // generate a point from the arrray
    int point = rand() % size;

    // the number generated needs to point to an even value wich contains the x coordinate
    if (point % 2 != 0)
        point--;

    // if the generated point goes outside of the image or is from an index wich
    // was not populated (the array is slightly bigger than the radius)
    // generate a new point
    while ((circle[point] == -1 || circle[point + 1] == -1) && cont < size * 3) {
        cont++;
        point = rand() % size;
        if (point % 2 != 0)
            point--;
    }

    // if the point generated goes outside of the image
    if (circle[point] == -1 || circle[point + 1] == -1) {
        // generate a random point in the grid
        *i = rand() % (gridSize - 1);
        *j = rand() % (gridSize - 1);
    } else {
        // if not output it
        *i = circle[point];
        *j = circle[point + 1];
    }

    // de-allocate the array
    free(circle);
}

void drawCircleFromCenter(int **grid, int ci, int cj, int radius, int color) {
    // get the array of points of the circle and its size
    int *circle = NULL;
    int size = generateCircleFromCenter(&circle, ci, cj, radius);

    // draw the circle
    for (int i = 0; i < size; i += 2)
        if (circle[i] != -1 && circle[i + 1] != -1)
            grid[circle[i]][circle[i + 1]] = color;

    // de-allocate the array
    free(circle);
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
    drawCircleFromCenter(grid, si, sj, closeRadius, 3);

    // place seed
    grid[si][sj] = 2;

    // set the seed for the rand() function
    srand(randomSeed);

    // conunter for the culled particles
    int culled = 0;

    // counter for the max distance from a particle of the structure and the original seed
    int maxDistance = 0;

    // counters for the average step and max step
    int avg = 0;
    int maxStep = 0;

    // temp variable used to increment the radius during the execution
    int closeRadiusT = closeRadius;

    printf("\nSimulating growth...\n");

    // send particles number of particles
    for (int x = 0; x < particles; x++) {
        // generate random position at distance from center
        int i, j;
        generatePointInCircle(&i, &j, si, sj, closeRadiusT - 2);

        // if the particle has been generated on an already stuck particle
        if ((grid[i][j] == 1) || (grid[i][j] == 2)) {
            x--;
            continue;
        }

        // the generated particle is not stuck
        bool stuck = false;

        // the number of moves a particle did
        int steps = 0;

        // the move to do
        int m;

        // while the particle is not stuck and has not moved more than steps times,
        // move randomly
        while (!stuck && ((steps < iterations) || (iterations == 0))) {
            // increment number of steps done
            steps++;

            // temp variables for checking if the movement is ok
            int mi = i;
            int mj = j;

            // move particle with a random move
            m = rand() % (8);
            moveParticle(&mi, &mj, m);

            // change the move if it is not ok until it is
            while (true) {
                if (!((mi < 0) || (mj < 0) || (mi >= gridSize) || (mj >= gridSize)))
                    // if the moved particle doesn't go on the circle
                    if (!(grid[mi][mj] == 3))
                        // if the moved particle doesn't go through the circle
                        if (!(m == 0 && grid[i - 1][j] == 3 && grid[i][j - 1] == 3) &&
                            !(m == 5 && grid[i + 1][j] == 3 && grid[i][j - 1] == 3) &&
                            !(m == 2 && grid[i - 1][j] == 3 && grid[i][j + 1] == 3) &&
                            !(m == 7 && grid[i + 1][j] == 3 && grid[i][j + 1] == 3)) break;

                // try moving the particle with a different move
                mi = i;
                mj = j;
                m = rand() % (8);
                moveParticle(&mi, &mj, m);
            }

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
            drawCircleFromCenter(grid, si, sj, closeRadiusT, 0);

            // increment radius
            float d = sqrt((j - sj) * (j - sj) + (i - si) * (i - si));
            int dist = ceil(d);
            if (dist > maxDistance) {
                closeRadiusT = dist + closeRadius;
                maxDistance = dist;
            }

            // draw the updated circle
            drawCircleFromCenter(grid, si, sj, closeRadiusT, 3);

            // place the stuck particle in the grid
            grid[i][j] = 1;

            // calculate avg and max steps
            avg += steps;
            if (steps > maxStep)
                maxStep = steps;
        }  // else the particle was culled, so increment counter
        else if (steps >= iterations)
            culled++;

        // if a particle did more than iteration steps it is culled
    }

    printf("Simulation finished.\n\n");

    // print the number of culled particles
    printf("Of %d particles:\n - drawn %d,\n - culled %d.\n\n", particles, particles - culled, culled);

    avg = round(avg / (particles - culled));
    printf("Average particle steps %d.\n", avg);
    printf("Max particle steps %d.\n", maxStep);

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
