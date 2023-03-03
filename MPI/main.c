// MPI implementation of a DLA simulation
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define particles 12000

void move_particle(int *x, int *y, int m);
void save_image(int *grid, int size);

int main(int argc, char **argv) {
    double start, end;
    // Simulation parameters
    int max_steps = 100000;  // Number of steps
    int grid_size = 400;     // Size of the grid
    int *my_grid;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    // Get the rank of the process
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Win win;
    MPI_Win_allocate(grid_size * grid_size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &my_grid, &win);

    // place the seed particle
    my_grid[(grid_size / 2) * grid_size + (grid_size / 2)] = 1;

    // Initialize the random number generator
    srand(my_rank);

    // divide the particles between the processes
    int my_particles = particles / process_count;
    if (my_rank == process_count - 1)
        my_particles += particles % process_count;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for (int g = 0; g < my_particles; g++) {
        // count my number of iterations
        int my_i = 0;

        int my_x, my_y;

        // generate a random particle
        // at this point the grid should be empty
        do {
            my_x = rand() % grid_size;
            my_y = rand() % grid_size;
            my_i++;
        } while (my_grid[my_x * grid_size + my_y] && my_i < max_steps);

        while (my_i < max_steps) {
            if (my_x < 1)
                my_x = 1;
            else if (my_x > grid_size - 2)
                my_x = grid_size - 2;
            if (my_y < 1)
                my_y = 1;
            else if (my_y > grid_size - 2)
                my_y = grid_size - 2;

            // if the particle is close to an already stuck particle
            if (my_grid[(my_x - 1) * grid_size + (my_y - 1)] ||  // top left
                my_grid[my_x * grid_size + (my_y - 1)] ||        // top
                my_grid[(my_x + 1) * grid_size + (my_y - 1)] ||  // top right
                my_grid[(my_x - 1) * grid_size + my_y] ||        // left
                my_grid[(my_x + 1) * grid_size + my_y] ||        // right
                my_grid[(my_x - 1) * grid_size + (my_y + 1)] ||  // bottom left
                my_grid[my_x * grid_size + (my_y + 1)] ||        // bottom
                my_grid[(my_x + 1) * grid_size + (my_y + 1)]) {  // bottom right

                // if the particle is close to an already stuck particle, attach it to the grid
                my_grid[my_x * grid_size + my_y]++;

                // put the value in the shared memory

                for (int p = 0; p < process_count; p++)
                    if (p != my_rank) {
                        int a = 1;
                        // lock
                        // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, p, 0, win);
                        MPI_Put(&my_grid[my_x * grid_size + my_y], 1, MPI_INT, p, my_x * grid_size + my_y, 1, MPI_INT, win);
                        MPI_Accumulate(&a, 1, MPI_INT, p, my_x * grid_size + my_y, 1, MPI_INT, MPI_SUM, win);
                        MPI_Win_unlock(p, win);
                    }

                break;
            }

            // move the particle in a random direction
            move_particle(&my_x, &my_y, rand() % 8);

            // increment the number of iterations for each time the move is made
            my_i++;
        }
    }
    int last_process = -1;
    // wait for all processes to finish
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0)
        printf("Execution time: %f seconds\n", end - start);

    // see who is the last process
    MPI_Reduce(&my_rank, &last_process, 1, MPI_INT, MPI_MAX, process_count - 1, MPI_COMM_WORLD);

    // if the last process, save the image
    if (my_rank == last_process)
        save_image(my_grid, grid_size);

    // free the memory
    MPI_Win_free(&win);

    // Finalize the MPI environment.
    MPI_Finalize();

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

void save_image(int *grid, int size) {
    // save image to .ppm file
    int count = 0;
    int i, j;
    FILE *fp = fopen("out.ppm", "wb"); /* b - binary mode */
    (void)fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            static unsigned char color[3];
            switch (grid[j * size + i]) {
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
                default:            // overlapping
                    color[0] = 255; /* red */
                    color[1] = 0;   /* green */
                    color[2] = 0;   /* blue */
                    count += grid[j * size + i];
                    break;
            }
            // seed
            if (j == size / 2 && i == size / 2) {
                color[0] = 0;   /* red */
                color[1] = 0;   /* green */
                color[2] = 255; /* blue */
            }
            (void)fwrite(color, 1, 3, fp);
        }
    }
    (void)fclose(fp);

    printf("Number of stuck particles: %d\n", count);
}