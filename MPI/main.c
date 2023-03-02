// MPI implementation of a DLA simulation
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void move_particle(int *x, int *y, int m);
void save_image(int *grid, int size);

int main(int argc, char **argv) {
    double start, end;
    // Simulation parameters
    int max_steps = 100000;  // Number of steps
    int grid_size = 500;     // Size of the grid
    int process_count, my_rank;
    int *grid;
    int particles = 6000;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Initialize the random number generator
    srand(my_rank);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &comm);

    MPI_Win win;
    // Allocate shared memory for the grid just for the root process
    if (my_rank == 0) {  // TODO look into mpi_info info hint called "noncontig"
        MPI_Win_allocate_shared(grid_size * grid_size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &grid, &win);
        for (int i = 0; i < grid_size; i++)
            for (int j = 0; j < grid_size; j++)
                grid[i * grid_size + j] = 0;

        // place the seed particle
        grid[(grid_size / 2) * grid_size + (grid_size / 2)] = 1;
    } else
        MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &grid, &win);
    // i have to do this because MPI_Win_allocate_shared allocates memory for all processes, even if they don't need it
    // this way i can use the same code for all processes and just ignore the memory allocation for the other processes
    // but i still need to call MPI_Win_allocate_shared for all processes because it's a collective call

    // get the address of the shared memory
    int disp_unit;
    MPI_Aint size;
    MPI_Win_shared_query(win, 0, &size, &disp_unit, &grid);
    // at this point grid is a pointer to the shared memory

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
        } while (grid[my_x * grid_size + my_y] && my_i < max_steps);

        while (my_i < max_steps) {
            // make it in bounds
            if (my_x < 1)
                my_x = 1;
            else if (my_x > grid_size - 2)
                my_x = grid_size - 2;
            if (my_y < 1)
                my_y = 1;
            else if (my_y > grid_size - 2)
                my_y = grid_size - 2;

            // shared lock
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            // if the particle is close to an already stuck particle
            if (grid[(my_x - 1) * grid_size + (my_y - 1)] ||  // top left
                grid[(my_x - 1) * grid_size + my_y] ||        // top
                grid[(my_x - 1) * grid_size + (my_y + 1)] ||  // top right
                grid[my_x * grid_size + (my_y - 1)] ||        // left
                grid[my_x * grid_size + (my_y + 1)] ||        // right
                grid[(my_x + 1) * grid_size + (my_y - 1)] ||  // bottom left
                grid[(my_x + 1) * grid_size + my_y] ||        // bottom
                grid[(my_x + 1) * grid_size + (my_y + 1)]) {  // bottom right

                // unlock
                MPI_Win_unlock(0, win);
                // lock
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
                // if the particle is close to an already stuck particle, attach it to the grid
                grid[my_x * grid_size + my_y]++;

                // unlock
                MPI_Win_unlock(0, win);

                break;
            }
            // unlock
            MPI_Win_unlock(0, win);

            // move the particle in a random direction
            move_particle(&my_x, &my_y, rand() % 8);

            // increment the number of iterations for each time the move is made
            my_i++;
        }
    }
    // wait for all processes to finish
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0)
        printf("Execution time: %f seconds\n", end - start);

    // if the last process, save the image
    if (my_rank == 0)
        save_image(grid, grid_size);

    // free the shared memory
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
            switch (grid[i * size + j]) {
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
                    count += grid[i * size + j];
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